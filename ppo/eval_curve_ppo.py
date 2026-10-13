import argparse
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PPO_ROOT.parent
CKPT_DIR = PPO_ROOT / "runs"
OUT_DIR = PROJECT_ROOT / "results"

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

cfg = _load_module("ppo_config_standalone", PPO_ROOT / "config.py").Config
ActorCritic = _load_module("ppo_model_standalone", PPO_ROOT / "model.py").ActorCritic
WEIGHT_KEY = "net"

def preprocess_frame(frame: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(frame).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr[arr > 1.0] = 255.0
    arr[arr <= 1.0] = 0.0
    return arr

def init_state(frame: np.ndarray, size: int, stack: int) -> np.ndarray:
    f = preprocess_frame(frame, size)
    return np.stack([f] * stack, axis=0)

def next_state(state: np.ndarray, frame: np.ndarray, size: int, stack: int) -> np.ndarray:
    f = preprocess_frame(frame, size)
    return np.concatenate([f[None, :, :], state[:stack - 1]], axis=0)

def make_game_env():
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT / "env"))
    import wrapped_flappy_bird as game_env
    return game_env.GameState()

def select_checkpoints(every: int, include_final: bool = True):
    if not CKPT_DIR.exists():
        return []
    pattern = re.compile(r"^ckpt-(\d+)\.pt$")
    found = []
    for p in CKPT_DIR.glob("ckpt-*.pt"):
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        return []
    found.sort(key=lambda x: x[0])
    max_step = found[-1][0]

    targets = []
    k = 1
    while True:
        t = every * k
        if t > max_step:
            break
        targets.append(t)
        k += 1

    selected = {}
    for t in targets:
        best = min(found, key=lambda fp: abs(fp[0] - t))
        selected[best[0]] = best[1]

    if include_final:
        final = found[-1]
        selected[final[0]] = final[1]

    return sorted(selected.items())

@torch.no_grad()
def evaluate_checkpoint(env, net, device, n_episodes: int,
                        size: int, stack: int, num_actions: int = 2):
    scores = []
    for _ in range(n_episodes):

        do_nothing = np.zeros(num_actions)
        do_nothing[0] = 1
        frame, _r, terminal = env.frame_step(do_nothing, render=False)
        state = init_state(frame, size, stack)

        score = 0
        while not terminal:
            s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _v = net(s)
            action = int(logits.argmax(dim=1).item())
            a_vec = np.zeros(num_actions)
            a_vec[action] = 1
            frame, reward, terminal = env.frame_step(a_vec, render=False)
            if reward == 1.0:
                score += 1
            state = next_state(state, frame, size, stack)
        scores.append(score)
    return scores

def bootstrap_ci(scores, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(scores, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

def dry_run(every: int, n_episodes: int, done_steps: set = None):
    ckpts = select_checkpoints(every)
    if not ckpts:
        print(f"[PPO] {CKPT_DIR} 下没找到 ckpt-*.pt")
        return
    todo = [(s, p) for s, p in ckpts if (done_steps is None or s not in done_steps)]
    print(f"\n[PPO] 预演（每 {every} 步一个存档，每个 {n_episodes} 局）")
    print(f"  存档目录        : {CKPT_DIR}")
    print(f"  命中存档总数    : {len(ckpts)}  (步数 {ckpts[0][0]} ~ {ckpts[-1][0]})")
    if done_steps is not None:
        print(f"  已有结果、将跳过: {len(ckpts) - len(todo)}")
    print(f"  本次将要评估    : {len(todo)}")
    if todo:
        print(f"  待评估步数      : {[s for s, _ in todo]}")
    print(f"  输出到          : {OUT_DIR / 'PPO.json'}  /  {OUT_DIR / 'curve_PPO.png'}")

def evaluate(every: int, n_episodes: int, device, env=None,
             resume: bool = False):
    ckpts = select_checkpoints(every)
    if not ckpts:
        print(f"[PPO] 没有可评估的存档，跳过")
        return None

    if env is None:
        env = make_game_env()

    net = ActorCritic(cfg.NUM_ACTIONS).to(device)
    net.eval()

    bundle = {
        "algo": "ppo",
        "episodes_per_checkpoint": n_episodes,
        "every_steps": every,
        "device": str(device),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "records": [],
    }

    done_steps = set()
    jp = OUT_DIR / "PPO.json"
    if resume and jp.exists():
        with open(jp, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("episodes_per_checkpoint") == n_episodes:
            bundle["records"] = old["records"]
            done_steps = {r["step"] for r in old["records"]}
            print(f"[PPO] 续跑：已有 {len(done_steps)} 个存档的结果，只补剩下的")
        else:
            bak = OUT_DIR / f"PPO_ep{old.get('episodes_per_checkpoint')}.json"
            with open(bak, "w", encoding="utf-8") as f:
                json.dump(old, f, ensure_ascii=False, indent=2)
            print(f"[PPO] 历史结果局数不同（{old.get('episodes_per_checkpoint')} vs "
                  f"{n_episodes}），无法续用 → 旧结果已备份到 {bak.name}，将全部重跑")

    ckpts = [(s, p) for s, p in ckpts if s not in done_steps]
    if not ckpts:
        print(f"[PPO] 所有存档都已评估过，无需重跑")
        return bundle

    print(f"\n[PPO] 共 {len(ckpts)} 个存档，每个玩 {n_episodes} 局")
    print(f"  {'步数':>10} | {'均值':>7} | {'最高':>5} | {'最低':>5} | 逐局得分")
    print("  " + "-" * 68)

    OUT_DIR.mkdir(exist_ok=True)
    for step, path in ckpts:
        ck = torch.load(str(path), map_location=device)
        net.load_state_dict(ck[WEIGHT_KEY])
        t0 = time.time()
        scores = evaluate_checkpoint(env, net, device, n_episodes,
                                     cfg.FRAME_SIZE, cfg.STACK_LEN, cfg.NUM_ACTIONS)
        rec = {
            "step": step,
            "checkpoint": str(path),
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "max": int(np.max(scores)),
            "min": int(np.min(scores)),
            "median": float(np.median(scores)),
            "eval_seconds": round(time.time() - t0, 1),
        }
        bundle["records"].append(rec)
        print(f"  {step:>10} | {rec['mean']:>7.2f} | {rec['max']:>5} | {rec['min']:>5} | {scores}",
              flush=True)

        with open(jp, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

    bundle["records"].sort(key=lambda r: r["step"])
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"\n[PPO] 结果已写入 {jp}  （共 {len(bundle['records'])} 个存档）")
    return bundle

def plot_curves(bundle: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["SimSun", "Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    recs = bundle["records"]
    if not recs:
        print("没有可绘制的数据")
        return

    color = "#C0504D"
    steps = np.array([r["step"] for r in recs], dtype=float)
    means = np.array([r["mean"] for r in recs])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8.6), dpi=200,
                             gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.32})
    ax = axes[0]

    for r in recs:
        ax.scatter([r["step"]] * len(r["scores"]), r["scores"],
                   s=9, color=color, alpha=0.18, edgecolors="none", zorder=1)

    cis = np.array([bootstrap_ci(r["scores"]) for r in recs])
    ax.fill_between(steps, cis[:, 0], cis[:, 1], color=color, alpha=0.22,
                    linewidth=0, zorder=2, label="PPO 均值 95% CI")

    ax.plot(steps, means, marker="o", markersize=3.6, linewidth=1.6,
            color=color, zorder=3, label="PPO 均值")

    k = int(np.argmax(means))
    ax.axvline(steps[k], color="#7F7F7F", linestyle=":", linewidth=1.0, zorder=0)
    ax.annotate(f"峰值 {means[k]:.1f}\n@ {steps[k] / 10000:.0f}万步",
                xy=(steps[k], means[k]), xytext=(12, -34),
                textcoords="offset points", fontsize=10, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))
    ax.plot([steps[k]], [means[k]], marker="*", markersize=15, color="#ED7D31",
            markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    ax.axhline(4.25, color="#8C8C8C", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(steps[0], 4.25, "人类均值 4.25 ", va="bottom", ha="left",
            fontsize=9, color="#666666")
    ax.axhline(16.4, color="#B07C50", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(steps[0], 16.4, "原版 DQN 16.4 ", va="bottom", ha="left",
            fontsize=9, color="#8A5A2B")

    ax2 = axes[1]
    pick = [r for r in recs if r["step"] % 200000 == 0]
    if recs and (not pick or pick[-1]["step"] != recs[-1]["step"]):
        pick = pick + [recs[-1]]
    bp = ax2.boxplot([r["scores"] for r in pick],
                     positions=range(len(pick)), widths=0.55,
                     patch_artist=True, showmeans=True,
                     medianprops=dict(color="#ED7D31", linewidth=1.6),
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="#1F4E79", markersize=4),
                     flierprops=dict(marker="o", markersize=3.5, alpha=0.45,
                                     markeredgecolor="none", color=color))
    for patch in bp["boxes"]:
        patch.set_facecolor("#F2DCDB")
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    ax2.set_xticks(range(len(pick)))
    ax2.set_xticklabels([f"{r['step'] // 10000:.0f}" for r in pick], fontsize=9)
    ax2.set_xlabel("训练步数（万步）", fontsize=11)
    ax2.set_ylabel("单局过管数", fontsize=11)
    ax2.set_title("(b) 单局得分分布（箱体=四分位距，菱形=均值，点=离群局）",
                  fontsize=11, loc="left")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.set_axisbelow(True)

    n_ep = bundle["episodes_per_checkpoint"]
    ax.set_xlabel("训练步数（万步）", fontsize=11.5)
    ax.set_ylabel("过管数（根）", fontsize=11.5)
    ax.set_title(f"(a) Flappy Bird PPO 学习曲线（每存档 {n_ep} 局，"
                 f"阴影为均值 95% 自助置信区间）", fontsize=11.5, loc="left")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 10000:.0f}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
        axes[1].spines[s].set_visible(False)

    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"\n曲线已保存到 {out_png}")

def main():
    parser = argparse.ArgumentParser(description="PPO 学习曲线：按步数批量评估存档并画图")
    parser.add_argument("--every", type=int, default=50176,
                        help="每隔多少【步】取一个存档评估（默认 50176，因 PPO 存档很密集）")
    parser.add_argument("--episodes", type=int, default=10,
                        help="每个存档连续玩几局（默认 10；建议汇报时用 30）")
    parser.add_argument("--resume", action="store_true",
                        help="续跑：跳过 PPO.json 里已经评过的存档，只补没评的")
    parser.add_argument("--plot-only", action="store_true",
                        help="不评估，只用现有 PPO.json 重新画图")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出「要跑哪些存档 / 跳过哪些」，不真正评估")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(exist_ok=True)
    print(f"使用设备: {device}")
    print(f"存档目录: {CKPT_DIR}")

    if args.plot_only:
        jp = OUT_DIR / "PPO.json"
        if not jp.exists():
            print(f"找不到 {jp}，无法画图")
            return
        with open(jp, encoding="utf-8") as f:
            bundle = json.load(f)
        plot_curves(bundle, OUT_DIR / "curve_PPO.png")
        return

    if args.dry_run:
        done_steps = None
        jp = OUT_DIR / "PPO.json"
        if jp.exists():
            with open(jp, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("episodes_per_checkpoint") == args.episodes:
                done_steps = {r["step"] for r in old["records"]}
        dry_run(args.every, args.episodes, done_steps)
        return

    env = make_game_env()
    bundle = evaluate(args.every, args.episodes, device, env=env, resume=args.resume)
    if bundle is None or not bundle["records"]:
        return

    plot_curves(bundle, OUT_DIR / "curve_PPO.png")

    print("\n" + "=" * 64)
    print("汇总（平均过管数）")
    print("=" * 64)
    print("  " + "  |  ".join(
        f"{r['step'] // 10000}万步:{r['mean']:.1f}" for r in bundle["records"]))

if __name__ == "__main__":
    main()
