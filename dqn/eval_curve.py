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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

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
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str(PROJECT_ROOT / "env"))
    import wrapped_flappy_bird as game_env
    return game_env.GameState()

def list_checkpoints(ckpt_dir: Path, every: int, include_final: bool = True):
    if not ckpt_dir.exists():
        return []
    pattern = re.compile(r"^ckpt-(\d+)\.pt$")
    found = []
    for p in ckpt_dir.glob("ckpt-*.pt"):
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        return []
    found.sort(key=lambda x: x[0])

    selected = [(s, p) for s, p in found if s % every == 0]
    if include_final:
        final = found[-1]
        if not selected or selected[-1][0] != final[0]:
            selected.append(final)
    return selected

@torch.no_grad()
def evaluate_checkpoint(env, net, algo: str, device, n_episodes: int,
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
            action = int(net(s).argmax(dim=1).item())
            a_vec = np.zeros(num_actions)
            a_vec[action] = 1
            frame, reward, terminal = env.frame_step(a_vec, render=False)
            if reward == 1.0:
                score += 1
            state = next_state(state, frame, size, stack)
        scores.append(score)
    return scores

def evaluate_algo(algo: str, every: int, n_episodes: int, device, env=None,
                  out_dir: Path = None, resume: bool = False,
                  only_steps: list = None, out_name: str = None):
    out_name = out_name or algo

    cfg = _load_module("dqn_config", PROJECT_ROOT / "dqn" / "config.py").Config
    net_cls = _load_module("dqn_model", PROJECT_ROOT / "dqn" / "model.py").QNet
    weight_key = "policy"

    ckpt_dir = PROJECT_ROOT / "runs"

    if only_steps:
        available = {}
        for p in ckpt_dir.glob("ckpt-*.pt"):
            m = re.match(r"^ckpt-(\d+)\.pt$", p.name)
            if m:
                available[int(m.group(1))] = p
        ckpts = []
        for s in only_steps:
            if s in available:
                ckpts.append((s, available[s]))
            else:
                print(f"  ⚠️ 找不到 ckpt-{s}.pt，已跳过")
        ckpts.sort()
    else:
        ckpts = list_checkpoints(ckpt_dir, every)

    if not ckpts:
        print(f"[{algo.upper()}] 没有可评估的存档，跳过")
        return None

    if env is None:
        env = make_game_env()

    net = net_cls(cfg.NUM_ACTIONS).to(device)
    net.eval()

    print(f"\n[{algo.upper()}] 共 {len(ckpts)} 个存档，每个玩 {n_episodes} 局")
    print(f"  {'步数':>10} | {'均值':>7} | {'最高':>5} | {'最低':>5} | 逐局得分")
    print("  " + "-" * 68)

    bundle = {
        "algo": algo,
        "episodes_per_checkpoint": n_episodes,
        "every_steps": every,
        "device": str(device),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "records": [],
    }

    done_steps = set()
    if resume and out_dir is not None:
        jp = out_dir / f"{out_name}.json"
        if jp.exists():
            with open(jp, encoding="utf-8") as f:
                old = json.load(f)

            if old.get("episodes_per_checkpoint") == n_episodes:
                bundle["records"] = old["records"]
                done_steps = {r["step"] for r in old["records"]}
                print(f"[{algo.upper()}] 续跑：已有 {len(done_steps)} 个存档的结果，只补剩下的")
            else:

                bak = out_dir / f"{out_name}_ep{old.get('episodes_per_checkpoint')}.json"
                with open(bak, "w", encoding="utf-8") as f:
                    json.dump(old, f, ensure_ascii=False, indent=2)
                print(f"[{algo.upper()}] 历史结果局数不同（{old.get('episodes_per_checkpoint')} vs "
                      f"{n_episodes}），无法续用 → 旧结果已备份到 {bak.name}，将全部重跑")

    ckpts = [(s, p) for s, p in ckpts if s not in done_steps]
    if not ckpts:
        print(f"[{algo.upper()}] 所有存档都已评估过，无需重跑")
        return bundle

    for step, path in ckpts:
        ck = torch.load(str(path), map_location=device)
        net.load_state_dict(ck[weight_key])
        t0 = time.time()
        scores = evaluate_checkpoint(env, net, algo, device, n_episodes,
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

        if out_dir is not None:
            with open(out_dir / f"{out_name}.json", "w", encoding="utf-8") as f:
                json.dump(bundle, f, ensure_ascii=False, indent=2)

    bundle["records"].sort(key=lambda r: r["step"])
    if out_dir is not None:
        with open(out_dir / f"{out_name}.json", "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)
    return bundle

def dry_run(algo: str, every: int, n_episodes: int, out_dir: Path):

    ckpt_dir = PROJECT_ROOT / "runs"
    all_ckpts = list_checkpoints(ckpt_dir, every)
    if not all_ckpts:
        print(f"[{algo.upper()}] {ckpt_dir} 下没找到 ckpt-*.pt")
        return

    done_steps, old_n = set(), None
    jp = out_dir / f"{algo}.json"
    if jp.exists():
        with open(jp, encoding="utf-8") as f:
            old = json.load(f)
        old_n = old.get("episodes_per_checkpoint")
        if old_n == n_episodes:
            done_steps = {r["step"] for r in old["records"]}

    todo = [(s, p) for s, p in all_ckpts if s not in done_steps]
    print(f"\n[{algo.upper()}] 预演（每 {every} 步一个存档，每个 {n_episodes} 局）")
    print(f"  存档目录        : {ckpt_dir}")
    print(f"  命中存档总数    : {len(all_ckpts)}  (步数 {all_ckpts[0][0]} ~ {all_ckpts[-1][0]})")
    print(f"  已有结果、将跳过: {len(all_ckpts) - len(todo)}")
    print(f"  本次将要评估    : {len(todo)}")
    if todo:
        print(f"  待评估步数      : {[s for s, _ in todo]}")
    if old_n is not None and old_n != n_episodes:
        print(f"  ⚠️ 历史结果每存档 {old_n} 局，与本次 {n_episodes} 局不一致，"
              f"无法续用，将全部重跑（旧结果会先备份）")
    print(f"  输出到          : {jp}")

def bootstrap_ci(scores, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(scores, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

def plot_curves(results: list, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["SimSun", "Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    results = [r for r in results if r and r["records"]]
    if not results:
        print("没有可绘制的数据")
        return

    colors = {"dqn": "#1F4E79"}
    single = len(results) == 1

    if single:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8.6), dpi=200,
                                 gridspec_kw={"height_ratios": [2.1, 1],
                                              "hspace": 0.32})
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5.4), dpi=200)
        axes = [axes]

    ax = axes[0]
    for res in results:
        recs = res["records"]
        steps = np.array([r["step"] for r in recs], dtype=float)
        means = np.array([r["mean"] for r in recs])
        color = colors.get(res["algo"], "#5F5E5A")
        label = res["algo"].upper()

        for r in recs:
            ax.scatter([r["step"]] * len(r["scores"]), r["scores"],
                       s=9, color=color, alpha=0.18, edgecolors="none", zorder=1)

        cis = np.array([bootstrap_ci(r["scores"]) for r in recs])
        ax.fill_between(steps, cis[:, 0], cis[:, 1], color=color, alpha=0.22,
                        linewidth=0, zorder=2,
                        label=f"{label} 均值 95% CI" if single else f"{label} 95% CI")

        ax.plot(steps, means, marker="o", markersize=3.6, linewidth=1.6,
                color=color, zorder=3, label=f"{label} 均值")
        if not single:
            continue

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
            patch.set_facecolor("#DCE6F1")
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

    n_ep = results[0]["episodes_per_checkpoint"]
    ax.set_xlabel("训练步数（万步）", fontsize=11.5)
    ax.set_ylabel("过管数（根）", fontsize=11.5)
    ax.set_title(f"(a) Flappy Bird 学习曲线（每存档 {n_ep} 局，"
                 f"阴影为均值 95% 自助置信区间）", fontsize=11.5, loc="left")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 10000:.0f}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
        if single:
            axes[1].spines[s].set_visible(False)

    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"\n曲线已保存到 {out_png}")

def plot_compare(res: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["SimSun", "Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    recs = res["records"]
    labels = [f"{r['step'] // 10000:.0f}万步" for r in recs]
    color = "#1F4E79"

    fig, ax = plt.subplots(figsize=(max(6.2, 2.6 * len(recs)), 5.4), dpi=200)

    bp = ax.boxplot([r["scores"] for r in recs], positions=range(len(recs)),
                    widths=0.5, patch_artist=True,
                    medianprops=dict(color="#ED7D31", linewidth=1.8),
                    flierprops=dict(marker="", alpha=0))
    for patch in bp["boxes"]:
        patch.set_facecolor("#DCE6F1")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)

    rng = np.random.default_rng(0)
    for i, r in enumerate(recs):
        jitter = rng.uniform(-0.16, 0.16, len(r["scores"]))
        ax.scatter(i + jitter, r["scores"], s=16, color=color,
                   alpha=0.35, edgecolors="none", zorder=3)

    for i, r in enumerate(recs):
        lo, hi = bootstrap_ci(r["scores"])
        ax.errorbar(i, r["mean"], yerr=[[r["mean"] - lo], [hi - r["mean"]]],
                    fmt="D", markersize=8, color="#C00000", markeredgecolor="white",
                    markeredgewidth=0.8, ecolor="#C00000", elinewidth=1.8,
                    capsize=6, zorder=5,
                    label="均值 ± 95% CI" if i == 0 else None)
        ax.annotate(f"{r['mean']:.1f}\n[{lo:.0f}, {hi:.0f}]",
                    xy=(i, hi), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9.5, color="#333333")

    ax.set_xticks(range(len(recs)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("模型存档（训练步数）", fontsize=11.5)
    ax.set_ylabel("单局过管数（根）", fontsize=11.5)
    n_ep = res["episodes_per_checkpoint"]
    ax.set_title(f"峰值候选对比（每个存档 {n_ep} 局，"
                 f"箱体=四分位距，红线=中位数，菱形=均值，误差棒=均值 95% CI）",
                 fontsize=11, loc="left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

    if len(recs) == 2:
        (l1, h1), (l2, h2) = bootstrap_ci(recs[0]["scores"]), bootstrap_ci(recs[1]["scores"])
        overlap = not (h1 < l2 or h2 < l1)
        verdict = ("两者 95% CI 重叠 → 差异不显著，不能断言谁更好"
                   if overlap else "两者 95% CI 不重叠 → 差异显著")
        ax.text(0.5, 0.015, verdict, transform=ax.transAxes, ha="center",
                fontsize=9.5, color="#555555",
                bbox=dict(boxstyle="round,pad=0.4", fc="#F2F2F2", ec="#CCCCCC", lw=0.8))

    fig.tight_layout()
    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"\n对比图已保存到 {out_png}")

def main():
    parser = argparse.ArgumentParser(description="按步数批量评估存档并画学习曲线")
    parser.add_argument("--algo", choices=["dqn"], default="dqn",
                        help="评估算法（本脚本只负责 DQN；PPO 请用 PPO/eval_curve_ppo.py）")
    parser.add_argument("--every", type=int, default=50000,
                        help="每隔多少【步】取一个存档评估（默认 50000）")
    parser.add_argument("--episodes", type=int, default=10,
                        help="每个存档连续玩几局（默认 10；建议汇报时用 30）")
    parser.add_argument("--resume", action="store_true",
                        help="续跑：跳过 JSON 里已经评过的存档，只补没评的")
    parser.add_argument("--plot-only", action="store_true",
                        help="不评估，只用现有 JSON 重新画图")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出「要跑哪些存档 / 跳过哪些」，不真正评估")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="只评估指定的这几个存档，如 --steps 1000000 1890000")
    parser.add_argument("--out", default=None,
                        help="输出 JSON 的文件名（不含扩展名），默认同算法名")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    print(f"使用设备: {device}")

    if args.plot_only:
        results = []
        for algo in ["dqn"]:
            jp = out_dir / f"{algo}.json"
            if jp.exists():
                with open(jp, encoding="utf-8") as f:
                    results.append(json.load(f))
            else:
                print(f"找不到 {jp}，跳过")
        if results:
            suffix = "compare" if len(results) > 1 else results[0]["algo"]
            plot_curves(results, out_dir / f"curve_{suffix}.png")
        return

    env = make_game_env()

    algos = ["dqn"]
    out_name = args.out or "dqn"
    results = []
    for algo in algos:
        name = args.out or algo
        if args.dry_run:
            dry_run(algo, args.every, args.episodes, out_dir)
            continue
        res = evaluate_algo(algo, args.every, args.episodes, device, env=env,
                            out_dir=out_dir, resume=args.resume,
                            only_steps=args.steps, out_name=name)
        if res is not None:
            results.append(res)

    if results:
        if args.steps:

            plot_compare(results[0], out_dir / f"compare_{out_name}.png")
        else:
            suffix = "compare" if len(results) > 1 else results[0]["algo"]
            plot_curves(results, out_dir / f"curve_{suffix}.png")

        print("\n" + "=" * 60)
        print("汇总（平均过管数）")
        print("=" * 60)
        for res in results:
            print(f"[{res['algo'].upper()}]  " + "  |  ".join(
                f"{r['step'] // 10000}万步:{r['mean']:.1f}" for r in res["records"]))

if __name__ == "__main__":
    main()
