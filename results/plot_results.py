import json
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "results"
PEV = ROOT / "ppo" / "results"

FONT = "SimSun"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 12,
    "axes.unicode_minus": False,
    "axes.linewidth": 1.1,
    "axes.edgecolor": "#333333",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

C_DQN = "#2E6FB7"
C_PPO = "#C0392B"
C_GREY = "#888888"

def style(ax, xlabel, ylabel, title="", grid=True):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    if grid:
        ax.grid(True, ls="--", lw=0.6, color="#CCCCCC", alpha=0.8)
        ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("#BBBBBB")

def read_tb_csv(path):
    s, v = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, c = line.rstrip("\n").split(",")
            s.append(float(b))
            v.append(float(c))
    return np.asarray(s), np.asarray(v)

def moving_average(x, w=100):
    out = np.empty_like(x, dtype=float)
    dq = deque(maxlen=w)
    for i, xi in enumerate(x):
        dq.append(xi)
        out[i] = np.mean(dq)
    return out

BOOT_B = 20000
BOOT_SEED = 0

def bootstrap_ci(scores, B=BOOT_B, rng=None):
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0:
        return (np.nan, np.nan)
    if rng is None:
        rng = np.random.default_rng(BOOT_SEED)
    means = s[rng.integers(0, n, size=(B, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)

def fig_train_ma100(out_path):

    s_d, v_d = read_tb_csv(EV / "episode_score.csv")
    ma_d = moving_average(v_d, 100)

    s_p, ma_p = read_tb_csv(PEV / "episode" / "episode_score_ma100.csv")

    all_s = np.concatenate([s_d, s_p]).max()
    xmax = np.ceil(all_s / 250000) * 250000 / 1e4

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.plot(s_d / 1e4, ma_d, color=C_DQN, lw=2.2, label="DQN")
    ax.plot(s_p / 1e4, ma_p, color=C_PPO, lw=2.2, label="ppo")

    ax.axhline(100, color=C_GREY, ls=":", lw=1.2)
    ax.text(xmax * 0.99, 101.5, "纵轴上限 100", ha="right", va="bottom",
            fontsize=10, color=C_GREY)

    over = np.where(ma_p >= 100)[0]
    if len(over):
        x_out = s_p[over[0]] / 1e4
        ax.annotate(f"PPO 越过量程\n末值 {ma_p[-1]:.1f}（穿出 {x_out:.0f} 万步）",
                    xy=(x_out, 100), xytext=(-118, -34),
                    textcoords="offset points", fontsize=10.5, color=C_PPO,
                    ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color=C_PPO, lw=1.4))

    ax.annotate(f"DQN  {ma_d[-1]:.1f}",
                xy=(s_d[-1] / 1e4, ma_d[-1]), xytext=(-6, 10),
                textcoords="offset points", ha="right", fontsize=11,
                color=C_DQN, fontweight="bold")

    style(ax, "训练步数（万步）", "最近 100 局平均得分",
          "训练期对比：最近 100 局平均得分（含探索）")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=12)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")
    return ma_d[-1], ma_p[-1]

def _load_eval(path):
    d = json.load(open(path, encoding="utf-8"))
    recs = sorted(d["records"], key=lambda r: r["step"])
    st = np.array([r["step"] for r in recs], dtype=float)
    mn = np.array([r["mean"] for r in recs], dtype=float)
    mx = np.array([r["max"] for r in recs], dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    ci = np.array([bootstrap_ci(r["scores"], rng=rng) for r in recs], dtype=float)
    return st, mn, ci[:, 0], ci[:, 1], mx, recs

def fig_inference(out_path):
    st_d, mn_d, lo_d, hi_d, mx_d, rec_d = _load_eval(EV / "dqn.json")
    st_p, mn_p, lo_p, hi_p, mx_p, rec_p = _load_eval(EV / "PPO.json")

    fig, ax = plt.subplots(figsize=(12.0, 9.0))

    for st, mn, lo, hi, mx, c, name in (
        (st_d, mn_d, lo_d, hi_d, mx_d, C_DQN, "DQN"),
        (st_p, mn_p, lo_p, hi_p, mx_p, C_PPO, "ppo"),
    ):
        ax.plot(st / 1e4, mn, color=c, lw=2.4, marker="o", ms=4.5,
                markerfacecolor="white", markeredgewidth=1.4, label=f"{name}（10 局平均）")
        ax.fill_between(st / 1e4, lo, hi, color=c, alpha=0.16, lw=0,
                        label=f"{name} 95% 自助置信区间")
        ax.plot(st / 1e4, mx, color=c, lw=1.0, ls="--", alpha=0.55,
                label=f"{name}（单局最高）")

    i_d = int(np.argmax(mn_d))
    i_p = int(np.argmax(mn_p))

    ax.scatter([st_p[i_p] / 1e4], [mn_p[i_p]], s=260, marker="*",
               color=C_PPO, zorder=6, edgecolor="white", linewidth=1.1)
    ax.annotate(f"PPO 新峰值 {mn_p[i_p]:.1f}\n（{st_p[i_p]/1e4:.0f} 万步，单局最高 {mx_p[i_p]:.0f}）",
                xy=(st_p[i_p] / 1e4, mn_p[i_p]), xytext=(-168, 26),
                textcoords="offset points", fontsize=11.5, color=C_PPO,
                ha="left", va="bottom", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PPO, lw=1.6))

    style(ax, "训练步数（万步）", "单局平均得分（贪心，无探索）",
          "推理期完整学习曲线：DQN vs PPO（每存档独立评估 10 局）")
    ax.set_xlim(0, 205)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=10.5, ncol=2)

    axi = ax.inset_axes([0.055, 0.335, 0.455, 0.335])
    axi.plot(st_d / 1e4, mn_d, color=C_DQN, lw=1.9, marker="o", ms=3.2,
             markerfacecolor="white", markeredgewidth=1.0)
    axi.fill_between(st_d / 1e4, lo_d, hi_d, color=C_DQN, alpha=0.18, lw=0)
    axi.scatter([st_d[i_d] / 1e4], [mn_d[i_d]], s=150, marker="*",
                color=C_DQN, zorder=6, edgecolor="white", linewidth=1.0)
    axi.annotate(f"峰值 {mn_d[i_d]:.1f}\n（{st_d[i_d]/1e4:.0f} 万步）",
                 xy=(st_d[i_d] / 1e4, mn_d[i_d]), xytext=(-70, 4),
                 textcoords="offset points", fontsize=10, color=C_DQN,
                 ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=C_DQN, lw=1.2))
    axi.set_title("DQN 放大（主图中被压平）", fontsize=10.5, pad=4)
    axi.set_ylim(0, 175)
    axi.set_xlim(0, 205)
    axi.tick_params(labelsize=9)
    axi.grid(True, ls="--", lw=0.5, color="#CCCCCC", alpha=0.8)
    axi.set_axisbelow(True)
    for s in ("top", "right"):
        axi.spines[s].set_color("#BBBBBB")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")
    return dict(dqn_peak=(st_d[i_d] / 1e4, mn_d[i_d], mx_d[i_d]),
                ppo_peak=(st_p[i_p] / 1e4, mn_p[i_p], mx_p[i_p]),
                dqn_last=mn_d[-1], ppo_last=mn_p[-1],
                n_dqn=len(rec_d), n_ppo=len(rec_p),
                ppo_over600=int((mn_p >= 600).sum()),
                dqn_over100=int((mn_d >= 100).sum()))

def fig_diag_dqn(out_path):
    s_q, q = read_tb_csv(EV / "step_q_max.csv")
    s_l, l = read_tb_csv(EV / "step_avg_loss.csv")
    s_e, e = read_tb_csv(EV / "step_epsilon.csv")
    s_r, r = read_tb_csv(EV / "step_reward.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    panels = [
        (s_q, q, "(a) 最大 Q 值 Q(s,·)（单向漂移 → 过估计信号）", "最大 Q 值", C_DQN, None),
        (s_l, l, "(b) 训练损失 MSE（后期不降反升）", "平均损失 (MSE)", C_DQN, None),
        (s_e, e, "(c) 探索率 ε（110 万步后触底 0.01）", "探索率 ε", "#7F8C8D", None),
        (s_r, r, "(d) 每步即时奖励（恒定 0.1 → 失真的指标）", "每步即时奖励", "#27AE60", None),
    ]
    for ax, (s, v, t, ylab, c, _) in zip(axes.ravel(), panels):
        ax.plot(s / 1e4, v, color=c, lw=1.0, alpha=0.35)
        ax.plot(s / 1e4, moving_average(v, 25), color=c, lw=2.2,
                label="滑动平均（窗口 25）")
        style(ax, "训练步数（万步）", ylab, t)
        ax.legend(loc="upper left", fontsize=10.5)

    fig.suptitle("DQN 训练诊断：为什么会崩？", fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")
    return dict(q_start=float(np.mean(q[:20])), q_end=float(moving_average(q, 25)[-1]),
                eps_end=float(e[-1]))

def fig_diag_ppo(out_path):
    rb = PEV / "rollout"
    tr = PEV / "train"
    ls = PEV / "loss"

    items = [
        ("train_entropy.csv", tr, "(a) 策略熵（探索程度）", "熵", C_PPO, "探索 25 万步就基本结束"),
        ("train_explained_variance.csv", tr, "(b) 解释方差（价值函数拟合优度）", "解释方差", "#8E44AD", "Critic 后期剧烈震荡"),
        ("train_clip_fraction.csv", tr, "(c) 裁剪比例（触发信任域的比例）", "裁剪比例", "#D35400", "信任域几乎没启动"),
        ("rollout_flap_ratio.csv", rb, "(d) 拍翅率（选择「拍翅」的步数占比）", "拍翅率", "#16A085", "学会「省力」打法"),
        ("loss_value.csv", ls, "(e) 价值损失", "价值损失", "#C0392B", "25 万步冲高后回落"),
        ("rollout_reward_mean.csv", rb, "(f) 每步平均奖励（早早饱和）", "平均奖励（每步）", "#27AE60", "饱和 → 不是好的能力指标"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 9.0))
    for ax, (fn, folder, t, ylab, c, note) in zip(axes.ravel(), items):
        s, v = read_tb_csv(folder / fn)
        ax.plot(s / 1e4, v, color=c, lw=0.9, alpha=0.32)
        sm = moving_average(v, 25)
        ax.plot(s / 1e4, sm, color=c, lw=2.1, label="滑动平均（窗口 25）")
        style(ax, "训练步数（万步）", ylab, t)
        ax.legend(loc="best", fontsize=10)

        ax.text(0.98, 0.06, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10.5, color="#444444",
                bbox=dict(boxstyle="round,pad=0.28", fc="#F5F5F5", ec="#DDDDDD", lw=0.8))

    fig.suptitle("PPO 训练诊断：为什么稳、瓶颈在哪？", fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")
    _, ent = read_tb_csv(tr / "train_entropy.csv")
    _, clp = read_tb_csv(tr / "train_clip_fraction.csv")
    _, flp = read_tb_csv(rb / "rollout_flap_ratio.csv")
    return dict(ent0=float(ent[0]), ent_end=float(moving_average(ent, 25)[-1]),
                clip_max=float(clp.max()), flap0=float(flp[0]),
                flap_end=float(moving_average(flp, 25)[-1]))

def main():
    print("开始重画实验结果图 …")
    a = fig_train_ma100(EV / "fig_train_ma100.png")
    print(f"     训练期末值：DQN={a[0]:.1f}, PPO={a[1]:.1f}")
    b = fig_inference(EV / "fig_inference.png")
    print(f"     推理期峰值：DQN={b['dqn_peak'][1]:.1f} @{b['dqn_peak'][0]:.0f}万步, "
          f"PPO={b['ppo_peak'][1]:.1f} @{b['ppo_peak'][0]:.0f}万步（单局最高 {b['ppo_peak'][2]:.0f}）")
    c = fig_diag_dqn(EV / "fig_diag_dqn.png")
    print(f"     DQN 诊断：Q {c['q_start']:.1f} → {c['q_end']:.1f}, ε 末值 {c['eps_end']:.3f}")
    d = fig_diag_ppo(EV / "fig_diag_ppo.png")
    print(f"     PPO 诊断：熵 {d['ent0']:.2f} → {d['ent_end']:.2f}, 裁剪率峰值 {d['clip_max']:.3f}, "
          f"拍翅率 {d['flap0']:.2f} → {d['flap_end']:.2f}")
    print("完成 →", EV)

if __name__ == "__main__":
    main()
