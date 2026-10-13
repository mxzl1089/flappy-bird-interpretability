import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "dqn"))

import attribution as A
from explain_dqn import _panel, load_model

DYN = os.path.join(OUT, "dynamics.json")
MAPS = os.path.join(OUT, "attr_maps.npz")
PROBE = os.path.join(OUT, "probe.npz")
EVAL = os.path.join(ROOT, "results", "dqn.json")
E2JSON = os.path.join(OUT, "e2_methods.json.gz")
E4JSON = os.path.join(OUT, "e4_faith.json")

C_SCORE = "#2F6FED"
C_Q = "#E07B39"
C_ENT = "#7F5BC4"
C_BAND = "#C0504D"
C_PIPE = "#2E8B7A"
C_SKY = "#9AA5B1"
C_RAND = "#8A8F98"
C_MODEL = ["#2F6FED", "#2E8B7A", "#7F5BC4", "#E07B39", "#C0504D"]

CRASHES = [(800000, 5.0), (1150000, 9.7), (1300000, 9.6)]

def load_dyn():
    d = json.load(open(DYN, encoding="utf-8"))
    R = d["records"]
    rec = {k: np.array([r[k] for r in R]) for k in R[0] if np.isscalar(R[0][k])}
    rec["frame_sal"] = np.array([r["frame_sal"] for r in R])
    rec["frame_sg"] = np.array([r["frame_sg"] for r in R])
    return d, rec

def load_eval():
    e = json.load(open(EVAL, encoding="utf-8"))["records"]
    return {r["step"]: r for r in e}

def load_maps():
    z = np.load(MAPS)
    out = {}
    for k in z.files:
        step, name = k.split("_", 1)
        out.setdefault(int(step), {})[name] = z[k]
    return out

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✔ {os.path.relpath(p, ROOT)}")
    return p

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])

def fig4_overview(d, rec):
    ev = load_eval()
    step = rec["step"] / 1e6
    mean = np.array([ev[s]["mean"] for s in rec["step"]])
    std = np.array([ev[s]["std"] for s in rec["step"]])
    base = d["baseline"]

    fig = plt.figure(figsize=(15.8, 13.6))
    gs = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.20,
                  left=0.062, right=0.972, top=0.905, bottom=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(step, mean, color=C_SCORE, lw=1.9, marker="o", ms=3.4, label="评估分数（10 局均分）")
    ax.fill_between(step, mean - std, mean + std, color=C_SCORE, alpha=0.12, lw=0)
    ax.set_xlabel("训练步数（百万）"); ax.set_ylabel("每局限过管数", color=C_SCORE)
    ax.tick_params(axis="y", colors=C_SCORE)
    ytop = (mean + std).max() * 1.06
    ax.set_ylim(-8, ytop)
    for cs, cv in CRASHES:
        ax.axvline(cs / 1e6, color=C_RAND, ls=":", lw=1.0)
        ax.text(cs / 1e6 + 0.012, ytop * 0.90, "崩塌", fontsize=8.2, color=C_RAND)
    ax2 = ax.twinx()
    ax2.plot(step, rec["q_chosen"], color=C_Q, lw=1.9, marker="s", ms=3.0, label="选中动作的 Q 值")
    ax2.set_ylabel("平均 Q 值", color=C_Q); ax2.tick_params(axis="y", colors=C_Q)
    ax2.set_ylim(0, rec["q_chosen"].max() * 1.12)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="upper left")
    ax.set_title("(a) 分数（蓝，±1 标准差）与 Q 值（橙）：\n"
                 "Q 单调涨到 10.4，分数却在 10~119 之间反复崩塌", fontsize=11)
    ax.grid(alpha=0.22); ax.spines["top"].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(step, rec["entropy"], color=C_ENT, lw=2.0, marker="o", ms=3.4)
    ax.axvspan(0.03, 0.62, color="#F1F3F7", zorder=0)
    ax.text(0.32, rec["entropy"].max() - 0.004, "快速收敛期", fontsize=9,
            color="#7A828E", ha="center")
    ax.set_xlabel("训练步数（百万）"); ax.set_ylabel("归因分布归一化熵")
    ax.set_title(f"(b) 归因熵：{rec['entropy'][0]:.3f} → {rec['entropy'][-1]:.3f}"
                 "（越小越聚焦）\n0.6M 之前就基本收敛，之后只有缓慢改善", fontsize=11)
    ax.grid(alpha=0.22)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(step, rec["share_band"] * 100, color=C_BAND, lw=2.1, marker="o", ms=3.2,
            label="地面带（恒定不动）")
    ax.plot(step, rec["share_dyn"] * 100, color=C_PIPE, lw=2.1, marker="^", ms=3.6,
            label="管子 + 小鸟（会动）")
    ax.plot(step, rec["share_sky"] * 100, color=C_SKY, lw=1.6, marker="s", ms=2.8,
            label="天空背景")
    for y, c in [(base["ground_band"], C_BAND), (base["dynamic_obj"], C_PIPE),
                 (base["sky"], C_SKY)]:
        ax.axhline(y * 100, color=c, ls="--", lw=1.0, alpha=0.6)
    ax.text(0.015, base["sky"] * 100 + 1.8, "三个区域的像素自然占比基线（22.5% / 22.5% / 55%）",
            transform=ax.get_yaxis_transform(), fontsize=8.6, color="#555B66")
    ax.set_xlabel("训练步数（百万）"); ax.set_ylabel("占全部归因质量的 %")
    ax.set_title("(c) 归因落点：从「看管子」迁移到「看地面带」\n"
                 "两条线近似完美镜像（ρ=±0.96），且都穿过各自的基线", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="center right")
    ax.set_ylim(0, 96); ax.grid(alpha=0.22)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    fg = rec["frame_sg"]
    cols = ["#2F6FED", "#6F9BE8", "#E6A15C", "#C0504D"]
    for j in range(4):
        lab = f"第 {j} 帧" + ("（最新）" if j == 0 else f"（早 {j} 帧）")
        ax.plot(step, fg[:, j], color=cols[j], lw=1.9, marker="o", ms=2.8, label=lab)
    ax.axhline(25, color=C_RAND, ls="--", lw=1.0)
    ax.text(0.015, 25.8, "均匀基线 25%", transform=ax.get_yaxis_transform(),
            fontsize=8.6, color=C_RAND)
    ax.set_xlabel("训练步数（百万）"); ax.set_ylabel("占全部归因的 %（SmoothGrad）")
    ax.set_title("(d) 四帧输入的重要性：最新帧只占到 33%，\n"
                 "并没有退化成「单帧看图」——仍在用帧间差异推速度", fontsize=11)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    ax.grid(alpha=0.22); ax.set_ylim(15, 38)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[2, 0])
    sc = ax.scatter(rec["share_band"] * 100, mean, c=step, cmap="viridis", s=46,
                    edgecolor="white", lw=0.7, zorder=3)
    rho = spearman(rec["share_band"], mean)
    ax.set_xlabel("地面带归因占比 (%)"); ax.set_ylabel("评估分数（每局限过管数）")
    ax.set_title(f"(e) 地面带占比 vs 分数：斯皮尔曼 ρ = {rho:+.2f}（很弱）\n"
                 "占比 78% 的存档，分数从 9.6 一直排到 106.6", fontsize=11)
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("训练步数（百万）", fontsize=9)
    ax.grid(alpha=0.22)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[2, 1])
    edges = [0, 0.5, 1.0, 1.5, 2.01]
    mids = [(lo + hi) / 2 for lo, hi in zip(edges[:-1], edges[1:])]
    med_s, med_q, med_b = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (step >= lo) & (step < hi)
        med_s.append(float(np.median(mean[m])))
        med_q.append(float(np.median(rec["q_chosen"][m])) * 10)
        med_b.append(float(np.median(rec["share_band"][m])) * 100)
    ax.plot(mids, med_s, color=C_SCORE, lw=2.4, marker="o", ms=7, label="分数中位数")
    ax.plot(mids, med_q, color=C_Q, lw=2.4, marker="s", ms=7, label="Q 值中位数 ×10")
    ax.plot(mids, med_b, color=C_BAND, lw=2.4, marker="^", ms=7, label="地面带归因占比 (%)")
    ax.scatter(step, mean, color=C_SCORE, s=16, alpha=0.28, zorder=1)
    for x, v in zip(mids, med_s):
        ax.annotate(f"{v:.0f}", (x, v), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8.6, color=C_SCORE)
    for x, v in zip(mids, med_b):
        ax.annotate(f"{v:.0f}", (x, v), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8.6, color=C_BAND)
    ax.set_xlabel("训练步数（百万）"); ax.set_ylabel("数值（分数 / % / Q×10 同轴比较）")
    ax.set_title("(f) 按 0.5M 分四段看中位数：归因指标平稳爬升，\n"
                 "分数中位数反而在 25~45 之间徘徊（Q 值则同步虚涨）", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="center right")
    ax.grid(alpha=0.22); ax.set_ylim(0, 118)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    fig.suptitle("实验 E1 · 归因随训练阶段的演化（40 个存档 × 400 帧固定探针集；主归因法 = 遮挡法／因果）\n"
                 "核心结论：网络「看哪里」在平滑收敛，而它的「打得好不好」在剧烈震荡 —— 两者解耦",
                 fontsize=13.5, y=0.978)
    return save(fig, "fig4_evolution_overview.png")

def fig5_maps(d, rec, maps, probe):
    picks = [0, 3, 6, 9, 14, 22, 30, -1]
    steps = rec["step"][picks]
    S = probe["states"]
    st_mean = (S > 0).mean(axis=(0, 1)).astype(float)

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.4),
                             gridspec_kw=dict(hspace=0.30, wspace=0.06))
    for ax, sp in zip(axes.ravel(), steps):
        mp = maps[int(sp)]
        heat = mp["oc"].mean(axis=0)
        _panel(ax, st_mean * 255, heat,
               f"{sp/1e6:.2f}M 步   分数 {rec['score'][rec['step']==sp][0]:.0f}   "
               f"地面带 {rec['share_band'][rec['step']==sp][0]*100:.0f}%")
    fig.suptitle("实验 E1 · 平均遮挡热力图随训练的变化（8 个阶段，同一批 400 帧画面上）\n"
                 "底图 = 探针集平均画面；红色越深 = 涂掉它对 Q 的打击越大",
                 fontsize=13.5, y=0.985)
    fig.subplots_adjust(top=0.83, bottom=0.015, left=0.012, right=0.99)
    return save(fig, "fig5_attribution_maps.png")

def compute_e2(d, rec, probe, device, states_n=4, samples=2048):
    if os.path.exists(E2JSON):
        with gzip.open(E2JSON, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    picks = [0, 7, 14, 25, 30, -1]
    steps = [int(s) for s in rec["step"][picks]]
    S = probe["states"][:states_n]
    out = {"steps": steps, "methods": list(A.METHODS), "maps": {}}

    for st in steps:
        net, _ = load_model(f"runs/ckpt-{st}.pt", device)

        x = torch.as_tensor(S, dtype=torch.float32, device=device)
        with torch.no_grad():
            act = net(x).argmax(1).cpu().numpy()
        acc = {}
        for name, fn in A.METHODS.items():
            m = np.stack([fn(net, s, int(a), device) for s, a in zip(S, act)])
            acc[name] = np.abs(m).mean(axis=0).astype(np.float32)
        out["maps"][str(st)] = {k: v.tolist() for k, v in acc.items()}
        print(f"    E2: {st} 步 完成（{len(A.METHODS)} 种方法 × {states_n} 帧）")

    with gzip.open(E2JSON, "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    return out

def fig6_methods(e2, probe):
    S = probe["states"][:4]
    st_mean = (S > 0).mean(axis=(0, 1)).astype(float)
    methods = e2["methods"]; steps = e2["steps"]
    nrow, ncol = len(steps), len(methods)

    fig, axes = plt.subplots(nrow, ncol, figsize=(2.42 * ncol, 2.22 * nrow),
                             gridspec_kw=dict(hspace=0.16, wspace=0.05))
    axes = np.atleast_2d(axes)
    for r, st in enumerate(steps):
        for c, mname in enumerate(methods):
            heat = np.array(e2["maps"][str(st)][mname]).mean(axis=0)
            _panel(axes[r, c], st_mean * 255, heat, mname if r == 0 else None)

        axes[r, 0].set_ylabel(f"{st/1e6:.2f}M 步", fontsize=10.5,
                              rotation=90, labelpad=6)

    fig.suptitle("实验 E2 · 五种归因方法 × 六个训练阶段（同一批 4 帧探针画面，取平均）\n"
                 "行 = 训练阶段（上：早期 → 下：最佳）；列 = 归因方法",
                 fontsize=13.5, y=0.982)
    fig.subplots_adjust(top=0.918, bottom=0.015, left=0.052, right=0.99)
    return save(fig, "fig6_method_matrix.png")

def fig7_crash(d, rec):
    ev = load_eval()
    step = rec["step"]
    mean = np.array([ev[s]["mean"] for s in step])

    late = step >= 1000000
    def cv(a): return float(np.std(a) / abs(np.mean(a)))
    band, q = rec["share_band"], rec["q_chosen"]
    r_band, r_score = cv(band[late]), cv(mean[late])

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4), gridspec_kw=dict(wspace=0.34))
    for ax, (cs, cvv) in zip(axes, CRASHES):
        idx = np.where(np.abs(step - cs) <= 100000)[0]
        xs = step[idx] / 1e6
        bars = ax.bar(xs, mean[idx], width=0.05, color=C_SCORE, alpha=0.85, zorder=2)
        for b, s in zip(bars, step[idx]):
            if s == cs:
                b.set_color("#C0504D")
        for b, v in zip(bars, mean[idx]):
            ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}",
                    ha="center", fontsize=8.8, color="#334155")
        ax.set_ylabel("评估分数（每局限过管数）", color=C_SCORE)
        ax.tick_params(axis="y", colors=C_SCORE); ax.set_xlabel("训练步数（百万）")
        ax.set_ylim(0, max(mean[idx]) * 1.9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)

        ax2 = ax.twinx()
        ax2.plot(xs, q[idx], color=C_Q, lw=2.4, marker="o", ms=7, zorder=4)
        ax2.set_ylabel("Q 值（选中动作）", color=C_Q); ax2.tick_params(axis="y", colors=C_Q)
        lo, hi = q[idx].min(), q[idx].max()
        ax2.set_ylim(lo - (hi - lo) * 0.9 - 0.15, hi + (hi - lo) * 0.9 + 0.15)
        for xi, v in zip(xs, q[idx]):
            ax2.annotate(f"{v:.2f}", (xi, v), xytext=(0, 9), textcoords="offset points",
                         ha="center", fontsize=8.6, color=C_Q)
        j = int(np.argmax(step[idx] == cs))
        ax.set_title(f"崩塌于 {cs/1e6:.2f}M：分数 {mean[idx][j-1]:.0f} → "
                     f"{mean[idx][j]:.0f} → {mean[idx][j+1]:.0f}，\n"
                     f"同期 Q 反而 {q[idx][j-1]:.2f} → {q[idx][j]:.2f} → {q[idx][j+1]:.2f}",
                     fontsize=11)

    fig.suptitle("实验 E3 · 三次训练崩塌：分数崩掉一个数量级，"
                 "Q 值却在继续上涨，归因几乎不动\n"
                 f"1.00M 之后：分数的变异系数 {r_score:.2f}，是地面带归因占比（{r_band:.3f}）的"
                 f" {r_score/r_band:.0f} 倍 —— 性能与「看哪里」解耦",
                 fontsize=13, y=0.985)
    fig.subplots_adjust(top=0.78, bottom=0.115, left=0.055, right=0.955)
    return save(fig, "fig7_crash_diagnosis.png")

def compute_e4(d, rec, probe, device, n_states=10, steps=10):
    if os.path.exists(E4JSON):
        return json.load(open(E4JSON, encoding="utf-8"))

    picks = [0, 7, 14, 22, 30, -1]
    sel = [int(s) for s in rec["step"][picks]]
    S = probe["states"]
    out = {"steps": sel, "methods": list(A.METHODS) + ["随机（对照）"], "auc": {},
           "mode": "relative", "n_states": n_states}

    for st in sel:
        net, _ = load_model(f"runs/ckpt-{st}.pt", device)
        x = torch.as_tensor(S, dtype=torch.float32, device=device)
        with torch.no_grad():
            Q = net(x).cpu().numpy()
        act = Q.argmax(1)
        q = Q[np.arange(len(S)), act]

        idx = np.argsort(-q)[:n_states]
        Ss, As = S[idx], act[idx].astype(int)
        row = {}
        for name, fn in A.METHODS.items():
            attrs = np.stack([fn(net, s, int(a), device) for s, a in zip(Ss, As)])
            fracs, c_attr, c_rand = A.deletion_curve(net, Ss, As, attrs, device,
                                                     steps=steps, mode="relative")
            row[name] = float(np.trapz(c_attr, fracs))
            row["随机（对照）"] = float(np.trapz(c_rand, fracs))
        out["auc"][str(st)] = row
        print("    E4: {:>8} 步   ".format(st)
              + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))

    json.dump(out, open(E4JSON, "w", encoding="utf-8"))
    return out

def fig8_faith(e4, rec):
    steps_all = [int(s) for s in e4["steps"]]
    keep = [s for s in steps_all if s >= 450000]
    x = np.array(keep) / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.9),
                             gridspec_kw=dict(wspace=0.24, width_ratios=[1.45, 1]))

    ax = axes[0]
    rand = [e4["auc"][str(s)]["随机（对照）"] for s in keep]
    ax.fill_between(x, 0, rand, color="#EDEFF3", zorder=0)
    ax.plot(x, rand, color=C_RAND, lw=2.4, ls="--", marker="s", ms=7,
            label="随机顺序（对照基线）", zorder=3)
    for i, m in enumerate([mm for mm in e4["methods"] if "随机" not in mm]):
        ys = [e4["auc"][str(s)][m] for s in keep]
        ax.plot(x, ys, lw=2.0, marker="o", ms=6, color=C_MODEL[i], label=m, zorder=4)
    ax.text(0.02, 0.93, "阴影区 = 比随机更可信", transform=ax.transAxes,
            fontsize=9.5, color="#7A828E")
    ax.set_xlabel("训练步数（百万）")
    ax.set_ylabel("相对删除曲线 AUC（越小 = 归因越可信）")
    ax.set_title("(a) 归因可信度随训练的演化（相对归一化，值域≈[0,1]）\n"
                 "全部五种方法都稳定优于随机；Saliency 最差，且随训练反而变差",
                 fontsize=11.5)
    ax.legend(fontsize=9.5, frameon=False, ncol=3, loc="lower left")
    ax.grid(alpha=0.22); ax.set_ylim(0, 0.45)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    ax = axes[1]
    sb = {s: rec["share_band"][rec["step"] == s][0] for s in keep}
    sal = [e4["auc"][str(s)]["Saliency"] for s in keep]
    occ = [e4["auc"][str(s)]["Occlusion"] for s in keep]
    xs = [sb[s] * 100 for s in keep]
    ax.scatter(xs, sal, s=110, color=C_MODEL[0], marker="o", edgecolor="white",
               lw=1.0, zorder=3, label="Saliency（梯度法）")
    ax.scatter(xs, occ, s=110, color=C_MODEL[4], marker="s", edgecolor="white",
               lw=1.0, zorder=3, label="Occlusion（因果法）")
    for a, b, s in zip(xs, sal, keep):
        ax.annotate(f"{s/1e6:.2f}M", (a, b), xytext=(7, 6),
                    textcoords="offset points", fontsize=8.5, color="#6B7280")
    rho = spearman(np.array(xs), np.array(sal))
    z1 = np.polyfit(xs, sal, 1); z2 = np.polyfit(xs, occ, 1)
    xx = np.linspace(min(xs) - 3, max(xs) + 3, 20)
    ax.plot(xx, np.polyval(z1, xx), color=C_MODEL[0], lw=1.3, ls=":")
    ax.plot(xx, np.polyval(z2, xx), color=C_MODEL[4], lw=1.3, ls=":")
    ax.set_xlabel("地面带归因占比 (%)  ← E1 的指标")
    ax.set_ylabel("相对删除曲线 AUC  ← E4 的指标")
    ax.set_title(f"(b) 把 E1 和 E4 串起来：越依赖静态地面带，\n"
                 f"梯度法越不可信（斯皮尔曼 ρ(Saliency) = {rho:+.2f}）", fontsize=11.5)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.22)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.suptitle("实验 E4 · 归因可信度（保真度）随训练的演化　"
                 "｜　0.05M 存档因模型未训练（Q 为负、删除曲线无定义）已剔除",
                 fontsize=13.5, y=0.985)
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.055, right=0.98)
    return save(fig, "fig8_faithfulness_evolution.png")

def fig9_verification(d, rec):
    v = json.load(open(os.path.join(OUT, "verify.json"), encoding="utf-8"))
    fig = plt.figure(figsize=(15.6, 10.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.22,
                  left=0.065, right=0.975, top=0.855, bottom=0.115)

    ax = fig.add_subplot(gs[0, 0])
    ab = v["v3_ablation"]
    steps = sorted(int(k) for k in ab)
    x = np.arange(len(steps)); w = 0.27
    band = [ab[str(s)][0] for s in steps]
    dynm = [ab[str(s)][1] for s in steps]
    ax.bar(x - w, band, w, label="抹掉地面带", color=C_BAND)
    ax.bar(x, dynm, w, label="抹掉管子+小鸟", color=C_PIPE)
    ax.bar(x + w, [ab[str(s)][2] for s in steps], w, label="抹掉天空", color=C_SKY)
    ax.set_xticks(x); ax.set_xticklabels([f"{s/1e6:.2f}M" for s in steps])
    ax.set_ylabel("|ΔQ|  （区域整体涂黑后 Q 的平均变化量）")
    ax.set_title("(a) 区域消融：不依赖任何归因算法、不依赖分块\n"
                 "直接整体抹掉一个区域，看 Q 掉多少", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.22)
    ax2 = ax.twinx()
    ratio = [band[i] / max(dynm[i], 1e-9) for i in range(len(steps))]
    ax2.plot(x, ratio, color="#7F5BC4", lw=2.2, marker="D", ms=7, label="地面带 / 管子")
    ax2.axhline(1.0, color=C_RAND, ls="--", lw=1.0)
    ax2.set_ylabel("地面带 / 管子（倍数）", color="#7F5BC4")
    ax2.tick_params(axis="y", colors="#7F5BC4")
    for xi, r in zip(x, ratio):
        ax2.annotate(f"{r:.2f}×", (xi, r), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, color="#7F5BC4", fontweight="bold")
    ax2.legend(fontsize=9, frameon=False, loc="upper right")
    ax2.set_ylim(0, max(ratio) * 1.55)
    for s in ("top",): ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    va = v["v6_ablation"]; vs = va["存活中位步数"]
    conds = ["原始", "等价对照_纯白", "抹掉地面带", "抹掉管子", "恢复后"]
    names = ["原始\n（对照）", "换纯白\n（等价对照）", "抹掉地面带", "抹掉管子\n（阳性对照）", "恢复后\n复查"]
    means = [va[c] for c in conds]
    sur = [vs[{"原始": "原始", "等价对照_纯白": "等价对照", "抹掉地面带": "抹掉地面带",
               "抹掉管子": "抹掉管子", "恢复后": "恢复后"}[c]] for c in conds]
    cols = [C_SCORE, C_SKY, C_BAND, C_PIPE, C_SCORE]
    bars = ax.bar(range(len(conds)), means, 0.6, color=cols)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 1.5, f"{m:.1f}",
                ha="center", fontsize=9.5, color="#334155")
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("每局均分（8 局）", color=C_SCORE); ax.tick_params(axis="y", colors=C_SCORE)
    ax.set_ylim(0, max(means) * 1.6)
    ax.set_title("(b) 行为消融：把区域从画面上真抹掉，让模型接着玩\n"
                 "①≡②≡⑤ 逐局完全一致 → 换贴图零副作用，实验可信", fontsize=11)
    ax.grid(axis="y", alpha=0.22)
    ax2 = ax.twinx()
    ax2.plot(range(len(conds)), sur, color="#E07B39", lw=2.0, marker="o", ms=7,
             label="存活步数（中位）")
    for xi, s in zip(range(len(conds)), sur):
        ax2.annotate(f"{s:.0f}", (xi, s), xytext=(0, -16), textcoords="offset points",
                     ha="center", fontsize=9, color="#E07B39")
    ax2.set_ylabel("存活步数（中位）", color="#E07B39")
    ax2.tick_params(axis="y", colors="#E07B39"); ax2.set_ylim(0, 4200)
    ax2.legend(fontsize=9, frameon=False, loc="upper center")
    for s in ("top",): ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    v4 = v["v4_probe_robust"]; v5 = v["v5_action_robust"]
    ss = sorted(int(k) for k in v4)
    a = [v4[str(s)][1] for s in ss]
    b = [v4[str(s)][3] for s in ss]
    c = [v5[str(s)][3] for s in ss]
    ax.plot([0, 0.9], [0, 0.9], color=C_RAND, ls="--", lw=1.1, label="y = x（完全一致）")
    ax.scatter(a, b, s=95, color=C_PIPE, marker="o", edgecolor="white", lw=0.9,
               zorder=3, label="换探针集（另一个模型采的画面）")
    ax.scatter(a, c, s=95, color=C_Q, marker="^", edgecolor="white", lw=0.9,
               zorder=3, label="固定动作（全部改为「不拍翅」）")
    for xi, yi, s in zip(a, b, ss):
        ax.annotate(f"{s/1e6:.2f}M", (xi, yi), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color="#6B7280")
    mx = max(max(abs(np.array(a) - np.array(b))), max(abs(np.array(a) - np.array(c))))
    ax.set_xlabel("主探针集 · 地面带归因占比"); ax.set_ylabel("换条件后的地面带归因占比")
    ax.set_title(f"(c) 稳健性：换成另一个模型采的画面、或把所有动作固定\n"
                 f"最大偏差仅 {mx:.4f} —— 结论不是探针分布或动作选择的产物", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(alpha=0.22); ax.set_xlim(0.25, 0.90); ax.set_ylim(0.25, 0.90)
    ax.set_aspect("equal")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    step = rec["step"] / 1e6
    v7 = v["v7_q_bound"]
    ax.plot(step, rec["q_chosen"], color=C_Q, lw=2.2, marker="o", ms=3.4,
            label="实测：选中动作的平均 Q")
    ax.axhline(v7["v_max"], color="#C0504D", ls="--", lw=1.8)
    ax.text(0.02, v7["v_max"] + 0.22,
            f"正确上界 V* = {v7['v_max']:.2f}（存活 0.1/帧 + 每 36 帧过一管）",
            transform=ax.get_yaxis_transform(), fontsize=8.5, color="#C0504D")
    ax.fill_between([0, 2.05], 0, v7["v_max"], color="#F2F4F8", zorder=0)
    ax.annotate("", xy=(step[-1], rec["q_chosen"][-1]),
                xytext=(step[-1], v7["v_max"]),
                arrowprops=dict(arrowstyle="<->", color="#334155", lw=1.4))
    ax.text(step[-1] - 0.36, (rec["q_chosen"][-1] + v7["v_max"]) / 2,
            f"{v7['ratio']:.2f}×\n（未越界）", fontsize=10, color="#334155",
            fontweight="bold", ha="center")
    ax.set_xlabel("训练步数（百万）"); ax.set_ylabel("Q 值")
    ax.set_ylim(0, max(14.0, v7["v_max"] * 1.12))
    ax.set_title("(d) Q 上界校验：实测 Q 涨到 %.1f，但【没有】越过正确上界 %.2f\n"
                 "旧口径漏算存活奖励，「3.15× 过估计」已作废（复核抓出的错误）"
                 % (v7["q_max"], v7["v_max"]), fontsize=10.5)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(alpha=0.22)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    txt = (f"算法自检（V1/V2）\n"
           f"· 批量实现 vs 逐像素暴力实现：最大差 {v['v1_max_abs_diff']:.2e}\n"
           f"· 纯背景块涂黑的 |drop|：{v['v2_sky_drop_max']:.1e}（结构必然为 0）")
    fig.text(0.065, 0.012, txt, fontsize=9, color="#4B5563",
             bbox=dict(boxstyle="round,pad=0.5", fc="#F6F7F9", ec="#D8DCE3"))

    fig.suptitle("实验 E5 · 结论可靠性复核：互相独立的证据链\n"
                 "①换算法 ②换分块粒度 ③换输入分布与动作 ④真实行为消融 ⑤Q 上界校验（本项曾纠出旧口径错误）",
                 fontsize=13, y=0.975)
    return save(fig, "fig9_verification.png")

def main():
    ap = argparse.ArgumentParser(description="可解释性实验图表")
    ap.add_argument("--only", default="all",
                    choices=["all", "e1", "e2", "e3", "e4", "e5"])
    ap.add_argument("--no-compute", action="store_true",
                    help="只用已有缓存，不补算 E2/E4（缓存不存在就跳过该图）")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.chdir(ROOT)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    t0 = time.time()

    d, rec = load_dyn()
    maps = load_maps()
    probe = np.load(PROBE)
    print(f"数据：{len(rec['step'])} 个存档，探针集 {len(probe['states'])} 帧，设备 {device}")

    if args.only in ("all", "e1"):
        print("\n[E1] 归因随训练阶段的演化")
        fig4_overview(d, rec)
        fig5_maps(d, rec, maps, probe)

    if args.only in ("all", "e2"):
        print("\n[E2] 五种方法 × 六个关键存档")
        try:
            e2 = compute_e2(d, rec, probe, device)
            fig6_methods(e2, probe)
        except FileNotFoundError as e:
            print(f"    跳过（{e}）")

    if args.only in ("all", "e3"):
        print("\n[E3] 三次训练崩塌的诊断")
        fig7_crash(d, rec)

    if args.only in ("all", "e4"):
        print("\n[E4] 保真度随训练的演化")
        try:
            e4 = compute_e4(d, rec, probe, device)
            fig8_faith(e4, rec)
        except FileNotFoundError as e:
            print(f"    跳过（{e}）")

    if args.only in ("all", "e5"):
        print("\n[E5] 结论可靠性复核")
        fig9_verification(d, rec)

    print(f"\n全部完成，用时 {time.time() - t0:.0f} 秒。")

if __name__ == "__main__":
    main()
