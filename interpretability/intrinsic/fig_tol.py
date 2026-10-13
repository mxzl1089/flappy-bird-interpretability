import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
for k in ("figure.facecolor", "axes.facecolor", "savefig.facecolor"):
    plt.rcParams[k] = "white"
plt.rcParams["font.size"] = 10

C_LIN, C_HI, C_MID, C_LO = "#185FA5", "#A32D2D", "#854F0B", "#B4B2A9"

def load(tag):
    p = os.path.join(OUT, tag + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

def curve():
    rows = {}
    for tag in ("noise_tol", "noise_tol_fine"):
        d = load(tag)
        if not d:
            continue
        for r in d["rows"]:
            rows[round(float(r["p"]), 6)] = r
    P = np.array(sorted(rows))
    score = np.array([rows[p]["score_mean"] for p in P])
    length = np.array([rows[p]["len_mean"] for p in P])
    return P, score, length

def eq_p(P, length, target_len):
    m = (P > 0) & (length > 0)
    lp, ll = np.log(P[m]), np.log(length[m])
    o = np.argsort(ll)
    return float(np.exp(np.interp(np.log(target_len), ll[o], lp[o])))

def draw_eq(ax, P, score, length, target_len, color, label, off=(6, -16), ha="left"):
    pe = eq_p(P, length, target_len)
    ss = np.exp(np.interp(np.log(pe), np.log(P[P > 0]), np.log(score[P > 0])))
    ax.plot([pe], [ss], marker="o", ms=8, mfc="none", mec=color, mew=2.0, zorder=6)
    ax.annotate(label, xy=(pe, ss), xytext=off, textcoords="offset points",
                fontsize=8.6, color=color, ha=ha,
                bbox=dict(fc="white", ec=color, lw=0.7, alpha=0.9, pad=2.2))
    return pe

def main():
    P, score, length = curve()
    dv = load("divergence")
    pr = json.load(open(os.path.join(OUT, "probe.json"), encoding="utf-8"))
    dqn_mean = pr["criteria"]["dqn_mean"]
    tree_mean = pr["criteria"]["mimic"]["linear"]["mean"]
    cnn_mean = dv["score_mean"]
    tree_dis = 1.0 - 0.8906
    cnn_dis = dv["expert_baseline"]["disagree"]
    tree_len = float(np.interp(tree_mean, score[score.argsort()], length[score.argsort()]))
    cnn_len = dv["frames_mean"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.6, 5.6),
                               gridspec_kw={"width_ratios": [1.45, 1]})

    m = P > 0
    a.plot(P[m], score[m], "o-", color=C_LIN, lw=2.0, ms=6, label="教师自身（贪心 + 概率 p 翻错动作）")
    a.plot([1e-4], [dqn_mean], marker="*", ms=16, color=C_HI, ls="none",
           label=f"p=0 的 DQN 本人（{dqn_mean:.1f} 分）")
    a.axhline(0.79 * dqn_mean, color=C_HI, ls="--", lw=1.3)
    a.axhline(0.64 * dqn_mean, color=C_MID, ls=":", lw=1.3)
    a.text(1.15e-4, 0.79 * dqn_mean + 2.5,
           f"论文 LMUT 水平 = DQN 的 79%（{0.79*dqn_mean:.0f} 分）", fontsize=9, color=C_HI)
    a.text(1.15e-4, 0.64 * dqn_mean - 7.5,
           f"论文 CART 水平 = DQN 的 64%（{0.64*dqn_mean:.0f} 分）", fontsize=9, color=C_MID)

    pe_t = draw_eq(a, P, score, length, tree_len, C_LO, f"线性叶树 {tree_mean:.1f} 分",
                   off=(-4, -20), ha="right")
    pe_c = draw_eq(a, P, score, length, cnn_len, "#0F6E5C", f"同架构 CNN {cnn_mean:.1f} 分",
                   off=(8, -18))
    a.set_xscale("log")
    a.set_xlim(3e-5, 0.2)
    a.set_ylim(-5, dqn_mean * 1.12)
    a.set_xlabel("每帧做错动作的概率 p（对数轴）")
    a.set_ylabel("10~30 局平均得分（过管数）")
    a.set_title("本环境对「动作错误率」的放大：p 从 0 到 0.5% 就丢掉 89% 的分", fontsize=10.5)
    a.legend(fontsize=8.6, frameon=False, loc="upper right")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(True, color="#D3D1C7", lw=0.6); a.set_axisbelow(True)

    xs = np.arange(2)
    dis = [tree_dis, cnn_dis]
    eqs = [pe_t, pe_c]
    w = 0.34
    b.bar(xs - w / 2, np.array(dis) * 100, width=w, color=C_LO,
          label="实测逐帧分歧率（与教师 argmax 不同）")
    b.bar(xs + w / 2, np.array(eqs) * 100, width=w, color="#0F6E5C",
          label="等效随机错误率（由存活帧数换算）")
    for i in range(2):
        b.text(i - w / 2, dis[i] * 100 + 0.25, f"{dis[i]*100:.2f}%", ha="center", fontsize=9)
        b.text(i + w / 2, eqs[i] * 100 + 0.25, f"{eqs[i]*100:.2f}%", ha="center", fontsize=9)
        b.annotate("", xy=(i + w / 2, eqs[i] * 100), xytext=(i - w / 2, dis[i] * 100),
                   arrowprops=dict(arrowstyle="->", color="#5F5E5A", lw=1.0,
                                   connectionstyle="arc3,rad=-0.25"))
        b.text(i, dis[i] * 100 * 0.55 + eqs[i] * 100 * 0.45,
               f"×{dis[i]/eqs[i]:.0f}", ha="center", fontsize=9.5, color="#A32D2D",
               bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.6))
    b.set_xticks(xs)
    b.set_xticklabels([f"线性叶模型树\n（{tree_mean:.1f} 分）", f"同架构 CNN\n（{cnn_mean:.1f} 分）"])
    b.set_ylabel("占全部帧的百分比 (%)")
    b.set_title("模仿器的错误「没那么值钱」：等效错误率比实测分歧低 8~30 倍", fontsize=10.5)
    b.legend(fontsize=8.6, frameon=False)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.grid(True, color="#D3D1C7", lw=0.6, axis="y"); b.set_axisbelow(True)

    need_cart = _need(P, score, length, 0.64 * dqn_mean)
    need_lmut = _need(P, score, length, 0.79 * dqn_mean)
    fig.suptitle(f"复现不上论文的定量归因：要够到论文的 64~79%，模仿器的等效错误率需 ≤{need_cart*100:.2f}%，"
                 f"本项目最好一档是 {pe_c*100:.2f}%（还需再降 {pe_c/need_cart:.1f} 倍）",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"figE4_错误容忍度与等效错误率.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("等效错误率：线性叶树 %.4f%%   同架构 CNN %.4f%%" % (pe_t * 100, pe_c * 100))
    print("论文目标（在实测区间 p∈[0.001,0.01] 上双对数外推）："
          "CART 64%% → p ≈ %.4f%% ；LMUT 79%% → p ≈ %.4f%%"
          % (need_cart * 100, need_lmut * 100))
    print("差距倍数：CNN 需再降 %.1f 倍（对 CART） / %.1f 倍（对 LMUT）"
          % (pe_c / need_cart, pe_c / need_lmut))
    print("→ figs/figE4_错误容忍度与等效错误率.png / .pdf")

def _need(P, score, length, target_score):
    m = (P >= 0.001) & (P <= 0.01)
    k, b = np.polyfit(np.log(P[m]), np.log(score[m]), 1)
    return float(np.exp((np.log(target_score) - b) / k))

if __name__ == "__main__":
    main()
