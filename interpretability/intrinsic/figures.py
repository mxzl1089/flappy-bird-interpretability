import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.edgecolor"] = "#9A9A93"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#444441"
plt.rcParams["ytick.color"] = "#444441"
plt.rcParams["text.color"] = "#2C2C2A"

C_LIN = "#185FA5"
C_CON = "#888780"
C_DQN = "#0F6E56"
C_FAIL = "#A32D2D"
C_OK = "#0F6E56"
C_PAPER = "#534AB7"
C_GRID = "#D3D1C7"

PAPER_T5 = {"Deep-DQN": 123.42, "CUT": 78.51, "CART": 79.13, "M5-RT": 42.14,
            "M5-MT": 78.26, "LMUT-Exp": 97.62, "LMUT-Active": 103.32}
PAPER_LEAVES = {"CART": 700.3, "M5-RT": 226.1, "M5-MT": 412.6,
                "LMUT-Exp": 578.5, "LMUT-Active": 229.0}

def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → figs/{name}.png / .pdf")

def style(ax):
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def figL1(pr, sd):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.1))
    for ax in (a, b):
        style(ax)

    d = pr["data_scaling"]
    n = [r["n"] for r in d]
    r = [r["test_RAE"] for r in d]
    a.plot(n, r, "o-", color=C_LIN, lw=2, ms=6, mfc="white", mew=1.8)
    a.set_xscale("log")
    a.set_ylim(0.05, 1.14)
    a.set_xlabel("训练样本数（对数刻度）")
    a.set_ylabel("测试 RAE（Q(s,1)，越低越好）")
    a.set_title("(a) 数据量扫描：样本越多误差越小，且未饱和")
    a.axhline(1.0, color=C_CON, ls=":", lw=1.2)
    a.text(n[0], 1.03, "永远输出均值 = 1.0", fontsize=8, color=C_CON)
    offs = [(0, 9), (-16, -15), (16, 8), (-14, 8), (-10, -16), (0, 9)]
    for (x, y), o in zip(zip(n, r), offs):
        a.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=o,
                   ha="center", fontsize=8, color=C_LIN)
    a.annotate(f"样本量最大的 {n[-1]:,} 条处\n仍在下降 → 未饱和", xy=(n[-1], r[-1]),
               xytext=(215, 0.075), fontsize=9, color=C_FAIL, ha="left", va="bottom",
               arrowprops=dict(arrowstyle="->", color=C_FAIL, lw=1.2, shrinkB=6))

    sd_rows = sorted(sd["rows"], key=lambda z: z["d"])
    dd = [x["d"] for x in sd_rows]
    rr = [x["test_RAE"] for x in sd_rows]
    b.plot(dd, rr, "s-", color=C_FAIL, lw=2, ms=6, mfc="white", mew=1.8)
    b.set_xscale("log")
    b.set_ylim(0.12, 0.56)
    b.set_xlabel("输入维度 d（对数刻度；f×f 平均池化）")
    b.set_ylabel("测试 RAE（Q(s,1)）")
    b.set_title("(b) 维度扫描：降维不改善，反而变差")
    for (x, y), o in zip(zip(dd, rr), [(-4, 9), (21, 5), (0, -16), (-4, -16)]):
        b.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=o,
                   ha="center", fontsize=8, color="#5F5E5A")
    b.text(0.97, 0.96, f"降 {dd[-1] / dd[-2]:.0f} 倍维度（{dd[-1]}→{dd[-2]}）\n"
                       f"RAE {rr[-1]:.3f}→{rr[-2]:.3f}，基本不动",
           transform=b.transAxes, ha="right", va="top", fontsize=9, color=C_FAIL,
           bbox=dict(boxstyle="round,pad=0.35", fc="#FCEBEB", ec="none"))

    fig.suptitle("LMUT 可行性：两个受控扫描把「模型不行」与「数据不够」分开",
                 fontsize=13, fontweight="500", y=1.02)
    fig.tight_layout()
    save(fig, "figL1_两个受控扫描")

def figL2(pr):
    fig = plt.figure(figsize=(12.4, 4.2))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.32)

    a = fig.add_subplot(gs[0]); style(a)
    q1 = pr["per_action"]["Q1"]
    ds = [5, 6, 7, 8]
    rl = [q1[f"linear_depth{x}"]["RAE"] for x in ds]
    rc = [q1[f"const_depth{x}"]["RAE"] for x in ds]
    x = np.arange(len(ds)); w = 0.36
    a.bar(x - w / 2, rc, w, color=C_CON, label="常数叶 CART")
    a.bar(x + w / 2, rl, w, color=C_LIN, label="线性叶模型树")
    a.set_xticks(x); a.set_xticklabels([f"depth={k}" for k in ds])
    a.set_ylabel("测试 RAE（Q(s,1)）"); a.set_ylim(0, 0.50)
    a.set_title("(a) 保真度：线性叶只小胜")
    a.legend(fontsize=8, frameon=False, loc="upper right")
    a.text(0.04, 0.93, f"depth=8 时 −{100*(1-rl[-1]/rc[-1]):.0f}%",
           transform=a.transAxes, ha="left", fontsize=9, color=C_LIN)

    b = fig.add_subplot(gs[1]); style(b)
    dm = pr["criteria"]["dqn_mean"]
    mm = pr["criteria"]["mimic"]
    vals = [dm, mm["linear"]["mean"], mm["const"]["mean"]]
    lbl = ["DQN\n（老师）", "线性叶\n模型树", "常数叶\nCART"]
    bars = b.bar(lbl, vals, color=[C_DQN, C_LIN, C_CON], width=0.55)
    for bb, v in zip(bars, vals):
        b.text(bb.get_x() + bb.get_width() / 2, v + 2.5, f"{v:.1f}", ha="center", fontsize=9)
    b.set_ylabel("10 局平均过管数"); b.set_ylim(0, dm * 1.18)
    b.set_title("(b) 可玩性：线性叶是唯一没归零的")
    b.axhline(0.60 * dm, color=C_FAIL, ls="--", lw=1.2)
    b.text(2.45, 0.60 * dm + 2, "判据 C 门槛 60%", ha="right", fontsize=8, color=C_FAIL)

    c = fig.add_subplot(gs[2]); style(c)
    names = ["LMUT\n(Active)", "LMUT\n(Experience)", "CART", "M5-MT", "CUT", "M5-RT"]
    paper = [PAPER_T5["LMUT-Active"], PAPER_T5["LMUT-Exp"], PAPER_T5["CART"],
             PAPER_T5["M5-MT"], PAPER_T5["CUT"], PAPER_T5["M5-RT"]]
    pn = [100 * v / PAPER_T5["Deep-DQN"] for v in paper]
    on = [100 * mm["linear"]["mean"] / dm, np.nan, 100 * mm["const"]["mean"] / dm,
          np.nan, np.nan, np.nan]
    y = np.arange(len(names))[::-1]
    c.barh(y + 0.19, pn, 0.36, color=C_PAPER, label="原论文（ECML PKDD 2018）")
    c.barh(y - 0.19, [v if not np.isnan(v) else 0 for v in on], 0.36,
           color=C_LIN, label="本次复现")
    for yy, v in zip(y + 0.19, pn):
        c.text(v + 1.5, yy, f"{v:.0f}%", va="center", fontsize=8, color=C_PAPER)
    for yy, v in zip(y - 0.19, on):
        if not np.isnan(v):
            c.text(v + 1.5, yy, f"{v:.1f}%", va="center", fontsize=8, color=C_FAIL)
    c.set_yticks(y); c.set_yticklabels(names, fontsize=9)
    c.set_xlim(0, 108); c.set_xlabel("游戏得分 / 各自 DQN（%，量纲无关）")
    c.set_title("(c) 端到端：论文能到 79%，本次 4.2%")
    c.legend(fontsize=8, frameon=False, loc="lower right")

    fig.suptitle("线性叶决定「能不能玩」，但没决定「能玩多好」——与原论文同构的对照",
                 fontsize=13, fontweight="500", y=1.03)
    save(fig, "figL2_线性叶的价值与论文对照")

def figL3(pr):
    fig, ax = plt.subplots(figsize=(8.6, 4.2)); style(ax)
    gold = pr["q_stats"]["flap_rate_train"]
    comb = pr["combined"]
    items = [
        ("永远不拍翅\n（平凡基线）", 1 - gold, 0.0, C_CON),
        ("永远拍翅\n（平凡基线）", gold, 1.0, C_CON),
        ("常数叶 CART\ndepth=6", comb["const_depth6"]["agree"], comb["const_depth6"]["recall_flap"], C_CON),
        ("常数叶 CART\ndepth=8", comb["const_depth8"]["agree"], comb["const_depth8"]["recall_flap"], C_CON),
        ("线性叶模型树\ndepth=6", comb["linear_depth6"]["agree"], comb["linear_depth6"]["recall_flap"], C_LIN),
        ("线性叶模型树\ndepth=8", comb["linear_depth8"]["agree"], comb["linear_depth8"]["recall_flap"], C_LIN),
    ]
    x = np.arange(len(items)); w = 0.36
    ax.bar(x - w / 2, [i[1] for i in items], w, color=[i[3] for i in items],
           alpha=0.45, label="决策总一致率")
    ax.bar(x + w / 2, [i[2] for i in items], w, color=[i[3] for i in items],
           label="拍翅召回率（少数类）")
    ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items], fontsize=8.5)
    ax.set_ylabel("比例"); ax.set_ylim(0, 1.32)
    ax.axhline(gold, color=C_FAIL, ls="--", lw=1.1)
    for i, (_, ag, rc, _c) in enumerate(items):
        ax.text(i - w / 2, ag + 0.02, f"{ag:.0%}", ha="center", fontsize=8)
        ax.text(i + w / 2, rc + 0.02, f"{rc:.0%}", ha="center", fontsize=8)
    ax.set_title("判据必须压在少数类上：总一致率越高反而越不能玩（它靠「少拍翅」换来）")
    handles = [Patch(fc="#C9C6BE", label="决策总一致率"),
               Patch(fc="#5F5E5A", label="拍翅召回率（少数类）"),
               Line2D([0], [0], color=C_FAIL, ls="--", lw=1.1,
                      label=f"DQN 实际拍翅率 {gold:.2%}")]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper center",
              ncol=3, bbox_to_anchor=(0.5, 1.02))
    save(fig, "figL3_类别不平衡陷阱")

def figL4(pr, sw):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.2))
    style(a); style(b)

    rows = sorted([r for r in sw["rows"] if r["mode"] == "linear" and r["min_child"] == 8],
                  key=lambda z: z["leaves_q1"])
    x = [r["leaves_q1"] for r in rows]
    te = [r["q1_RAE"] for r in rows]
    tr = [r["q1_train_RAE"] for r in rows]
    a.plot(x, te, "o-", color=C_LIN, lw=2, ms=6, mfc="white", mew=1.8, label="测试 RAE")
    a.plot(x, tr, "s--", color=C_FAIL, lw=1.7, ms=5, mfc="white", mew=1.6, label="训练 RAE")
    for xi, yi in zip(x, te):
        a.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                   ha="center", fontsize=8, color=C_LIN)
    for xi, yi in zip(x, tr):
        a.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points", xytext=(0, -16),
                   ha="center", fontsize=8, color=C_FAIL)
    old = pr["per_action"]["Q1"]["linear_depth8"]
    a.plot([old["leaves"]], [old["RAE"]], "*", ms=14, color="#5F5E5A",
           label="第一轮探针 (mc=20)")
    for nm, lv in PAPER_LEAVES.items():
        if nm.startswith("LMUT"):
            a.axvline(lv, color=C_PAPER, ls=":", lw=1.2)
    g0 = te[1] / tr[1] if len(te) > 1 else 0
    g1 = te[-1] / tr[-1]
    a.set_ylim(min(tr) - 0.012, max(te) + 0.045)
    a.set_xlabel("叶子数（单动作）"); a.set_ylabel("RAE（Q(s,1)）")
    a.set_title(f"(a) 容量扫描：叶数 ×{x[-1]/old['leaves']:.1f}，测试 RAE 仅 "
                f"−{100*(1-te[-1]/old['RAE']):.0f}%（训练/测试差距 {old['RAE']/old['train_RAE']:.1f}× "
                f"→ {g1:.1f}×）", fontsize=9.5)
    a.text(0.03, 0.97, "虚线 = 论文 LMUT 叶数（229 / 578）", transform=a.transAxes,
           ha="left", va="top", fontsize=8, color=C_PAPER)
    a.legend(fontsize=8, frameon=False, loc="lower right")

    e2e = sw["end2end"]
    dm = sw["dqn_mean"]
    names = ["DQN\n（老师）", f"线性叶\n{e2e['linear']['leaves']} 叶",
             f"常数叶\n{e2e['const']['leaves']} 叶"]
    vals = [dm, e2e["linear"]["mean"], e2e["const"]["mean"]]
    bars = b.bar(names, vals, color=[C_DQN, C_LIN, C_CON], width=0.55)
    for bb, v in zip(bars, vals):
        b.text(bb.get_x() + bb.get_width() / 2, v + 2.5, f"{v:.1f}", ha="center", fontsize=9)
    b.set_ylabel("10 局平均过管数"); b.set_ylim(0, max(dm, 1) * 1.2)
    b.set_title("(b) 同容量（640 叶）下的可玩性")
    b.axhline(0.60 * dm, color=C_FAIL, ls="--", lw=1.2)
    b.text(2.45, 0.60 * dm + 2, "判据 C 门槛 60%", ha="right", fontsize=8, color=C_FAIL)

    fig.suptitle("容量扫描：把树做到论文量级 —— 训练误差猛降、测试误差几乎不动，可玩性没有回来",
                 fontsize=12.5, fontweight="500", y=1.04)
    fig.tight_layout()
    save(fig, "figL4_容量扫描")

def main():
    pr = load("probe.json")
    sd = load("scan_dim.json")
    print("生成图表 →", FIGS)
    figL1(pr, sd)
    figL2(pr)
    figL3(pr)
    sw_path = os.path.join(OUT, "sweep_capacity.json")
    if os.path.exists(sw_path):
        figL4(pr, load("sweep_capacity.json"))
    else:
        print("  （缺 sweep_capacity.json，跳过 figL4）")

if __name__ == "__main__":
    main()
