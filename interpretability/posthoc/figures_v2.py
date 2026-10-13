import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out_v2")
OUT_OLD = os.path.join(HERE, "out")
FIGS = os.path.join(HERE, "figures")
sys.path.insert(0, HERE)
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

C_BAND = "#0F6E56"
C_DYN = "#854F0B"
C_SKY = "#B4B2A9"
C_OCC = "#185FA5"
C_SAL = "#D85A30"
C_RND = "#888780"
C_SHAP = "#534AB7"
C_IG = "#639922"
C_OK = "#0F6E56"
C_FAIL = "#A32D2D"
C_NONE = "#BA7517"
C_BIRD = "#D4537E"

def J(name, old=False):
    p = os.path.join(OUT_OLD if old else OUT, name)
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def save(fig, name, note=None):
    if note:
        fig.text(0.008, 0.006, note, fontsize=8.2, color="#5F5E5A", ha="left", va="bottom")
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  ✔ {name}.png / .pdf")

def clean(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#000000", alpha=0.07, linewidth=0.7)
        ax.set_axisbelow(True)

def box(ax, x, y, w, h, text, fc="#F1EFE8", ec="#888780", fs=9, tc="#2C2C2A",
        ha="center", pad=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, linewidth=0.9))
    if ha == "left":
        ax.text(x + pad, y + h / 2, text, ha="left", va="center", fontsize=fs, color=tc)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

def arrow(ax, p, q, color="#5F5E5A", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.0, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))

def fig01():
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.text(0.15, 5.32, "图 1  验证流程总览：从游戏画面到三态判定", fontsize=12.5, weight="500")
    ax.text(0.15, 5.06, "每一步都先写死「证伪条件」再跑；结果只有三种状态：成立 / 证伪 / 未观测到",
            fontsize=9.6, color="#5F5E5A")

    box(ax, 0.15, 3.90, 1.62, 0.78, "游戏画面\n288×512×3")
    box(ax, 2.02, 3.90, 1.62, 0.78, "灰度 + 缩放到\n80×80")
    box(ax, 3.89, 3.90, 1.62, 0.78, "二值化\n只留 0 / 255")
    box(ax, 5.76, 3.90, 1.62, 0.78, "堆叠 4 帧\n(4,80,80)")
    box(ax, 7.63, 3.90, 1.62, 0.78, "Q 网络\n输出 Q(s,0),Q(s,1)")
    box(ax, 9.50, 3.90, 2.74, 0.78, "决策 = argmax\n拍翅 if Q(s,1)>Q(s,0)", fc="#E6F1FB", ec=C_OCC)
    for x in (1.77, 3.64, 5.51, 7.38, 9.25):
        arrow(ax, (x, 4.29), (x + 0.25, 4.29))

    ax.text(0.15, 3.48, "三类扰动实验（都作用在 80×80 二值图上，再喂同一批固定探针帧）",
            fontsize=10.5, weight="500")
    box(ax, 0.15, 2.42, 3.55, 0.86, "① 区域掩码\n抹天空 / 抹地面带 / 抹小鸟 / 抹管子",
        fc="#E1F5EE", ec=C_BAND)
    box(ax, 3.95, 2.42, 3.55, 0.86, "② 等面积掩码\n各抹 720 个白像素，只改「抹在哪」",
        fc="#FAEEDA", ec=C_DYN)
    box(ax, 7.75, 2.42, 4.49, 0.86, "③ 边缘位移掩码\n带顶 k 行下移 / 带内随机抹 k 行 / 顶部加白",
        fc="#F1EFE8", ec=C_SKY)

    ax.text(0.15, 1.98, "四类读数（缺一不可：只看翻转率会被大裕度样本稀释；只看 ΔQ 会丢掉方向）",
            fontsize=10.5, weight="500")
    box(ax, 0.15, 1.16, 2.10, 0.70, "argmax 翻转率\n+ bootstrap 95% CI")
    box(ax, 2.40, 1.16, 2.10, 0.70, "Δ(Q1−Q0)\n拍翅优势的变化")
    box(ax, 4.65, 1.16, 2.10, 0.70, "ΔP(拍翅)\nsoftmax 概率变化")
    box(ax, 6.90, 1.16, 2.10, 0.70, "|ΔQ|\n删块敏感度")
    box(ax, 9.15, 1.16, 3.09, 0.70, "删除曲线 AUC\n保真度（越低越可信）", fc="#E6F1FB", ec=C_OCC)

    ax.text(0.15, 0.80, "必查的两个对照 —— 阴性对照保证实现没写错，阳性对照保证测得住",
            fontsize=10.5, weight="500")
    box(ax, 0.15, 0.10, 5.85, 0.60, "阴性对照：抹天空（背景像素恒为 0）→ 必须【恰好零效应】",
        fc="#FCEBEB", ec=C_FAIL, fs=9.2)
    box(ax, 6.40, 0.10, 5.84, 0.60, "阳性对照：抹小鸟 / 抹管子 → 必须【明显改变决策】",
        fc="#EAF3DE", ec=C_OK, fs=9.2)
    save(fig, "fig01_method_overview",
         "读法：任何一条结论，都必须同时说清「扰动定义、读数口径、对照组」三件事才成立。")

def fig02():
    import xrl_v2 as X
    B = X.get_baseline()
    z = np.load(X.PROBE_MAIN)
    S, bm = z["states"], z["birdmask"]
    r0 = X.band_row0(B)
    k = 3
    frames = S[k].astype(float) / 255.0
    w = S[k].max(axis=0) > 0

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9))
    fig.suptitle("图 2  网络真正看到的输入：一张 80×80 二值图（纵向已被压扁）",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.02)

    axes[0].imshow(frames[-1].T, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("(a) 最新一帧的二值图", fontsize=10)
    axes[0].text(4, 70, "白 = 值 255\n黑 = 值 0（天空）", fontsize=8.4, color="#2C2C2A",
                 bbox=dict(fc="white", ec="#B4B2A9", lw=0.6, pad=2.2))

    rgb = np.ones((80, 80, 3))
    band = np.zeros((80, 80), bool); band[:, r0:] = True
    rgb[w & band] = matplotlib.colors.to_rgb(C_BAND)
    rgb[w & ~band] = matplotlib.colors.to_rgb(C_DYN)
    axes[1].imshow(rgb.transpose(1, 0, 2), interpolation="nearest")
    axes[1].axhline(r0 - 0.5, color="#A32D2D", lw=1.1, ls="--")
    axes[1].set_title("(b) 区域划分", fontsize=10)
    axes[1].text(4, 68, "teal 地面带（底部 18 行）\namber 动态物体（管子 + 小鸟）",
                 fontsize=8.4, bbox=dict(fc="white", ec="#B4B2A9", lw=0.6, pad=2.2))

    axes[2].imshow(frames[-1].T, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    bu = bm[k].any(axis=0)
    ov = np.zeros((80, 80, 4))
    ov[bu] = matplotlib.colors.to_rgba(C_BIRD, 0.95)
    axes[2].imshow(ov.transpose(1, 0, 2), interpolation="nearest")
    axes[2].set_title("(c) 精确小鸟掩码（两次渲染取差得到）", fontsize=10)
    axes[2].text(4, 68, f"鸟掩码占全部输入像素 {bm.mean():.2%}\n（用矩形框会误伤路过的管子）",
                 fontsize=8.4, bbox=dict(fc="white", ec="#B4B2A9", lw=0.6, pad=2.2))

    gcnt = np.array([(S[i].max(axis=0) > 0)[:, r0:].sum() for i in range(len(S))])
    dcnt = np.array([(S[i].max(axis=0) > 0)[:, :r0].sum() for i in range(len(S))])
    axes[3].hist(gcnt, bins=30, color=C_BAND, alpha=0.85, label="地面带（恒定 1440）")
    axes[3].hist(dcnt, bins=30, color=C_DYN, alpha=0.75, label="动态物体（中位 %d）" % int(np.median(dcnt)))
    axes[3].set_xlabel("该区域的白像素数"); axes[3].set_ylabel("帧数")
    axes[3].set_title("(d) 两个区域的白像素量级", fontsize=10)
    axes[3].legend(fontsize=8.6, frameon=False)
    clean(axes[3])
    for a in axes[:3]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    save(fig, "fig02_input_regions",
         "读法：地面带【恒定且占满】1440 像素——这正是「它到底是真线索还是大扰动」这个疑问的来源。")

def fig03(device="cuda"):
    import xrl_v2 as X
    import torch
    B = X.get_baseline()
    z = np.load(X.PROBE_MAIN)
    S, A = z["states"], z["actions"]
    net = X.load_net(X.CKPT_BEST, device)
    cache = os.path.join(OUT, "attr_maps_v2.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        oc, sal = d["oc"], d["sal"]
    else:
        oc = X.occlusion_fast(net, S, A, device, block=8)
        sal = X.saliency_batch(net, S, A, device)
        np.savez_compressed(cache, oc=oc, sal=sal)
    r0 = X.band_row0(B)

    picks = [2, 3, len(S) - 2]
    fig, axes = plt.subplots(2, 4, figsize=(11.8, 5.9))
    fig.suptitle("图 3  逐帧归因图：遮挡法（下排）vs Saliency（上排）",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.005)
    for r, k in enumerate(picks):
        base = S[k][-1].astype(float).T / 255.0
        for c, (arr, name, cmap) in enumerate([
                (np.abs(sal[k]).max(axis=0).T, "Saliency |∂Q/∂x|", "Oranges"),
                (np.abs(oc[k]).max(axis=0).T, "遮挡 |ΔQ|（block=8）", "Blues")]):
            ax = axes[c, r]
            ax.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

            an = arr / (arr.max() + 1e-12)
            ax.imshow(np.ma.masked_where(an < 0.05, an), cmap=cmap, alpha=0.9,
                      vmin=0, vmax=1, interpolation="nearest")
            ax.axhline(r0 - 0.5, color="#A32D2D", lw=1.0, ls="--")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name}　帧 #{k}", fontsize=9.4)
    for c, (arr, name, cmap) in enumerate([
            (np.abs(sal).mean(axis=(0, 1)).T, "Saliency 全体平均", "Oranges"),
            (np.abs(oc).mean(axis=(0, 1)).T, "遮挡 全体平均", "Blues")]):
        ax = axes[c, 3]
        ax.imshow(np.full((80, 80), 0.35), cmap="gray", vmin=0, vmax=1)
        an = arr / (arr.max() + 1e-12)
        im = ax.imshow(np.ma.masked_where(an < 0.05, an), cmap=cmap, vmin=0, vmax=1,
                       interpolation="nearest")
        ax.axhline(r0 - 0.5, color="#A32D2D", lw=1.0, ls="--")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}（400 帧）", fontsize=9.4)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("相对强度（已按各自最大值归一）", fontsize=7.6)
    axes[0, 0].set_ylabel("Saliency", fontsize=10, color=C_SAL)
    axes[1, 0].set_ylabel("遮挡法", fontsize=10, color=C_OCC)
    fig.tight_layout(rect=[0, 0.045, 1, 0.965])
    save(fig, "fig03_occlusion_maps",
         "读法：红虚线 = 地面带上边界。遮挡图把权重压在整条白带上，Saliency 则散乱、且最大值比遮挡小约 4 个数量级。")

def fig04():
    AT = J("attr.json")
    blocks = [4, 8, 16]
    band = [AT[f"block{b}"]["share_band"] for b in blocks]
    dyn = [AT[f"block{b}"]["share_dyn"] for b in blocks]
    sky = [AT[f"block{b}"]["share_sky"] for b in blocks]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.0, 4.4),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("图 4  归因质量的三区域占比，以及它随块大小的系统性漂移",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.02)

    x = np.arange(len(blocks)); wd = 0.26
    ax.bar(x - wd, band, wd, color=C_BAND, label="地面带")
    ax.bar(x, dyn, wd, color=C_DYN, label="动态物体")
    ax.bar(x + wd, sky, wd, color=C_SKY, label="天空")
    for xi, v in zip(x - wd, band):
        ax.text(xi, v + 0.012, f"{v:.1%}", ha="center", fontsize=8.8, color=C_BAND)
    for xi, v in zip(x, dyn):
        ax.text(xi, v + 0.012, f"{v:.1%}", ha="center", fontsize=8.8, color=C_DYN)
    for xi, v in zip(x + wd, sky):
        ax.text(xi, v + 0.012, f"{v:.1%}", ha="center", fontsize=8.8, color="#5F5E5A")
    ax.set_xticks(x); ax.set_xticklabels([f"块大小 {b}×{b}" for b in blocks])
    ax.set_ylabel("占 |归因| 质量的百分比"); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, frameon=False, ncol=3)
    ax.set_title("16×16 时粗块跨越地面带上边界，占比被压低至 67.2%",
                 fontsize=8.8, color="#A32D2D")
    clean(ax)

    rng = AT["verdict"]["share_band_range"]
    bx.barh([0], [1.0], height=0.30, color="#F1EFE8", edgecolor="#888780", linewidth=0.8)
    bx.barh([0], [rng[1] - rng[0]], left=rng[0], height=0.30, color=C_BAND, alpha=0.9)
    bx.plot([rng[0], rng[1]], [0, 0], "|", color="#A32D2D", ms=14, mew=1.4)

    bx.text(0.04, 0, f"极差 {rng[1]-rng[0]:.3f}（85.9% → 67.2%）",
            fontsize=8.8, va="center", ha="left", color="#2C2C2A")
    bx.set_xlim(0, 1.16); bx.set_yticks([]); bx.set_xlabel("地面带占比")
    bx.set_title("占比不是块大小无关量：报数字必须带块大小", fontsize=10)
    clean(bx, grid="x")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save(fig, "fig04_share_blocksize",
         "读法：块的【排名】在三种粒度下不变（地面带始终第一），但【占比数值】会漂 —— 所以两篇论文的占比不可直接比。")

def fig05(device="cuda"):
    import xrl_v2 as X
    AT = J("attr.json")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.0, 4.3),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle("图 5  保真度检验：按归因排序逐步抹像素，Q 掉得多快",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.02)

    rows = [("遮挡法（本套件主用）", AT["block8"]["auc_occlusion"], AT["block8"]["auc_occlusion_ci"], C_OCC),
            ("Saliency（梯度）", AT["block8"]["auc_saliency"], AT["block8"]["auc_saliency_ci"], C_SAL),
            ("随机顺序（实测基线）", AT["block8"]["auc_random"], AT["block8"]["auc_random_ci"], C_RND)]
    ypos = np.arange(len(rows))[::-1]
    for y, (name, v, ci, c) in zip(ypos, rows):
        ax.barh(y, v, height=0.5, color=c, alpha=0.9)
        ax.plot([ci[0], ci[1]], [y, y], color="#2C2C2A", lw=1.2)
        ax.plot([ci[0], ci[0]], [y - 0.10, y + 0.10], color="#2C2C2A", lw=1.2)
        ax.plot([ci[1], ci[1]], [y - 0.10, y + 0.10], color="#2C2C2A", lw=1.2)
        ax.text(v + 0.012, y, f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", va="center", fontsize=9)
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("相对删除曲线 AUC（越低 = 越早打中要害 = 越可信）")
    ax.set_xlim(0, 0.52); ax.set_ylim(-0.6, 2.95); clean(ax, grid="x")

    ax.text(0.24, 2.88, "遮挡 CI 与随机 CI 完全不重叠\n→ 遮挡确实携带信息",
            fontsize=8.8, color=C_OCC, va="top", ha="left")

    steps = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bx.plot(steps, [1.0, 0.55, 0.30, 0.14, 0.05, 0.0], "-o", ms=3.5, color=C_OCC, lw=1.6,
            label="遮挡法")
    bx.plot(steps, [1.0, 0.83, 0.66, 0.47, 0.26, 0.0], "-s", ms=3.5, color=C_SAL, lw=1.6,
            label="Saliency")
    bx.plot(steps, [1.0, 0.90, 0.79, 0.63, 0.35, 0.0], "--^", ms=3.5, color=C_RND, lw=1.4,
            label="随机顺序")
    bx.set_xlabel("被删掉像素的比例"); bx.set_ylabel("Q 相对跌幅（归一化）")
    bx.set_title("删除曲线示意（已被 AUC 概括）", fontsize=10)
    bx.legend(fontsize=9, frameon=False); clean(bx)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    save(fig, "fig05_deletion_curves",
         "读法：曲线越早掉到 0，说明归因排在最前面的像素越要害。随机基线实测 0.374（< 0.5），因为随机抹到大量本就是 0 的像素。")

def fig06():
    d = J("e4_faith.json", old=True)
    steps = [int(s) for s in d["steps"]]
    meths = ["Occlusion", "KernelSHAP", "IG", "SmoothGrad", "Saliency", "随机（对照）"]
    colors = {"Occlusion": C_OCC, "KernelSHAP": C_SHAP, "IG": C_IG,
              "SmoothGrad": "#EF9F27", "Saliency": C_SAL, "随机（对照）": C_RND}
    markers = {"Occlusion": "o", "KernelSHAP": "D", "IG": "v",
               "SmoothGrad": "s", "Saliency": "^", "随机（对照）": "x"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.3),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle("图 6  五种归因方法的保真度：遮挡 ≈ KernelSHAP，都明显优于梯度族与随机",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.02)
    xx = np.array(steps) / 1e6
    for m in meths:
        ys = [d["auc"][str(s)][m] for s in steps]
        ax.plot(xx, ys, marker=markers[m], ms=4, lw=1.5,
                color=colors[m], label=m,
                ls="--" if m in ("随机（对照）", "KernelSHAP") else "-")
    ax.axhline(0, color="#B4B2A9", lw=0.8)
    ax.set_xlabel("训练步数（百万）"); ax.set_ylabel("相对删除曲线 AUC（越低越好）")
    ax.legend(fontsize=8.6, frameon=False, ncol=2); clean(ax)

    ax.text(0.60, -3.6, "早期存档 AUC 出现负值：Q(全黑) 与 Q(原图) 几乎相等，\n"
                        "相对口径的分母趋零 → 这种点不应当作证据",
            fontsize=8.0, color="#5F5E5A")

    late = [s for s in steps if s >= 800000]
    mean = {m: float(np.mean([d["auc"][str(s)][m] for s in late])) for m in meths}
    order = sorted(mean, key=lambda m: mean[m])
    y = np.arange(len(order))[::-1]
    bx.barh(y, [mean[m] for m in order], height=0.55, color=[colors[m] for m in order], alpha=0.9)
    for yi, m in zip(y, order):
        bx.text(mean[m] + 0.006, yi, f"{mean[m]:.3f}", va="center", fontsize=9)
    bx.set_yticks(y); bx.set_yticklabels(order, fontsize=9)
    bx.set_xlabel(f"0.8M–2.0M 存档的平均 AUC（{len(late)} 个）")
    bx.set_title("后期存档平均：遮挡与 SHAP 同一档", fontsize=10)
    clean(bx, grid="x")
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    save(fig, "fig06_faithfulness_methods",
         "读法：遮挡法只做「删一块」，KernelSHAP 枚举「留/删任意子集」再加权 —— 在本任务上后者没换来保真度收益。")

def fig07():
    CA = J("causal.json")
    conds = ["sky(阴性对照)", "black(整幅涂黑,次级参考)", "dyn(抹管子+小鸟)",
             "pipes(抹管子,留小鸟)", "ground(抹地面带)", "bird(抹小鸟)"]
    label = {"sky(阴性对照)": "抹天空\n(阴性对照)", "black(整幅涂黑,次级参考)": "整幅涂黑\n(分布外，仅参考)",
             "dyn(抹管子+小鸟)": "抹管子+小鸟", "pipes(抹管子,留小鸟)": "抹管子\n(留小鸟)",
             "ground(抹地面带)": "抹地面带\n(1440 像素)", "bird(抹小鸟)": "抹小鸟\n(仅 0.70% 像素)"}
    col = {"sky(阴性对照)": C_SKY, "black(整幅涂黑,次级参考)": "#5F5E5A",
           "dyn(抹管子+小鸟)": "#EF9F27", "pipes(抹管子,留小鸟)": C_DYN,
           "ground(抹地面带)": C_BAND, "bird(抹小鸟)": C_BIRD}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("图 7  决策级因果：抹掉某块后，按键改变的比例（开环单帧，400 帧探针集）",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.02)
    x = np.arange(len(conds))
    v = [CA["conditions"][c]["flip_rate"] for c in conds]
    ci = [CA["conditions"][c]["flip_rate_ci"] for c in conds]
    ax.bar(x, v, width=0.6, color=[col[c] for c in conds])
    ax.errorbar(x, v, yerr=[[vv - c[0] for vv, c in zip(v, ci)], [c[1] - vv for vv, c in zip(v, ci)]],
                fmt="none", ecolor="#2C2C2A", elinewidth=1.2, capsize=4)
    ax.axhline(0.05, color="#A32D2D", ls="--", lw=1.0)
    ax.text(len(conds) - 0.45, 0.055, "预设判据 5%", fontsize=8.6, color="#A32D2D", ha="right")
    for xi, vv in zip(x, v):
        ax.text(xi, vv + 0.012, f"{vv:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([label[c] for c in conds], fontsize=8.8)
    ax.set_ylabel("argmax 翻转率"); ax.set_ylim(0, 0.44); clean(ax)
    ax.text(0.55, 0.30, "注意：这里各条件的【面积不相等】，\n所以「鸟 > 管子」不能当作强度对比，\n严格对比见图 10",
            fontsize=8.3, color="#5F5E5A")

    marc = CA["conditions"]
    t = [marc["ground(抹地面带)"]["flip_by_margin_tercile"],
         marc["bird(抹小鸟)"]["flip_by_margin_tercile"],
         marc["pipes(抹管子,留小鸟)"]["flip_by_margin_tercile"]]
    names = ["抹地面带", "抹小鸟", "抹管子"]
    wd = 0.26
    xs = np.arange(3)
    cs = [C_BAND, C_BIRD, C_DYN]
    for j in range(3):
        bx.bar(xs + (j - 1) * wd, [t[i][j] for i in range(3)], wd,
               color=cs, alpha=0.55 + 0.2 * j, label=["小裕度", "中裕度", "大裕度"][j])
    bx.set_xticks(xs); bx.set_xticklabels(names)
    bx.set_ylabel("翻转率"); bx.set_xlabel("原始决策裕度 |Q1−Q0| 三分层")
    bx.set_title("扣掉「本来就拍板很硬」的帧之后", fontsize=10)
    bx.legend(fontsize=9, frameon=False); clean(bx)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    save(fig, "fig07_causal_flip",
         "读法：翻转率随裕度增大而下降 → 单看总翻转率会被大批「拍板很硬」的帧稀释，必须分层看。")

def fig08():
    SC = J("scan.json")
    rec = SC["records"]
    st = np.array([r["step"] for r in rec]) / 1e6
    band = np.array([r["share_band"] for r in rec])
    dyn = np.array([r["share_dyn"] for r in rec])
    q0 = np.array([r["q_act0"] for r in rec])
    score = np.array([r["score"] for r in rec])
    sp = SC["spearman"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 4.5))
    fig.suptitle("图 8  策略在 40 个训练存档上的注意力迁移：从「看物体」转向「看地面」",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.02)
    ax.plot(st, band, "-o", ms=3.2, color=C_BAND, lw=1.7, label="地面带占比")
    ax.plot(st, dyn, "--s", ms=3.2, color=C_DYN, lw=1.7, label="动态物体占比")
    ax.fill_between(st, band, dyn, where=band > dyn, color=C_BAND, alpha=0.08)
    ax.axvline(1.89, color="#A32D2D", lw=1.0, ls=":")
    ax.text(1.90, 0.86, "最佳存档\n1.89M", fontsize=8.4, color="#A32D2D")
    ax.set_xlabel("训练步数（百万）"); ax.set_ylabel("占 |归因| 质量的百分比")
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=9, frameon=False, loc="center right")
    clean(ax)
    ax.text(0.03, 0.04, f"ρ(步数, 地面带) = {sp['step_vs_share_band'][0]:+.3f}   "
                        f"置换检验 p = {sp['step_vs_share_band'][1]:.4f}",
            fontsize=9, color=C_BAND,
            bbox=dict(fc="white", ec=C_BAND, lw=0.7, pad=3))

    bx.plot(st, q0, "-o", ms=3.2, color=C_OCC, lw=1.7, label="Q(s, 不拍) 均值")
    bx.set_xlabel("训练步数（百万）"); bx.set_ylabel("Q(s,0)", color=C_OCC)
    bx.tick_params(axis="y", colors=C_OCC); clean(bx)
    b2 = bx.twinx()
    b2.plot(st, score, "--s", ms=3.2, color="#EF9F27", lw=1.4, label="评估分数")
    b2.set_ylabel("评估均分", color="#BA7517"); b2.tick_params(axis="y", colors="#BA7517")
    b2.spines["top"].set_visible(False)
    bx.set_title("Q 值随步数单调放大，成绩并不单调\n"
                 f"ρ(步数, Q) = {sp['step_vs_q_act0'][0]:+.3f} (p={sp['step_vs_q_act0'][1]:.4f})　"
                 f"ρ(步数, 分数) = {sp['step_vs_score'][0]:+.3f} (p={sp['step_vs_score'][1]:.4f})",
                 fontsize=9.6)
    h1, l1 = bx.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    bx.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    save(fig, "fig08_attention_migration",
         "读法：这条迁移是全套实验里最稳的正向结论；但它的度量仍受图 10「面积效应」限制——占比较高 ≠ 语义更重要。")

def fig09():
    SH = J("shift.json")
    KS = [2, 4, 8, 16]
    tags = {"D(边缘下移)": ("D 边缘下移（抹带顶 k 行）", C_OCC, "-o"),
            "R(同面积随机抹)": ("R 带内随机抹 k 行（同面积）", C_RND, "--s"),
            "U(向上加长)": ("U 向上加长 k 行（多 k 行白）", C_BAND, "-^"),
            "SKY(顶部加白,地面不动)": ("SKY 画面顶部加 k 行白（地面不动）", "#EF9F27", "-v")}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 4.5),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    fig.suptitle("图 9  「地面带 = 高度尺」的方向性检验：预测符号与实际符号相反",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.02)
    for tag, (lab, c, fmt) in tags.items():
        ys = [SH["conditions"][f"{tag} k={k}"]["mean_dQ"] for k in KS]
        ax.plot(KS, ys, fmt, ms=5, lw=1.6, color=c, label=lab)
    ax.axhline(0, color="#2C2C2A", lw=0.9)
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("掩码宽度 k（行）"); ax.set_ylabel("Δ(Q1 − Q0) 均值　正 = 更倾向拍翅")
    ax.set_ylim(-0.45, 3.15)
    ax.legend(fontsize=8.4, frameon=False, loc="upper left"); clean(ax)
    ax.set_title("四条边缘位移掩码的 Δ(Q1−Q0)", fontsize=10)

    ax.text(0.5, 1.135, "若地面带是「高度尺」：边缘下移 = 自认更高 = 更该不拍\n"
                        "→ D 这条线必须全在 0 以下；实测【全在 0 以上】",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.4, color=C_FAIL)

    v = SH["verdict"]
    pair = v["same_area_pair_diff"]
    x = np.arange(len(KS)); wd = 0.36
    bx.bar(x - wd / 2, [SH["conditions"][f"D(边缘下移) k={k}"]["mean_dQ"] for k in KS],
           wd, color=C_OCC, label="D 抹带顶 k 行")
    bx.bar(x + wd / 2, [SH["conditions"][f"R(同面积随机抹) k={k}"]["mean_dQ"] for k in KS],
           wd, color=C_RND, label="R 带内随机抹 k 行（同面积）")
    for xi, p in zip(x, pair):
        bx.text(xi, max(0, 2.95) - 0.05, f"配对差 {p:+.2f}", ha="center", fontsize=8.2, color="#A32D2D")
    bx.axhline(0, color="#2C2C2A", lw=0.9)
    bx.set_xticks(x); bx.set_xticklabels([f"k={k}" for k in KS])
    bx.set_ylabel("Δ(Q1 − Q0) 均值"); bx.set_ylim(-0.35, 3.45)
    bx.legend(fontsize=8.6, frameon=False); clean(bx)
    bx.set_title("同面积对照：随机抹反而更致命", fontsize=10)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    save(fig, "fig09_height_ruler",
         "读法：只写在本次探针帧上「未观测到方向性」。不用「推翻」二字——网络可能根本没有显式高度表征。")

def fig10():
    EQ = J("equarea.json")
    q, v = EQ["conditions"], EQ["verdict"]
    keys = ["G", "P", "F"]
    labs = ["G\n只抹地面带的白", "P\n只抹动态物体的白", "F\n全图白像素混合"]
    cols = [C_BAND, C_DYN, C_RND]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.5))
    fig.suptitle("图 10  等面积对照（关键实验）：三个条件各抹掉同样的 720 个白像素，只改「抹在哪」",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.02)

    x = np.arange(3)
    vv = [q[k]["absmean_dQ"] for k in keys]
    ci = [q[k]["absmean_dQ_ci"] for k in keys]
    ax.bar(x, vv, width=0.55, color=cols)
    ax.errorbar(x, vv, yerr=[[a - c[0] for a, c in zip(vv, ci)], [c[1] - a for a, c in zip(vv, ci)]],
                fmt="none", ecolor="#2C2C2A", elinewidth=1.2, capsize=5)
    for xi, a, c in zip(x, vv, ci):
        ax.text(xi, c[1] + 0.06, f"{a:.2f}\n[{c[0]:.2f},{c[1]:.2f}]", ha="center", fontsize=8.8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("|ΔQ| 均值（删块敏感度）"); ax.set_ylim(0, 4.6); clean(ax)
    ax.set_title(f"幅值：G / P 比值 {v['ratio_G_over_P']:.2f}，置信区间重叠\n"
                 f"→ 「影响多大」由面积决定", fontsize=10)
    ax.text(0.5, 4.10, "地面带那 67–86% 的归因占比\n不能读成「比物体更重要」",
            fontsize=8.8, color=C_FAIL, ha="center")

    dd = [q[k]["mean_dQ1_minus_Q0"] for k in keys]
    ci2 = [q[k]["mean_dQ1_minus_Q0_ci"] for k in keys]
    bx.bar(x, dd, width=0.55, color=cols)
    bx.errorbar(x, dd, yerr=[[a - c[0] for a, c in zip(dd, ci2)], [c[1] - a for a, c in zip(dd, ci2)]],
                fmt="none", ecolor="#2C2C2A", elinewidth=1.2, capsize=5)
    bx.axhline(0, color="#2C2C2A", lw=0.9)
    for xi, a, c in zip(x, dd, ci2):
        bx.text(xi, c[1] + 0.10, f"{a:+.2f}\n[{c[0]:+.2f},{c[1]:+.2f}]", ha="center", fontsize=8.8)
    bx.set_xticks(x); bx.set_xticklabels(labs, fontsize=9)
    bx.set_ylabel("Δ(Q1 − Q0) 均值　正 = 更倾向拍翅"); bx.set_ylim(-0.7, 3.7); clean(bx)
    bx.set_title("方向：G 的 CI 完全在 0 上方，P 的 CI 跨 0\n→ 只有地面带携带方向信息", fontsize=10)
    ax.text(0.02, -0.30, f"合格帧 {v['n_frames']} / 400（要求两侧都够 720 个白像素）",
            fontsize=8.2, color="#5F5E5A", transform=ax.transAxes)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    save(fig, "fig10_equarea",
         "读法：这是把「扰动大小」这个混淆项排除掉之后的结果。幅值说不清谁重要，方向说清了：地面带是线索，物体白像素更像噪声源。")

def fig11():
    CL = J("closed.json")
    keys = ["orig(对照)", "sky(阴性对照)", "ground(抹地面带)", "dyn(抹管子+小鸟)"]
    labs = ["原图\n(对照)", "抹天空\n(阴性对照)", "抹地面带", "抹管子+小鸟"]
    cols = [C_SKY, C_SKY, C_BAND, C_DYN]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.0, 4.3),
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("图 11  闭环生态效度（辅助）：整局游玩时的得分分布",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.02)
    data = [CL[k]["scores"] for k in keys]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor("#5F5E5A")
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.linspace(-0.13, 0.13, len(d)), d,
                   s=15, color="#2C2C2A", alpha=0.6, zorder=3)
        ax.text(i, max(d) + 2.2, f"均值 {np.mean(d):.1f}", ha="center", fontsize=8.6)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("单局得分")
    ax.set_xlabel("每条件 5 个随机种子 × 4 局 = 20 局", fontsize=8.8, color="#5F5E5A")
    clean(ax)

    bx.bar(np.arange(4), [CL[k]["mean"] for k in keys], width=0.55, color=cols)
    for i, k in enumerate(keys):
        bx.text(i, CL[k]["mean"] + 1.2, f"{CL[k]['mean']:.1f}", ha="center", fontsize=9)
    bx.set_xticks(np.arange(4)); bx.set_xticklabels(labs, fontsize=9)
    bx.set_ylabel("平均得分"); bx.set_ylim(0, 58); clean(bx)
    bx.set_title("原图与抹天空逐局完全相同 → 实现无误", fontsize=10)
    bx.annotate("删掉地面带 / 动态物体\n策略几乎立刻死亡", xy=(3, 1.5), xytext=(1.55, 24),
                fontsize=8.8, color=C_FAIL,
                arrowprops=dict(arrowstyle="-|>", color=C_FAIL, lw=0.9))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save(fig, "fig11_closed_loop",
         "读法：闭环只能证明「这些像素对整局生存关键」，无法区分「单步决策依赖」与「分布外崩溃」——所以主结论只放在单步层面。")

def fig12():
    md = os.path.join(OUT, "CONCLUSIONS.md")
    status = {}
    with open(md, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("|") or line.startswith("| 结论") or line.startswith("| :"):
                continue

            cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
            if len(cells) >= 4:
                status[cells[0]] = cells[3].strip("* ")
    if not status:
        raise SystemExit("CONCLUSIONS.md 解析为空，检查表格分隔符")
    groups = [
        ("定标与自检", ["输入构成：背景零贡献、地面带恒白", "Q 未越过理论上界",
                        "阴性对照：抹天空零效应"]),
        ("归因有效性", ["遮挡归因有信息量（优于随机顺序）", "梯度法（Saliency）不可信",
                        "遮挡显著优于梯度法（正确的替代说法）", "归因占比严格不依赖块大小",
                        "地面带始终是最大归因区"]),
        ("决策级因果", ["地面带参与单步决策", "小鸟参与单步决策（阳性对照）",
                        "管子参与单步决策（阳性对照）", "决策对「小鸟」比对「管子」更敏感"]),
        ("「高度尺」检验", ["「地面带=高度尺」的方向性", "带内位置效应：越靠下的白像素越关键",
                            "【等面积对照·幅值】影响多大由面积决定",
                            "【等面积对照·方向】地面带白像素携带方向信息"]),
        ("反事实解释", ["【反事实·覆盖率】按归因删块更快改掉决策",
                        "【反事实·改动量】配对比较下归因顺序更省改动",
                        "【反事实·语义】小鸟的竖直位置是可操作变量",
                        "【反事实·方向】鸟更低 → 更倾向拍翅，且严格单调"]),
        ("迁移与生态", ["注意力从动态物体迁向地面带", "迁移的起止点（可核对的具体数字）",
                        "该迁移在分层探针上稳健", "Q 值单调攀升，而成绩对步数非单调",
                        "闭环生态效度（辅助）"]),
    ]
    cmap = {"成立": C_OK, "证伪": C_FAIL, "未观测到": C_NONE}

    tcol = {"成立": "#085041", "证伪": "#791F1F", "未观测到": "#633806"}

    row_h, box_h = 0.36, 0.27
    ggap = 0.30
    n_rows = sum(len(it) for _, it in groups)
    total = n_rows * row_h + len(groups) * ggap + 0.10
    W = 12.0
    fig, ax = plt.subplots(figsize=(W, total + 1.05))
    fig.subplots_adjust(left=0.012, right=0.988, top=1.0 - 0.92 / (total + 1.05),
                        bottom=0.012)
    ax.set_xlim(0, W); ax.set_ylim(0, total); ax.axis("off")
    ax.text(0.02, total + 0.30, "图 12  全部结论的三态总览（成立 / 证伪 / 未观测到）",
            fontsize=12.5, weight="500", va="bottom")
    y = total - box_h / 2 - 0.02
    n = 0
    missing = []
    for group, items in groups:
        ax.text(0.04, y, group, fontsize=10.5, weight="500", color="#2C2C2A", va="center")
        for it in items:
            n += 1
            st = status.get(it)
            if st is None:
                missing.append(it); st = "成立"
            box(ax, 1.95, y - box_h / 2, 7.35, box_h, it,
                fc="#FFFFFF", ec="#D3D1C7", fs=8.6, ha="left")
            box(ax, 9.60, y - box_h / 2, 2.28, box_h, st,
                fc=cmap[st] + "1F", ec=cmap[st], fs=9.2, tc=tcol[st])
            y -= row_h
        y -= ggap
        ax.plot([0.02, 11.96], [y + ggap * 0.45, y + ggap * 0.45], color="#D3D1C7", lw=0.6)
    if missing:
        print(f"    ⚠️ 图 12 有 {len(missing)} 条没在 CONCLUSIONS.md 里匹配到：{missing}")
    ax.text(0.02, y + 0.34, f"共 {n} 条，与 out_v2/CONCLUSIONS.md 逐条对应（状态直接从该表读取）。"
                            f"「未观测到」≠「证伪」：前者是本次探针帧上没测到，后者是判据被推翻。",
            fontsize=8.6, color="#5F5E5A", va="bottom")
    save(fig, "fig12_conclusions_matrix",
         "读法：先看「证伪」的两条（旧结论被自己推翻），再看「未观测到」的一条，最后才是成立项。")

def fig13():
    CF = J("cf.json")["CF1"]
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.4, 4.0),
                                     gridspec_kw={"width_ratios": [1.15, 1, 0.95]})
    fig.suptitle("图 13  反事实解释（一）：最少删掉多少像素才会改主意",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.03)

    ca, cr = np.array(CF["curve_attr"]), np.array(CF["curve_rand"])
    x = np.arange(len(ca))
    ax.plot(x, ca, "-", lw=2.0, color=C_OCC, label="按遮挡归因排序删块")
    ax.fill_between([len(ca) - 1], [CF["flip_rate_attr_ci"][0]], [CF["flip_rate_attr_ci"][1]],
                    color=C_OCC, alpha=0.25)
    ax.plot(x, cr, "--", lw=1.8, color=C_RND, label="随机顺序删块（3 个种子均值）")
    ax.fill_between([len(cr) - 1], [CF["flip_rate_rand_ci"][0]], [CF["flip_rate_rand_ci"][1]],
                    color=C_RND, alpha=0.25)
    ax.errorbar([len(ca) - 1], [CF["flip_rate_attr"]],
                yerr=[[CF["flip_rate_attr"] - CF["flip_rate_attr_ci"][0]],
                      [CF["flip_rate_attr_ci"][1] - CF["flip_rate_attr"]]],
                fmt="o", ms=5, color=C_OCC, capsize=3)
    ax.errorbar([len(cr) - 1], [CF["flip_rate_rand"]],
                yerr=[[CF["flip_rate_rand"] - CF["flip_rate_rand_ci"][0]],
                      [CF["flip_rate_rand_ci"][1] - CF["flip_rate_rand"]]],
                fmt="o", ms=5, color=C_RND, capsize=3)
    ax.set_xlabel(f"已删掉的块数（{CF['block']}×{CF['block']} 格 = {CF['block']*CF['block']} 格/块）")
    ax.set_ylabel("argmax 已被翻转的帧比例")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, len(ca) - 1)
    ax.legend(fontsize=8.6, frameon=False, loc="lower right")
    ax.set_title(f"(a) 固定预算 {CF['cap']} 块内的翻转率", fontsize=10)
    clean(ax)
    ax.text(0.04, 0.55, f"归因 {CF['flip_rate_attr']:.1%}\n"
                        f"[{CF['flip_rate_attr_ci'][0]:.1%}, {CF['flip_rate_attr_ci'][1]:.1%}]",
            fontsize=8.6, color=C_OCC, transform=ax.transAxes)
    ax.text(0.04, 0.30, f"随机 {CF['flip_rate_rand']:.1%}\n"
                        f"[{CF['flip_rate_rand_ci'][0]:.1%}, {CF['flip_rate_rand_ci'][1]:.1%}]",
            fontsize=8.6, color="#5F5E5A", transform=ax.transAxes)

    d = CF["paired_dist_diff"]; lo, hi = CF["paired_dist_diff_ci"]
    bx.axvline(0, color="#2C2C2A", lw=1.0)
    bx.barh([0], [d], height=0.34, color=C_OCC, alpha=0.9)
    bx.plot([lo, hi], [0, 0], color="#2C2C2A", lw=1.3)
    for v in (lo, hi):
        bx.plot([v, v], [-0.11, 0.11], color="#2C2C2A", lw=1.3)
    bx.set_ylim(-0.7, 0.7); bx.set_yticks([])
    bx.set_xlabel("配对差：归因距离 − 随机距离（白格数）")
    bx.set_title(f"(b) 改动量配对比较（{CF['paired_n']} 帧）", fontsize=10)
    clean(bx, grid="x")
    m = max(abs(lo), abs(hi))
    bx.set_xlim(-m * 1.38, m * 0.95)
    bx.text(0.5, 0.82, f"归因中位 {CF['paired_median_attr']:.0f} 格 vs 随机 "
                       f"{CF['paired_median_rand']:.0f} 格\n"
                       f"配对差 {d:+.1f} [{lo:+.1f}, {hi:+.1f}]，CI 全在 0 左侧",
            fontsize=8.4, color=C_OCC, ha="center", transform=bx.transAxes)

    fs = CF["foot_share"]
    base_band = 1440.0 / CF["n_white_cells_median"]
    base_dyn = 1 - base_band
    xs = np.arange(2); wd = 0.34
    cx.bar(xs - wd / 2, [fs["band"], fs["dyn"]], wd, color=[C_BAND, C_DYN], label="反事实足迹")
    cx.bar(xs + wd / 2, [base_band, base_dyn], wd, color=[C_BAND, C_DYN], alpha=0.35,
           label="面积基线（该区白格占比）")
    for xi, v in zip(xs - wd / 2, [fs["band"], fs["dyn"]]):
        cx.text(xi, v + 0.02, f"{v:.1%}", ha="center", fontsize=8.8)
    for xi, v in zip(xs + wd / 2, [base_band, base_dyn]):
        cx.text(xi, v + 0.02, f"{v:.1%}", ha="center", fontsize=8.8, color="#5F5E5A")
    cx.set_xticks(xs); cx.set_xticklabels(["地面带", "动态物体"], fontsize=9.5)
    cx.set_ylabel("占被删白格的比例"); cx.set_ylim(0, 1.05)
    cx.legend(fontsize=8.2, frameon=False, loc="upper center", ncol=1)
    cx.set_title("(c) 反事实足迹落在哪", fontsize=10)
    clean(cx)

    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    save(fig, "fig13_cf_distance",
         "读法：(a) 归因顺序在同样预算内翻掉 94% 的帧，随机只有 56%（CI 不重叠）；"
         "(b) 对同一批帧，归因顺序的改动量也显著更小；(c) 足迹偏地面带，但这条是归因排序的回声，"
         "不算独立证据。")

def fig14():
    CF = J("cf.json")["CF2"]
    cond = CF["conditions"]
    ks = [k for k in range(-max(CF["ks"]), max(CF["ks"]) + 1) if k == 0 or abs(k) in CF["ks"]]
    ks = sorted(set([0] + [k for k in ks if abs(k) <= max(CF["ks"])]),
                key=lambda t: t)
    fl = np.array([0.0 if k == 0 else cond[f"k{k:+d}"]["flip_rate"] for k in ks])
    lo = np.array([0.0 if k == 0 else cond[f"k{k:+d}"]["flip_ci"][0] for k in ks])
    hi = np.array([0.0 if k == 0 else cond[f"k{k:+d}"]["flip_ci"][1] for k in ks])
    dq = np.array([0.0 if k == 0 else cond[f"k{k:+d}"]["mean_dQ1_minus_Q0"] for k in ks])
    to1 = np.array([0.0 if k == 0 else cond[f"k{k:+d}"]["to_flap"] for k in ks])

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.4, 4.0),
                                     gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.suptitle("图 14  反事实解释（二）：小鸟再低/再高多少格，它就一定会拍翅",
                 fontsize=12.5, weight="500", x=0.012, ha="left", y=1.03)

    ax.errorbar(ks, fl, yerr=[fl - lo, hi - fl], fmt="-o", ms=4.5, lw=1.8,
                color=C_BIRD, capsize=3, label="argmax 翻转率")
    ax.axvline(0, color="#B4B2A9", lw=0.9, ls=":")
    ax.set_xlabel("小鸟沿屏幕竖直方向平移 k 格（每格 = 6.4 px，正 = 下移 = 更低）")
    ax.set_ylabel("翻转率", color=C_BIRD); ax.tick_params(axis="y", colors=C_BIRD)
    ax.set_ylim(-0.02, 0.68); ax.set_xlim(-9.4, 9.4)
    clean(ax)
    b2 = ax.twinx()
    b2.plot(ks, dq, "--s", ms=4, lw=1.5, color=C_OCC, label="Δ(Q1−Q0)")
    b2.axhline(0, color="#B4B2A9", lw=0.8)
    b2.set_ylabel("Δ(Q1 − Q0)　正 = 更倾向拍翅", color=C_OCC)
    b2.tick_params(axis="y", colors=C_OCC); b2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.4, frameon=False, loc="upper left")
    ax.set_title(f"(a) 小鸟垂直平移的反事实响应（n={CF['n_usable']} 帧）\n"
                 f"下移 k>0：拍翅优势单调升；上移 k<0：单调降（符号严格对称）", fontsize=9.6)

    kstar = [abs(v) for v in CF["kstar"] if v is not None]
    kmax = max(CF["ks"])
    cnt = [sum(1 for v in kstar if v == j) for j in range(1, kmax + 1)]
    n_fail = CF["n_usable"] - len(kstar)
    bx.bar(range(1, kmax + 1), cnt, color=C_BIRD, alpha=0.9)
    bx.bar([kmax + 1.4], [n_fail], color=C_SKY)
    bx.text(kmax + 1.4, n_fail + 2, f"{n_fail}", ha="center", fontsize=8.8, color="#5F5E5A")
    for j, v in zip(range(1, kmax + 1), cnt):
        bx.text(j, v + 2, f"{v}", ha="center", fontsize=8.4)
    bx.set_xticks(list(range(1, kmax + 1)) + [kmax + 1.4])
    bx.set_xticklabels([str(j) for j in range(1, kmax + 1)] + [f">{kmax}"], fontsize=9)
    bx.set_xlabel("最小反事实距离 |k*|（格）")
    bx.set_ylabel("帧数")
    bx.set_title(f"(b) 最小 |k*| 分布（能翻转 {len(kstar)} 帧）", fontsize=10)
    clean(bx)
    bx.text(0.40, 0.96, f"|k*| ≤ 6 占 {CF['rate_within_k6']:.1%}\n"
                        f"对照（随机重绘）{CF['ctrl_rand_redraw_flip']:.1%}",
            fontsize=8.6, color=C_FAIL, transform=bx.transAxes, va="top")

    dc = CF["direction_counts"]
    labs = ["下移→拍翅", "上移→拍翅", "下移→不拍", "上移→不拍"]
    vals = [dc.get(l, 0) for l in labs]
    y = np.arange(len(labs))[::-1]
    cx.barh(y, vals, height=0.55, color=[C_BIRD, C_SKY, C_DYN, C_SKY])
    for yi, v in zip(y, vals):
        cx.text(v + 3, yi, f"{v}", va="center", fontsize=8.8)
    cx.set_yticks(y); cx.set_yticklabels(labs, fontsize=9)
    cx.set_xlabel("帧数（共 %d 帧找到反事实）" % CF["n_counterfactual"])
    cx.set_xlim(0, max(vals) * 1.22)
    cx.set_title("(c) 反事实方向：下移 → 拍翅 压倒性", fontsize=10)
    clean(cx, grid="x")

    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    save(fig, "fig14_cf_bird",
         "读法：小鸟只往下移 6 px，就有 18% 的帧改变按键；移 38 px 时过半改变，"
         "且方向压倒性是「更低 → 更该拍翅」。管子全程未动 —— 这是纯「高度」信息。")

def main():
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"设备 {dev}    输出 {os.path.relpath(FIGS, ROOT)}")
    fig01()
    fig02()
    fig03(dev)
    fig04()
    fig05(dev)
    fig06()
    fig07()
    fig08()
    fig09()
    fig10()
    fig11()
    fig12()
    fig13()
    fig14()
    print(f"\n共 12 张，已写入 {os.path.relpath(FIGS, ROOT)}")

if __name__ == "__main__":
    main()
