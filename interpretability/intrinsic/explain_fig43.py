import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage, stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGS = os.path.join(HERE, "figures")
FRAME_CACHE = os.path.join(OUT, "fig4-3_frame0.npz")
SZ, BS = 80, 8
KX, KY = 288 / SZ, 512 / SZ
PLAY_Y0 = 60
EPISODE = 7

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "font.size": 9.5, "axes.linewidth": 0.9, "axes.unicode_minus": False,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_PT, C_FIT, C_GREY = "#3D6DA8", "#8A8A8A", "#5F5E5A"
C_A, C_B, C_BIRD = "#A32D2D", "#185FA5", "#FFD400"
XC0, XC1, YC0, YC1 = 8, 176, 64, 248

def _bird_bbox(f80):
    lab, n = ndimage.label(f80[:, :PLAY_Y0] > 0.5)
    best = None
    for L in range(1, n + 1):
        a, b = np.where(lab == L)
        wx, wy = a.max() - a.min() + 1, b.max() - b.min() + 1
        if not (8 <= wx <= 13 and 3 <= wy <= 7) or len(a) < 18:
            continue
        if best is None or len(a) > best[0]:
            best = (len(a), a, b)
    if best is None:
        return None
    _, a, b = best
    return int(a.min()), int(a.max()), int(b.min()), int(b.max())

def _hunt(seed=1000 + EPISODE):
    import probe
    G = probe.new_game_module()
    net, dev = probe.load_net()
    probe.seed_env(G, seed)
    env = G.GameState()
    a0 = np.zeros(2); a0[0] = 1
    frame, _, term = env.frame_step(a0, render=False)
    st = probe.init_state(frame)
    best, t = None, 0
    while not term and t < 900:
        bb = _bird_bbox(st[0].astype(np.float32) / 255.0)
        if bb:
            _, _, y0, y1 = bb
            hit = 0.0 if (y0 <= 24 <= y1) else 20.0
            sc = hit + abs((y0 + y1) / 2 - 24) + 0.0005 * abs(t - 60)
            if best is None or sc < best[0]:
                best = (sc, t, bb, np.transpose(frame, (1, 0, 2)).copy())
        q = probe.q_batch(net, st[None], dev)[0]
        av = np.zeros(2); av[int(q.argmax())] = 1
        frame, _r, term = env.frame_step(av, render=False)
        st = probe.next_state(st, frame)
        t += 1
    if best is None:
        raise RuntimeError("回放中未定位到小鸟：连通域规则失效，右图底图无法给出")
    print(f"  底图帧：t={best[1]}，小鸟框 屏X[{best[2][0]},{best[2][1]}] 屏Y[{best[2][2]},{best[2][3]}]")
    return best[3], np.array(best[2])

def load_frame(force=False):
    if os.path.exists(FRAME_CACHE) and not force:
        d = np.load(FRAME_CACHE)
        return d["color"], d["bb"]
    color, bb = _hunt()
    np.savez_compressed(FRAME_CACHE, color=color, bb=bb)
    print(f"  → 写缓存 {FRAME_CACHE}")
    return color, bb

def main():
    maps = np.load(os.path.join(OUT, "explain_maps.npz"))
    cache = pickle.load(open(os.path.join(OUT, "explain_fig68_cache.pkl"), "rb"))
    bird = cache["bird"]
    blk, occ = maps["blk_tree"], maps["occ_grid"]
    bt, oc = blk.ravel(), occ.ravel()
    hot = bt > 0
    vx, vy = bt[hot], oc[hot]
    rho, p_value = stats.spearmanr(vx, vy)

    key = []
    for idx in np.argsort(-bt)[:2]:
        f, bi, bj = np.unravel_index(idx, blk.shape)
        key.append({"bt": float(bt[idx]), "oc": float(oc[idx]),
                    "x": (bi * BS, bi * BS + BS), "y": (bj * BS, bj * BS + BS)})
    print(f"  A: 屏X{key[0]['x']} 屏Y{key[0]['y']} | B: 屏X{key[1]['x']} 屏Y{key[1]['y']}")
    print(f"  秩相关 rho = {rho:.3f}（{int(hot.sum())} 个有效块）；"
          f"小鸟实测框 屏X{bird['x']} 屏Y{bird['y']}")

    color, bb = load_frame(force="--reframe" in sys.argv)

    fig = plt.figure(figsize=(11.2, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.16,
                          left=0.078, right=0.968, top=0.90, bottom=0.115)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(vx, vy, s=26, facecolor=C_PT, alpha=0.70, edgecolor="white",
               linewidth=0.5, zorder=3)
    k, b = np.polyfit(vx, vy, 1)
    xr = np.linspace(vx.min(), vx.max(), 60)
    ax.plot(xr, k * xr + b, color=C_FIT, lw=1.3, ls="--", zorder=2, label="最小二乘拟合")
    for q, lab, col in zip(key, "AB", (C_A, C_B)):
        ax.scatter([q["bt"]], [q["oc"]], s=112, facecolor=col, edgecolor="white",
                   linewidth=1.5, zorder=6)
        ax.annotate(lab, (q["bt"], q["oc"]), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=12.5, fontweight="bold", color=col, zorder=7)
    ax.set_xlabel("树给出的块级影响（区块内像素特征影响力之和）")
    ax.set_ylabel("DQN 遮挡敏感度  $|\\Delta Q(s,1)|$")
    ax.set_xlim(-1.5, vx.max() + 4.5)
    ax.set_ylim(-0.12, vy.max() + 0.24)
    ax.grid(True, color="#D9D7CE", lw=0.55)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("Spearman $\\rho$ = %.3f   ($p$ = %.2g)\n"
                "n = %d 个树影响非零的区块" % (rho, p_value, int(hot.sum())),
                xy=(0.035, 0.96), xycoords="axes fraction", va="top", ha="left",
                fontsize=9.5, color=C_GREY)
    ax.annotate("A、B 的位置见右图", xy=(0.985, 0.06), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color=C_GREY)
    ax.text(-0.085, 1.04, "a", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")

    axb = fig.add_subplot(gs[0, 1])
    axb.imshow(color[YC0:YC1, XC0:XC1], origin="upper", interpolation="nearest",
               extent=(XC0, XC1, YC1, YC0), aspect="equal")
    for q, lab, col in zip(key, "AB", (C_A, C_B)):
        rx, ry = q["x"][0] * KX, q["y"][0] * KY
        rw, rh = (q["x"][1] - q["x"][0]) * KX, (q["y"][1] - q["y"][0]) * KY
        axb.add_patch(Rectangle((rx + 1.4, ry + 1.4), rw - 2.8, rh - 2.8,
                                fill=False, ec=col, lw=2.1, zorder=6))
        axb.text(rx + rw + 4, ry + rh / 2, lab, color=col, fontsize=13,
                 fontweight="bold", va="center", ha="left", zorder=7)
    bx0, bx1, by0, by1 = [int(v) for v in bb]
    axb.add_patch(Rectangle((bx0 * KX, by0 * KY), (bx1 - bx0 + 1) * KX, (by1 - by0 + 1) * KY,
                            fill=False, ec=C_BIRD, lw=1.5, ls=(0, (3.5, 2)), zorder=7))
    axb.annotate("实测小鸟", xy=(bx0 * KX, by0 * KY), xytext=(bx0 * KX - 9, (by0 + 0.5) * KY),
                 ha="right", va="top", fontsize=9.5, color=C_BIRD,
                 arrowprops=dict(arrowstyle="-", color=C_BIRD, lw=1.0), zorder=8)
    axb.text(0.022, 0.975, "帧 0（最新帧）· 局部放大", transform=axb.transAxes, color="white",
             fontsize=10.5, va="top", ha="left", zorder=9)
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_xlim(XC0, XC1); axb.set_ylim(YC1, YC0)
    for s in axb.spines.values():
        s.set_color("#2B2B2B"); s.set_linewidth(0.8)
    axb.text(-0.02, 1.015, "b", transform=axb.transAxes, fontsize=15,
             fontweight="bold", va="bottom", ha="left")

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig4-3_解释一致性.{ext}"),
                    dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  → figs/fig4-3_解释一致性.png / .pdf")

if __name__ == "__main__":
    main()
