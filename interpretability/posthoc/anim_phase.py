import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(ROOT, "dqn"))

FRAME = 80

ASPECT = 512.0 / 288.0

REGION_NAMES_DEFAULT = ["地面带", "小鸟", "上管道", "下管道", "背景残差"]
REGION_COLORS_DEFAULT = ["#9AA0A6", "#D85A30", "#185FA5", "#1D9E75", "#E4E1DC"]

DENSITY_YLIM = (0.05, 20.0)

def _regions(npz):
    names = [str(v) for v in npz["region_names"]] if "region_names" in npz.files \
        else list(REGION_NAMES_DEFAULT)
    cols = [str(v) for v in npz["region_colors"]] if "region_colors" in npz.files \
        else list(REGION_COLORS_DEFAULT)
    return names, cols

def _to_share(rows):
    rows = np.asarray(rows, np.float64)
    return rows / np.maximum(rows.sum(axis=-1, keepdims=True), 1e-12) * 100.0

def _region_rows(acc, cnt, g=None):
    acc = np.asarray(acc, np.float64)
    cnt = np.asarray(cnt, np.float64)
    if g is None:
        num, den = acc.sum(0), cnt.sum(0)
    else:
        num, den = acc[g], cnt[g]
    return _to_share(num / np.maximum(den, 1.0)[:, None])

def _area_rows(acc, cnt, g=None):
    acc = np.asarray(acc, np.float64)
    cnt = np.asarray(cnt, np.float64)
    if g is None:
        num, den = acc.sum(0), cnt.sum(0)
    else:
        num, den = acc[g], cnt[g]
    return num / np.maximum(den, 1.0)[:, None]

def _density_lines(ax, x, share, area, colors, names, xlabel="", title="",
                   ms=3.2, ylim=DENSITY_YLIM, ff=6.6, ncol=3):
    xs = np.asarray(x, np.float64)
    for r in range(share.shape[1]):
        d = np.asarray(share[:, r], np.float64) / np.maximum(
            np.asarray(area[:, r], np.float64), 1e-9)
        ax.plot(xs, d, lw=1.8, marker="o", ms=ms, color=colors[r],
                label=names[r], markeredgewidth=0)
    ax.axhline(1.0, color="#888780", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_ylabel("归因密度（单位面积归因）", fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    if title:
        ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.20, which="both")
    h = ax.get_legend_handles_labels()[0]
    ax.legend(h + [plt.Line2D([], [], color="#888780", ls="--", lw=1.0)],
              list(names) + ["1 = 全图平均"], fontsize=ff, ncol=ncol,
              frameon=True, framealpha=0.82, edgecolor="none",
              loc="upper center", columnspacing=0.8, handlelength=1.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

def _avg(acc, cnt, g=0):
    acc = np.asarray(acc)
    cnt = np.asarray(cnt)
    P = acc.shape[1]
    res = np.zeros((P,) + acc.shape[2:], np.float32)
    for b in range(P):
        c = int(cnt[g, b])
        if c > 0:
            res[b] = acc[g, b] / c
    return res

def _feat_centroidY(avg_maps, G):
    m = np.abs(avg_maps[:, 0]).sum(axis=1)[:, :G]
    ys = np.arange(m.shape[1], dtype=np.float64)
    tot = m.sum(axis=1)
    return np.where(tot > 0, (m * ys[None, :]).sum(axis=1) / np.maximum(tot, 1e-12), np.nan)

def _overlay(img, attr, vmax, G, cmap_name="YlOrRd"):

    g = np.clip(1.0 - np.asarray(img, np.float64).T / 255.0, 0, 1)
    rgb = np.stack([g, g, g], axis=-1)
    a = np.array(attr, np.float64).T
    a[G:, :] = 0.0
    n = np.clip(a / max(float(vmax), 1e-9), 0, 1)
    heat = plt.get_cmap(cmap_name)(n)[..., :3]
    al = np.clip(n * 1.15, 0, 0.88)[..., None]
    return np.clip(rgb * (1 - al) + heat * al, 0, 1)

def _grab(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

def _save_gif(path, frames, fps):
    import imageio.v2 as iio
    try:
        iio.mimsave(path, frames, fps=int(round(fps)), loop=0)
    except TypeError:
        iio.mimsave(path, frames, duration=1.0 / float(fps), loop=0)
    return path

def gif_phase_sync(npz, group, out_name, tag, fps=4):
    G = int(npz["ground_band"][0])
    P = int(npz["phase_bins"][0])

    img = _avg(npz["acc_img"], npz["cnt_img"], group)
    att = _avg(npz["acc_dead"], npz["cnt_dead"], group)
    n = np.asarray(npz["cnt_dead"])[group]
    have = n > 0

    pool = np.abs(att[:, 0])[:, :, :G].ravel()
    vmax = float(np.percentile(pool, 99.5)) if pool.size else 1.0

    fig = plt.figure(figsize=(12.6, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.55], wspace=0.14)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[0, 2])
    for ax in (ax_f, ax_a):
        ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(G - 0.5, color="#555555", lw=0.8, alpha=0.6)
    im_f = ax_f.imshow(np.ones((FRAME, FRAME, 3)), vmin=0, vmax=1, aspect=ASPECT)
    im_a = ax_a.imshow(np.ones((FRAME, FRAME, 3)), vmin=0, vmax=1, aspect=ASPECT)

    names, cols = _regions(npz)

    reg = _region_rows(npz["acc_reg"], npz["cnt_reg"], group)
    are = _area_rows(npz["acc_area"], npz["cnt_area"], group)
    _density_lines(ax_d, range(P), reg, are, cols, names,
                   xlabel="管道相位 φ（0 = 刚过管）",
                   title="各区域归因密度随相位的变化（单位面积归因，对数纵轴）",
                   ms=3.2, ff=6.6, ncol=3)
    ax_d.set_xticks([0, 6, 12, 18, 24, 30, 35])
    ax_d.set_xlim(-0.5, P - 0.5)
    cursor = ax_d.axvline(0, color="#A32D2D", lw=1.6)

    sup = fig.suptitle("", fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.03, right=0.985)

    frames = []
    for b in range(P):
        if have[b]:
            im_f.set_data(np.stack([np.clip(1.0 - img[b][0].T / 255.0, 0, 1)] * 3, -1))
            im_a.set_data(_overlay(img[b][0], np.abs(att[b][0]), vmax, G))
            ax_f.set_title(f"平均画面   n={int(n[b])}", fontsize=10)
            ax_a.set_title(f"归因叠加   n={int(n[b])}", fontsize=10)
        else:
            im_f.set_data(np.ones((FRAME, FRAME, 3)))
            im_a.set_data(np.ones((FRAME, FRAME, 3)))
            ax_f.set_title("该相位无样本（留白）", fontsize=9, color="#A32D2D")
            ax_a.set_title("该相位无样本（留白）", fontsize=9, color="#A32D2D")
        cursor.set_xdata([b, b])
        sup.set_text(f"{tag}　｜　管道相位 φ = {b}/{P}（周期折叠）　｜　色阶固定 0~{vmax:.2f}")
        frames.append(_grab(fig))

    plt.close(fig)
    path = os.path.join(OUT, out_name)
    return _save_gif(path, frames, fps), len(frames)

def main():
    ap = argparse.ArgumentParser(description="读 phase_acc.npz 出相位同步 GIF（存活 / 死亡）")
    ap.add_argument("--npz", default=os.path.join(OUT, "phase_acc.npz"))
    ap.add_argument("--which", choices=["alive", "death", "both"], default="both")
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        raise SystemExit(f"找不到缓存 {args.npz} —— 请先跑 explain_dqn.py --only phase")
    npz = np.load(args.npz)
    print(f"读到缓存：{args.npz}")
    print(f"  键：{sorted(npz.files)}")

    if args.which in ("alive", "both"):
        p, n = gif_phase_sync(npz, 0, "gif_cycle_vs_death.gif",
                              "图 3-5　相位同步图：存活基线的周期归因", fps=args.fps)
        print(f"✔ {os.path.relpath(p, ROOT)}  （{n} 帧, {args.fps:g} fps）")
    if args.which in ("death", "both"):
        p, n = gif_phase_sync(npz, 1, "gif_death_phase.gif",
                              "图 3-7　死亡相位同步图：死亡区间（临死前 30 帧）的周期归因",
                              fps=args.fps)
        print(f"✔ {os.path.relpath(p, ROOT)}  （{n} 帧, {args.fps:g} fps）")

if __name__ == "__main__":
    main()
