import copy
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGS = os.path.join(HERE, "figures")
CACHE = os.path.join(OUT, "explain_fig68_cache.pkl")
os.makedirs(FIGS, exist_ok=True)

def _imp(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, m)
    spec.loader.exec_module(m)
    return m

xl = _imp("explain_lmut")
probe, sleaf = xl.probe, xl.sleaf
NF, SZ, GROUND_Y0 = xl.NF, xl.SZ, xl.GROUND_Y0

K = 1024
CMAP_ATTR = "YlOrRd"
C_FLAP, C_NOFLAP = "#A32D2D", "#185FA5"
NAME = {1: "拍翅 Q(s,1)", 0: "不拍翅 Q(s,0)"}
FRAME_LABEL = ["帧0（最新）", "帧1", "帧2", "帧3（最早）"]

def analyze(tree, Q_col, S, Ftr, gidx, n_cand, bird):
    t = xl.fit_node_weights(copy.deepcopy(tree), Ftr, Q_col, S)
    Inf, G = xl.influence(t, Ftr, Q_col, gidx, NF * SZ * SZ)

    thr = float(sum(Inf.values()) / n_cand)
    vals = np.array(sorted(Inf.values(), reverse=True))
    top = sorted(Inf.items(), key=lambda kv: -kv[1])[:20]

    total = float(G.sum()) or 1.0
    quad = {"左上": float(G[:, :SZ // 2, :SZ // 2].sum()),
            "右上": float(G[:, SZ // 2:, :SZ // 2].sum()),
            "左下": float(G[:, :SZ // 2, SZ // 2:].sum()),
            "右下": float(G[:, SZ // 2:, SZ // 2:].sum())}
    qsum = sum(quad.values()) or 1.0

    in_bird = sum(1 for i, _ in top
                  if bird["x"][0] <= xl.coords(gidx[i])[1] <= bird["x"][1]
                  and bird["y"][0] <= xl.coords(gidx[i])[2] <= bird["y"][1])

    return {"Inf": Inf, "G": G, "thr": thr, "leaves": int(tree["leaves"]),
            "n_split_feats": int(len(Inf)), "inf_total": float(sum(Inf.values())),
            "n_above_thr": int((vals > thr).sum()),
            "top_pixels": [{"idx": int(gidx[i]), "frame": xl.coords(gidx[i])[0],
                            "x": xl.coords(gidx[i])[1], "y": xl.coords(gidx[i])[2],
                            "inf": float(v)} for i, v in top],
            "frame_share": [float(x) for x in (G.sum(axis=(1, 2)) / total)],
            "quadrant": {k: float(v) for k, v in quad.items()},
            "left_half": float((quad["左上"] + quad["左下"]) / qsum),
            "top_half": float((quad["左上"] + quad["右上"]) / qsum),
            "top20_in_bird_box": int(in_bird),
            "split_counts": [int(x) for x in xl.split_counts(tree, gidx)]}

def replay(seed, need_t):
    need = set(int(t) for t in need_t)
    G = probe.new_game_module()
    net, dev = probe.load_net()
    probe.seed_env(G, seed)
    env = G.GameState()
    a0 = np.zeros(2); a0[0] = 1
    frame, _, term = env.frame_step(a0, render=False)
    st = probe.init_state(frame)
    states, frames, t = {}, {}, 0
    while not term:
        if t in need:
            states[t] = st.copy()
            frames[t] = np.transpose(frame, (1, 0, 2)).copy()
        q = probe.q_batch(net, st[None], dev)[0]
        a = int(q.argmax())
        av = np.zeros(2); av[a] = 1
        frame, _r, term = env.frame_step(av, render=False)
        st = probe.next_state(st, frame)
        t += 1
    return states, frames

def _panel(ax, ghost, Gk, title, vmax):
    ax.imshow(ghost, cmap="Greys", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    ax.imshow(np.where(Gk.T > 0, Gk.T, np.nan), cmap=CMAP_ATTR, vmin=0.0, vmax=vmax,
              origin="upper", interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

def draw_influence(res, bird, frame_mean):
    vmax = max(r["G"][k].max() for r in res.values() for k in range(NF)) or 1e-9
    fig = plt.figure(figsize=(15.2, 8.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 0.72], hspace=0.42,
                          left=0.040, right=0.988, top=0.885, bottom=0.235, wspace=0.10)

    first_ax = None
    for k in range(NF):
        ax = fig.add_subplot(gs[0, k])
        _panel(ax, frame_mean[k].T / 255.0, res[1]["G"][k], FRAME_LABEL[k], vmax)
        xl._bird_box(ax, bird, color="#0F6E5C")
        if k == 0:
            first_ax = ax
    for px in res[1]["top_pixels"][:8]:
        if px["frame"] == 0:
            first_ax.plot(px["x"], px["y"], marker="o", ms=4.2, mfc="none",
                          mec="#185FA5", mew=1.2, zorder=7)
    ax = fig.add_subplot(gs[0, 4])
    Sm = res[1]["G"].sum(axis=0)
    _panel(ax, frame_mean[0].T / 255.0, Sm, "四帧汇总", Sm.max() or 1e-9)
    xl._bird_box(ax, bird, color="#0F6E5C")
    fig.text(0.007, 0.80, NAME[1], rotation=90, va="center",
             fontsize=11, fontweight="bold", color=C_FLAP)

    xs = np.arange(NF)

    def _frame_bars(ax, vals, title, ylab):
        for j, a in enumerate((1, 0)):
            ax.bar(xs + (j - 0.5) * 0.38, vals[a], width=0.38,
                   color=(C_FLAP, C_NOFLAP)[j], label=NAME[a])
        ax.set_xticks(xs); ax.set_xticklabels(["帧0", "帧1", "帧2", "帧3"], fontsize=9)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(True, color="#D3D1C7", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 0:2])
    _frame_bars(ax, {a: res[a]["split_counts"] for a in (1, 0)}, "各帧的分裂次数", "分裂节点数")

    ax = fig.add_subplot(gs[1, 2:4])
    _frame_bars(ax, {a: np.array(res[a]["frame_share"]) * 100 for a in (1, 0)},
                "各帧的影响占比", "占比 (%)")

    ax = fig.add_subplot(gs[1, 4])
    ks = ["左上", "右上", "左下", "右下"]
    xq = np.arange(4)
    for j, a in enumerate((1, 0)):
        q = res[a]["quadrant"]; tot = sum(q.values()) or 1.0
        ax.bar(xq + (j - 0.5) * 0.38, [100 * q[k] / tot for k in ks], width=0.38,
               color=(C_FLAP, C_NOFLAP)[j], label=NAME[a])
    ax.set_xticks(xq); ax.set_xticklabels(ks, fontsize=9)
    ax.set_ylabel("占比 (%)"); ax.set_title("影响的空间分布", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, color="#D3D1C7", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    sc1, sc0 = res[1]["split_counts"], res[0]["split_counts"]
    fr1 = "/".join(f"{round(x * 100)}%" for x in res[1]["frame_share"])
    txt = (
        f"空间分布：左半 {res[1]['left_half']:.0%}、上半 {res[1]['top_half']:.0%}、"
        f"左上象限 {100 * res[1]['quadrant']['左上'] / sum(res[1]['quadrant'].values()):.0f}%；"
        f"前 20 大影响像素中 {res[1]['top20_in_bird_box']} 个落入实测小鸟框"
        f"（屏X {bird['x'][0]}–{bird['x'][1]}，屏Y {bird['y'][0]}–{bird['y'][1]}）\n"
        f"时间分布：分裂次数 {sc1[0]}/{sc1[1]}/{sc1[2]}/{sc1[3]}（帧0→帧3），"
        f"最新帧占 {sc1[0] / sum(sc1):.0%}；影响量占比 {fr1}\n"
        f"不拍翅树对照：帧0 占 {sc0[0] / sum(sc0):.0%}、左上象限 "
        f"{100 * res[0]['quadrant']['左上'] / sum(res[0]['quadrant'].values()):.0f}%，"
        f"影响总和仅为拍翅树的 2% —— 拍翅依赖画面，不拍翅近乎缺省动作\n"
        "灰底 = 该帧平均画面；青框 = 实测小鸟位置；深色 = 影响大；蓝圈 = 前 8 大影响像素（画在帧0）"
    )
    fig.text(0.012, 0.200, txt, fontsize=9.2, va="top", ha="left", linespacing=1.8)

    fig.suptitle("图 4-1　特征影响：四个输入帧与四帧汇总（横轴 = 屏幕横坐标，纵轴 = 屏幕纵坐标，"
                 "画面下缘为地面；色深表示影响大小）", fontsize=12, y=0.968)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig4-1_特征影响.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  → figs/fig4-1_特征影响.png / .pdf")

def draw_superpixel(paths, colors, bird, gidx):
    fig = plt.figure(figsize=(15.0, 7.2))
    gs = fig.add_gridspec(len(paths), NF, left=0.175, right=0.995,
                          top=0.885, bottom=0.055, hspace=0.22, wspace=0.05)
    for r, pr in enumerate(paths):
        for k in range(NF):
            ax = fig.add_subplot(gs[r, k])
            ax.imshow(colors[r][k], origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(FRAME_LABEL[k], fontsize=10)
            n = 0
            for j in pr["hot"]:
                f, x, y = xl.coords(gidx[j])
                if f == k:
                    ax.plot(x * 3.6, y * 6.4, marker="*", ms=14, color="#E24B4A",
                            mec="white", mew=0.8, zorder=5)
                    n += 1
            if n:
                ax.set_xlabel(f"红星 {n} 个", fontsize=9, color="#A32D2D")
            bx0, bx1 = bird["x"][0] * 3.6, bird["x"][1] * 3.6
            by0, by1 = bird["y"][0] * 6.4, bird["y"][1] * 6.4
            ax.add_patch(Rectangle((bx0 - 3.6, by0 - 6.4), bx1 - bx0 + 7.2, by1 - by0 + 12.8,
                                   fill=False, ec="#00E5FF", lw=1.4, zorder=6))
        fig.text(0.011, 0.47 - r * 0.42, pr["label"], rotation=90, va="center",
                 fontsize=11.5, fontweight="bold", color=C_FLAP)
        fig.text(0.042, 0.47 - r * 0.42,
                 f"路径 {len(pr['path'])} 步，高亮 {len(pr['hot'])}\n按帧 {pr['byf']}",
                 fontsize=8.8, va="center", ha="left", color="#5F5E5A")

    fig.suptitle("图 4-2　超像素高亮：根→目标叶分裂路径上影响超阈值的像素（红星）；"
                 "底图为真实游戏画面，青框为实测小鸟位置", fontsize=12, y=0.965)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig4-2_超像素高亮.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  → figs/fig4-2_超像素高亮.png / .pdf")

def draw_consistency(blk_tree, occ):
    bt, oc = blk_tree.ravel(), occ.ravel()
    hot = bt > 0
    rho, pv = xl._rho(bt[hot], oc[hot])
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(bt[hot], oc[hot], s=34, color="#185FA5", alpha=0.78,
               edgecolor="white", linewidth=0.6, zorder=3)
    k, b = np.polyfit(bt[hot], oc[hot], 1)
    xr = np.linspace(bt[hot].min(), bt[hot].max(), 50)
    ax.plot(xr, k * xr + b, color="#A32D2D", lw=1.6, zorder=4)
    ax.set_xlabel("模型树给出的块级影响", fontsize=11)
    ax.set_ylabel("模型自身的遮挡敏感度  |ΔQ|", fontsize=11)
    ax.set_title(f"图 4-3　解释一致性（{int(hot.sum())} 个有效块）\n"
                 f"秩相关 ρ = {rho:.3f}，p = {pv:.1e}", fontsize=12)
    ax.grid(True, color="#D3D1C7", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig4-3_解释一致性.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  → figs/fig4-3_解释一致性.png / .pdf")
    return {"rho": rho, "p": pv, "n_blocks": int(hot.sum())}

def compute():
    Xtr, Xte, Ftr, Fte, Qtr, Qte, eid_tr, ti, vi, cand, gidx, order = xl.prepare()
    n_cand = int(cand.sum())

    fr0 = Xte[0, 0].astype(float) / 255.0
    assert fr0[:, GROUND_Y0:].mean() > 0.9 > fr0[:, :GROUND_Y0].mean(), \
        "方向自检失败：轴1 不再是屏幕Y，方位结论不可信"
    print("【方向自检】通过：轴1 = 屏幕Y（地面带在底部），画图一律 .T")

    bird = xl.locate_bird(Xtr, eid_tr, np.random.default_rng(0))
    print(f"【小鸟实测】{bird['n']} 帧（漏检 {bird['miss']}）；屏X∈[{bird['x'][0]},{bird['x'][1]}] "
          f"屏Y∈[{bird['y'][0]},{bird['y'][1]}]，中心X={bird['cx_mean']:.0f}\n")

    print("【建树】逐动作一棵（论文 Fig.4 口径）")
    trees = [sleaf.fit(Ftr[ti], Qtr[ti, a], "full", None) for a in (0, 1)]
    order_a = {1: order, 0: sleaf.screening(Ftr[ti], Qtr[ti, 0])}
    print(f"  拍翅树 {trees[1]['leaves']} 叶 | 不拍翅树 {trees[0]['leaves']} 叶")
    print("  不拍翅树的节点权重用该动作自己的目标重筛特征序\n")

    print(f"【特征影响】K={K}")
    res = {}
    for a in (1, 0):
        res[a] = analyze(trees[a], Qtr[:, a], order_a[a][:K], Ftr, gidx, n_cand, bird)
        r = res[a]; sc = r["split_counts"]
        print(f"  {NAME[a]}：分裂特征 {r['n_split_feats']} 个，影响总和 {r['inf_total']:.2f}，"
              f"超阈值 {r['n_above_thr']} 个")
        print(f"      分裂次数按帧 {sc} → 最新帧占 {sc[0] / max(sum(sc), 1):.0%}"
              f" | 影响量按帧 {[round(x * 100, 1) for x in r['frame_share']]}")
        print(f"      左半 {r['left_half']:.0%} · 上半 {r['top_half']:.0%} · "
              f"前 20 大像素 {r['top20_in_bird_box']}/20 落在小鸟框")
    print()

    uniq, cnts = np.unique(eid_tr, return_counts=True)
    e_long = int(uniq[np.argmax(cnts)])
    idx_ep = np.where(eid_tr == e_long)[0]
    ords = sorted(set(max(0, min(o, len(idx_ep) - 1)) for o in (1, int(len(idx_ep) * 0.6))))
    paths = []
    for lab, o in zip(["回合早期", "回合中期"], ords):
        pk = int(idx_ep[o])
        path, _leaf = xl.path_of(trees[1], Ftr[pk])
        hot = [j for j in path if res[1]["Inf"].get(j, 0.0) > res[1]["thr"]]
        byf = [sum(1 for j in hot if xl.coords(gidx[j])[0] == k) for k in range(NF)]
        paths.append({"label": lab, "pk": pk, "ordinal": int(o), "path": path,
                      "hot": hot, "byf": byf})
        print(f"  {lab}（回合{e_long} 第{o}个采样）：路径 {len(path)} 步，高亮 {len(hot)} 个，按帧 {byf}")

    return {"res": res, "bird": bird, "paths": paths, "gidx": gidx,
            "frame_mean": Xtr.mean(axis=0), "e_long": e_long,
            "ords": ords, "n_cand": n_cand}

def main():
    t0 = time.time()
    if os.path.exists(CACHE) and "--refit" not in sys.argv:
        with open(CACHE, "rb") as fh:
            d = pickle.load(fh)
        print(f"【复用缓存】{CACHE}（要重算加 --refit）")
    else:
        d = compute()
        with open(CACHE, "wb") as fh:
            pickle.dump(d, fh, protocol=4)
        print(f"【写缓存】{CACHE}")

    res, bird, gidx = d["res"], d["bird"], d["gidx"]

    print("\n【出图】fig4-1 特征影响")
    draw_influence(res, bird, d["frame_mean"])

    print("【出图】fig4-2 超像素高亮")
    need = set()
    for o in d["ords"]:
        tt = o * probe.STRIDE
        need.update(tt - k for k in range(NF) if tt - k >= 0)
    states, frames = replay(1000 + d["e_long"], need)

    z = np.load(os.path.join(OUT, "data.npz"))
    Xtr = z["Xtr"]
    all_ok = True
    for pr in d["paths"]:
        tt = pr["ordinal"] * probe.STRIDE
        same = np.array_equal(np.asarray(states[tt], np.uint8), Xtr[pr["pk"]])
        print(f"   对齐校验：{pr['label']}（t={tt}）重放状态 == data.npz → {same}")
        all_ok &= bool(same)
    if not all_ok:
        raise RuntimeError("重放与 data.npz 不一致：seed / 采样顺序对不上，底图会错位，已中止")

    colors = [[frames[max(pr["ordinal"] * probe.STRIDE - k, 0)] for k in range(NF)]
              for pr in d["paths"]]
    draw_superpixel(d["paths"], colors, bird, gidx)

    print("【出图】fig4-3 解释一致性")
    maps = np.load(os.path.join(OUT, "explain_maps.npz"))
    cons = draw_consistency(maps["blk_tree"], maps["occ_grid"])

    out = {"K": K, "n_cand": d["n_cand"], "bird": bird,
           "leaves": {str(a): res[a]["leaves"] for a in (1, 0)},
           "per_action": {str(a): {k: v for k, v in res[a].items() if k not in ("Inf", "G")}
                          for a in (1, 0)},
           "consistency": cons,
           "superpixel_paths": [{"label": pr["label"], "ordinal": pr["ordinal"],
                                 "path_len": len(pr["path"]), "n_hot": len(pr["hot"]),
                                 "byf": pr["byf"]} for pr in d["paths"]],
           "fig4_2_episode": d["e_long"]}
    with open(os.path.join(OUT, "explain_fig68.json"), "w", encoding="utf-8") as fh:
        json.dump(xl._clean(out), fh, ensure_ascii=False, indent=1)
    print(f"\n已写 out/explain_fig68.json 与 figs/fig4-1~4-3   用时 {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
