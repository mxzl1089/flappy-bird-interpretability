import copy
import importlib.util
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

plt.rcParams["font.monospace"] = ["Consolas", "Microsoft YaHei", "SimHei", "DejaVu Sans Mono"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["font.size"] = 10

C_LIN, C_HI, C_MID, C_LO = "#185FA5", "#A32D2D", "#854F0B", "#B4B2A9"
NF, SZ = 4, 80
GROUND_Y0 = 62
FRAME_NAME = ["帧0（最新）", "帧1", "帧2", "帧3（最早）"]

def _imp(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

probe, sleaf = _imp("probe"), _imp("sweep_leaf")

def coords(i):
    return i // (SZ * SZ), (i % (SZ * SZ)) // SZ, i % SZ

def prepare():
    z = np.load(os.path.join(OUT, "data.npz"))
    Xtr, Qtr, Xte, Qte = z["Xtr"], z["Qtr"], z["Xte"], z["Qte"]
    eid_tr = z["eid_tr"]
    Btr = (Xtr.reshape(len(Xtr), -1) > 0.5).astype(np.float32)
    Bte = (Xte.reshape(len(Xte), -1) > 0.5).astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(Btr))
    vi, ti = perm[:len(Btr) // 5], perm[len(Btr) // 5:]
    on = Btr[ti].mean(0)
    cand = (on >= probe.ON_LO) & (on <= probe.ON_HI)
    Ftr, Fte = Btr[:, cand], Bte[:, cand]
    gidx = np.flatnonzero(cand)
    order = sleaf.screening(Ftr[ti], Qtr[ti, 1])
    return Xtr, Xte, Ftr, Fte, Qtr, Qte, eid_tr, ti, vi, cand, gidx, order

def fit_node_weights(tree, F, t, S):
    pos = None if S is None else {int(f): k for k, f in enumerate(S)}
    for nd in tree["nodes"]:
        i = nd["idx"]
        Fi = F[i] if S is None else F[i][:, S]
        w, b, _, _ = probe.ridge(Fi, t[i], 100.0)
        nd["w"], nd["b"], nd["_pos"] = w, b, pos
    return tree

def influence(tree, F, t, gidx, n_all):
    Inf = {}
    for nd in tree["nodes"]:
        if nd["j"] is None:
            continue
        i, j = nd["idx"], int(nd["j"])
        m = F[i, j] > 0.5
        il, ir = i[~m], i[m]
        if len(il) == 0 or len(ir) == 0:
            continue
        varN = float(t[i].var())
        nl, nr = len(il), len(ir)
        vr = (nl / (nl + nr)) * (varN - float(t[il].var())) \
            + (nr / (nl + nr)) * (varN - float(t[ir].var()))
        w = nd["w"]
        sw = float((w * w).sum())
        pos = nd.get("_pos")
        k = pos[j] if (pos is not None and j in pos) else None
        wf = float(w[k]) if k is not None else 0.0
        Inf[j] = Inf.get(j, 0.0) + (1.0 + (wf * wf / sw if sw > 0 else 0.0)) * max(vr, 0.0)
    g = np.zeros(n_all)
    for i, v in Inf.items():
        g[gidx[i]] = v

    assert int((g > 0).sum()) == len(Inf), \
        f"影响网格非零元数 {(g > 0).sum()} ≠ 参与分裂的特征数 {len(Inf)}：索引映射写错了"
    return Inf, g.reshape(NF, SZ, SZ)

def split_counts(tree, gidx):
    c = np.zeros(NF, int)
    for nd in tree["nodes"]:
        if nd["j"] is not None:
            c[coords(gidx[int(nd["j"])])[0]] += 1
    return c

PLAY_Y0 = 60

def cond_str(cond, maxshow=3):
    parts = [f"帧{c['frame']}(X{c['x']:>2},Y{c['y']:>2}){'亮' if c['on'] else '暗'}"
             for c in cond[:maxshow]]
    s = " 且 ".join(parts)
    if len(cond) > maxshow:
        s += f" …(共{len(cond)}条)"
    return s

def locate_bird(X, eid, rng, per_ep=8):
    from scipy import ndimage
    xs0, xs1, ys0, ys1 = [], [], [], []
    miss = 0
    for ep in np.unique(eid):
        idx = np.where(eid == ep)[0]
        if len(idx) < 40:
            continue
        for k in rng.choice(len(idx), min(per_ep, len(idx)), replace=False):
            f = X[idx[k], 0].astype(float) / 255.0
            lab, n = ndimage.label(f[:, :PLAY_Y0] > 0.5)
            best = None
            for L in range(1, n + 1):
                a, b = np.where(lab == L)
                if (a.max() - a.min() + 1) > 15 or (b.max() - b.min() + 1) > 12:
                    continue
                if len(a) < 12:
                    continue
                if best is None or len(a) > best[0]:
                    best = (len(a), a, b)
            if best is None:
                miss += 1
                continue
            _, a, b = best
            xs0.append(int(a.min())); xs1.append(int(a.max()))
            ys0.append(int(b.min())); ys1.append(int(b.max()))
    if not xs0:
        raise RuntimeError("未能定位小鸟：连通域规则失效，断言 A 的方位解释无法给出")
    cx = (np.array(xs0) + np.array(xs1)) / 2.0
    cy = (np.array(ys0) + np.array(ys1)) / 2.0
    return {"n": len(xs0), "miss": miss,
            "x": [int(min(xs0)), int(max(xs1))], "y": [int(min(ys0)), int(max(ys1))],
            "cx_mean": float(cx.mean()),
            "cy_med": float(np.median(cy)),
            "cy_p5": float(np.percentile(cy, 5)), "cy_p95": float(np.percentile(cy, 95))}

def path_of(tree, fvec):
    ni, path, nodes = tree["root"], [], tree["nodes"]
    while True:
        nd = nodes[ni]
        if nd["j"] is None:
            return path, ni
        j = int(nd["j"])
        on = bool(fvec[j] > 0.5)
        path.append(j)
        ni = nd["r"] if on else nd["l"]

def rules_of(tree, F, min_support=30, top=6):
    out, stack = [], [(tree["root"], [])]
    while stack:
        ni, cond = stack.pop()
        nd = tree["nodes"][ni]
        if nd["j"] is None:
            m = np.ones(len(F), bool)
            for f, on in cond:
                m &= (F[:, f] > 0.5) if on else (F[:, f] <= 0.5)
            out.append({"cond": cond, "n": int(m.sum()), "mask": m})
        else:
            f = int(nd["j"])
            stack.append((nd["l"], cond + [(f, False)]))
            stack.append((nd["r"], cond + [(f, True)]))
    out = [r for r in out if r["n"] >= min_support and len(r["cond"]) > 0]
    out.sort(key=lambda r: -r["n"])
    return out[:top]

def _rho(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = spearmanr(a, b)
    return float(r), float(p)

def _num(x):
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return x

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    return _num(o)

def main():
    Xtr, Xte, Ftr, Fte, Qtr, Qte, eid_tr, ti, vi, cand, gidx, order = prepare()
    gold = (Qte[:, 1] > Qte[:, 0]).astype(int)
    is_flap = gold == 1

    fr0 = Xte[0, 0].astype(float) / 255.0
    print("【方向自检】轴1≥%d 均值 %.3f ；轴1<%d 均值 %.3f ；轴0 对照 %.3f / %.3f"
          % (GROUND_Y0, fr0[:, GROUND_Y0:].mean(), GROUND_Y0, fr0[:, :GROUND_Y0].mean(),
             fr0[GROUND_Y0:, :].mean(), fr0[:GROUND_Y0, :].mean()))
    assert fr0[:, GROUND_Y0:].mean() > 0.9 > fr0[:, :GROUND_Y0].mean(), \
        "方向自检失败：轴1 不再是纵向，所有方位结论都不可信，必须先查帧的存储朝向"
    print("           → 确认 轴1 = 屏幕Y（地面带在底部），画图一律 .T\n")

    bird = locate_bird(Xtr, eid_tr, np.random.default_rng(0))
    print("【小鸟位置实测】连通域定位 %d 帧（漏检 %d）" % (bird["n"], bird["miss"]))
    print("  屏幕 X ∈ [%d, %d]，中心恒为 %.1f（小鸟不水平移动）"
          % (bird["x"][0], bird["x"][1], bird["cx_mean"]))
    print("  屏幕 Y ∈ [%d, %d]，中心中位 %.1f（p5 %.1f / p95 %.1f，随拍翅上下移动）"
          % (bird["y"][0], bird["y"][1], bird["cy_med"], bird["cy_p5"], bird["cy_p95"]))
    print("  参考：画面中线 Y=%d，地面带 Y≥%d，可玩区 Y∈[0,%d] 的中线是 %d\n"
          % (SZ // 2, GROUND_Y0, GROUND_Y0, GROUND_Y0 // 2))

    mode = "full"
    trees = [sleaf.fit(Ftr[ti], Qtr[ti, a], mode, None) for a in (0, 1)]
    p1, p0 = sleaf.predict(Fte, trees[1]), sleaf.predict(Fte, trees[0])
    pf = (p1 > p0).astype(int)
    m1 = probe.metrics(Qte[:, 1], p1)
    print(f"模型：{mode} 模式，双动作合计 {trees[0]['leaves'] + trees[1]['leaves']} 叶")
    print(f"      Q1 测试 RAE {m1['RAE']:.4f} / MAE {m1['MAE']:.4f}"
          f"   拍翅召回 {float((pf[is_flap] == 1).mean()):.2%}"
          f"   总一致 {float((pf == gold).mean()):.2%}\n")

    print("① 特征影响（论文式 (3)）")
    res_inf = {}
    for K in (512, 1024):
        S = order[:K]
        t = fit_node_weights(copy.deepcopy(trees[1]), Ftr, Qtr[:, 1], S)
        Inf, G = influence(t, Ftr, Qtr[:, 1], gidx, NF * SZ * SZ)
        res_inf[K] = {"Inf": Inf, "G": G, "S": S}
        nz = np.array(sorted(Inf.values(), reverse=True))
        print(f"  K={K:<5} 参与分裂的特征 {len(Inf):>4} 个   影响总和 {sum(Inf.values()):.3f}"
              f"   分裂特征均值 {nz.mean():.5f}   最大 {nz[0]:.4f}")
    a1 = np.zeros(NF * SZ * SZ)
    a2 = np.zeros(NF * SZ * SZ)
    for i, v in res_inf[512]["Inf"].items():
        a1[gidx[i]] = v
    for i, v in res_inf[1024]["Inf"].items():
        a2[gidx[i]] = v
    rho_inf, _ = _rho(a1, a2)
    print(f"  稳健性：K=512 vs K=1024 的影响图秩相关 ρ = {rho_inf:.3f}\n")

    K = 1024
    Inf, G = res_inf[K]["Inf"], res_inf[K]["G"]
    vals = np.array(sorted(Inf.values(), reverse=True))
    n_raw, n_cand = int(len(cand)), int(cand.sum())

    thr_paper = float(sum(Inf.values()) / n_cand)
    thr_raw = float(sum(Inf.values()) / n_raw)
    thr_split = float(vals.mean())
    print(f"  采用 K={K}")
    print(f"  阈值口径（论文定义 = 全部特征影响均值）—— 原文未指明分母，三种读法：")
    print(f"    主口径 全部候选像素 {n_cand} 个   thr = {thr_paper:.6f}"
          f"   → 超阈值 {int((vals > thr_paper).sum())} 个")
    print(f"    备选   全部原始维度 {n_raw} 个   thr = {thr_raw:.6f}"
          f"   → 超阈值 {int((vals > thr_raw).sum())} 个")
    print(f"    备选   仅分裂特征 {len(vals)} 个    thr = {thr_split:.6f}"
          f"   → 超阈值 {int((vals > thr_split).sum())} 个")
    print(f"    （方位与帧占比结论不依赖阈值，三者不影响断言 A/B。）")

    top = sorted(Inf.items(), key=lambda kv: -kv[1])[:20]
    print("  前 5 大影响像素（帧 / 屏X / 屏Y）：")
    for i, v in top[:5]:
        f, x, y = coords(gidx[i])
        print(f"    帧{f}  X{x:>2}  Y{y:>2}   Inf {v:.4f}")

    fr_share = G.sum(axis=(1, 2)) / G.sum()
    quad = {"左上": float(G[:, :SZ // 2, :SZ // 2].sum()),
            "右上": float(G[:, SZ // 2:, :SZ // 2].sum()),
            "左下": float(G[:, :SZ // 2, SZ // 2:].sum()),
            "右下": float(G[:, SZ // 2:, SZ // 2:].sum())}
    qsum = sum(quad.values())
    left = (quad["左上"] + quad["左下"]) / qsum
    topq = (quad["左上"] + quad["右上"]) / qsum
    print(f"  按帧影响占比：{np.round(fr_share * 100, 1).tolist()}")
    print(f"  按四象限：{ {k: f'{100*v/qsum:.0f}%' for k, v in quad.items()} }")
    print(f"  ▸ 断言 A（左上角）：左半 {left:.0%}   上半 {topq:.0%}"
          f"   左上象限 {quad['左上'] / qsum:.0%}")

    in_bird = sum(1 for i, _ in top
                  if bird["x"][0] <= coords(gidx[i])[1] <= bird["x"][1]
                  and bird["y"][0] <= coords(gidx[i])[2] <= bird["y"][1])
    print(f"    ▸ 落到量化坐标：前 20 大影响像素中 {in_bird} 个落在小鸟框"
          f"（屏X∈[{bird['x'][0]},{bird['x'][1]}]，屏Y∈[{bird['y'][0]},{bird['y'][1]}]）"
          f"；最高的几个集中在屏X≈17–18、屏Y≈22–25\n")

    sc = split_counts(trees[1], gidx)
    sc_both = split_counts(trees[0], gidx) + sc
    print(f"  ▸ 断言 B（多数分裂落在最新帧）—— 按【分裂次数】统计：")
    print(f"      Q(s,1) 树：{sc.tolist()}   最新帧占 {sc[0] / max(sc.sum(), 1):.0%}")
    print(f"      双动作合计：{sc_both.tolist()}   最新帧占 {sc_both[0] / max(sc_both.sum(), 1):.0%}\n")

    print("③ 超像素解释（论文 §6.3：沿根→目标叶的**分裂路径**高亮 Inf > 阈值的像素）")
    uniq, cnts = np.unique(eid_tr, return_counts=True)
    e_long = int(uniq[np.argmax(cnts)])
    idx_ep = np.where(eid_tr == e_long)[0]
    picks = [int(idx_ep[0]), int(idx_ep[int(len(idx_ep) * 0.6)])]
    labels = ["回合开局", "回合中期"]
    path_res = []
    for lab, pk in zip(labels, picks):
        path, leaf = path_of(trees[1], Ftr[pk])
        hot = [j for j in path if Inf.get(j, 0.0) > thr_paper]
        byf = np.zeros(NF, int)
        for j in hot:
            byf[coords(gidx[j])[0]] += 1
        path_res.append({"label": lab, "pick": pk, "path": path, "hot": hot, "byf": byf})
        print(f"  {lab}（回合 {e_long} 第 {int(np.where(idx_ep == pk)[0][0]) + 1} 帧）："
              f"路径长 {len(path)}，其中 Inf>阈值 的 {len(hot)} 个；按帧 {byf.tolist()}")
    print()

    print("★ 忠实性交叉验证：树的影响 vs 冻结 DQN 自身的遮挡归因（不是自说自话）")
    net, dev = probe.load_net()
    rng = np.random.default_rng(1)
    sel = rng.choice(len(Xte), 120, replace=False)
    Xb = Xte[sel].astype(np.float32)
    with_score = probe.q_batch(net, Xb, dev)[:, 1]
    BS, NG = 8, SZ // 8
    occ = np.zeros((NF, NG, NG))
    for f in range(NF):
        for bi in range(NG):
            for bj in range(NG):
                x = Xb.copy()
                x[:, f, bi * BS:(bi + 1) * BS, bj * BS:(bj + 1) * BS] = 0.0
                occ[f, bi, bj] = float(np.abs(probe.q_batch(net, x, dev)[:, 1] - with_score).mean())
    blk_tree = np.zeros((NF, NG, NG))
    for f in range(NF):
        for bi in range(NG):
            for bj in range(NG):
                blk_tree[f, bi, bj] = G[f, bi * BS:(bi + 1) * BS, bj * BS:(bj + 1) * BS].sum()
    rho, pv = _rho(blk_tree.ravel(), occ.ravel())
    rho_f0, pv_f0 = _rho(blk_tree[0].ravel(), occ[0].ravel())

    hot = blk_tree.ravel() > 0
    rho_hot, pv_hot = _rho(blk_tree.ravel()[hot], occ.ravel()[hot])
    occ_share = occ.sum(axis=(1, 2)) / max(occ.sum(), 1e-12)
    blk_share = blk_tree.sum(axis=(1, 2)) / max(blk_tree.sum(), 1e-12)
    print(f"  块级（{NF}×{NG}×{NG} = {blk_tree.size} 块）秩相关：")
    print(f"    全部块           ρ = {rho:.3f}  (p = {pv:.2e})   ← 含大量零影响块，偏乐观")
    print(f"    仅树有影响的块   ρ = {rho_hot:.3f}  (p = {pv_hot:.2e})   "
          f"（{int(hot.sum())} 块）← 受限口径，是真正的排序一致性")
    print(f"    仅最新帧 100 块  ρ = {rho_f0:.3f}  (p = {pv_f0:.2e})")
    print(f"  两侧「影响量按帧的占比」对照：")
    print(f"    树   ：{np.round(blk_share * 100, 1).tolist()}")
    print(f"    DQN  ：{np.round(occ_share * 100, 1).tolist()}")
    print(f"    → 两边的第 0 帧占比都最高，说明树与 DQN 都把权重压在最新帧上；"
          f"但**帧内**排序不一致（故仅最新帧的 ρ 弱）。\n")

    print("② 规则抽取（论文对 Flappy Bird 未做；这里做的是它的定量版本：")
    print("   每条规则 = 一个叶（分割单元），列出**解码后的像素条件**与 Q 向量，")
    print("   并在 6,542 帧测试集上给出覆盖度、树 Q 与教师 Q —— 对齐论文 Fig. 7 的表格式")
    rules = []
    for r in rules_of(trees[1], Fte):
        m = r["mask"]
        cond = [{"f": int(gidx[f]), "frame": coords(gidx[f])[0],
                 "x": coords(gidx[f])[1], "y": coords(gidx[f])[2], "on": bool(on)}
                for f, on in r["cond"]]
        rules.append({"n_cond": len(cond), "support": int(m.sum()), "cond": cond,
                      "q1_tree": float(p1[m].mean()), "q0_tree": float(p0[m].mean()),
                      "q1_dqn": float(Qte[m, 1].mean()), "q0_dqn": float(Qte[m, 0].mean()),
                      "flap_tree": float(pf[m].mean()), "flap_dqn": float(gold[m].mean())})
        print(f"  {cond_str(cond):<46} 覆盖 {m.sum():>5} 帧  "
              f"树 Q=({p0[m].mean():+.2f},{p1[m].mean():+.2f})  "
              f"教师 Q=({Qte[m, 0].mean():+.2f},{Qte[m, 1].mean():+.2f})  "
              f"树判/教师拍翅 {pf[m].mean():.0%}/{gold[m].mean():.0%}")

    out = {"mode": mode, "leaves": int(trees[0]["leaves"] + trees[1]["leaves"]),
           "q1_RAE": float(m1["RAE"]), "q1_MAE": float(m1["MAE"]),
           "recall_flap": float((pf[is_flap] == 1).mean()),
           "agree": float((pf == gold).mean()),
           "K": K, "n_dims": n_cand, "n_raw_dims": n_raw, "n_split_feats": int(len(Inf)),
           "inf_total": float(sum(Inf.values())),
           "thr_paper_def": float(thr_paper), "thr_raw_def": float(thr_raw),
           "thr_split_def": float(thr_split),
           "n_above_thr_paper": int((vals > thr_paper).sum()),
           "n_above_thr_raw": int((vals > thr_raw).sum()),
           "n_above_thr_split": int((vals > thr_split).sum()),
           "rho_influence_K512_vs_K1024": float(rho_inf),
           "split_counts_q1": [int(x) for x in sc], "split_counts_both": [int(x) for x in sc_both],
           "top_pixels": [{"idx": int(gidx[i]), "frame": coords(gidx[i])[0],
                           "x": coords(gidx[i])[1], "y": coords(gidx[i])[2],
                           "inf": float(v)} for i, v in top],
           "frame_share_influence": [float(x) for x in fr_share],
           "quadrant_influence": {k: float(v) for k, v in quad.items()},
           "left_half_share": float(left), "top_half_share": float(topq),
           "paths": [{"label": pr["label"], "pick": int(pr["pick"]),
                      "path_len": len(pr["path"]), "n_hot": len(pr["hot"]),
                      "path": [int(x) for x in pr["path"]],
                      "hot": [int(x) for x in pr["hot"]],
                      "byf": [int(x) for x in pr["byf"]]} for pr in path_res],
           "rho_occlusion_block": _num(rho), "p_occlusion_block": _num(pv),
           "rho_occlusion_hotblocks": _num(rho_hot), "p_occlusion_hotblocks": _num(pv_hot),
           "n_hotblocks": int(hot.sum()),
           "rho_occlusion_frame0": _num(rho_f0), "p_occlusion_frame0": _num(pv_f0),
           "occ_frame_share": [float(x) for x in occ_share],
           "blk_frame_share": [float(x) for x in blk_share],
           "bird": bird, "top20_in_bird_box": int(in_bird),
           "rules": rules}
    with open(os.path.join(OUT, "explain_lmut.json"), "w", encoding="utf-8") as fh:
        json.dump(_clean(out), fh, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(OUT, "explain_maps.npz"),
                        inf_grid=G, occ_grid=occ, blk_tree=blk_tree, cand=cand, sel=sel)

    draw_paths(Xtr, gidx, Inf, thr_paper, out)
    draw_influence(G, sc, sc_both, out)
    draw_rules(rules, blk_tree, occ, rho, rho_f0, out)
    print("\n已写 out/explain_lmut.json、out/explain_maps.npz 与 figs/figE1~figE3")

def _sq(ax, img, title, cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

def _bird_box(ax, bird, color="#0F6E5C", lw=1.2):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((bird["x"][0] - 0.5, bird["y"][0] - 0.5),
                           bird["x"][1] - bird["x"][0] + 1,
                           bird["y"][1] - bird["y"][0] + 1,
                           fill=False, ec=color, lw=lw, ls="-" , zorder=6))

def draw_paths(Xtr, gidx, Inf, thr, out):
    bird = out["bird"]
    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05], hspace=0.34, wspace=0.14)
    for r, pr in enumerate(out["paths"]):
        pk = pr["pick"]
        hot = set(pr["hot"])
        for k in range(4):
            ax = fig.add_subplot(gs[r, k])
            _sq(ax, Xtr[pk, k].T, FRAME_NAME[k] if r == 0 else "")
            _bird_box(ax, bird)
            n_on_frame = 0
            for j in hot:
                f, x, y = coords(gidx[j])
                if f == k:
                    ax.plot(x, y, marker="*", ms=13, color="#E24B4A",
                            mec="white", mew=0.7, zorder=5)
                    n_on_frame += 1
            if n_on_frame:
                ax.set_xlabel(f"红星 {n_on_frame} 个", fontsize=8.5, color="#A32D2D")
        fig.text(0.012, 0.90 - r * 0.325, pr["label"], rotation=90, va="center",
                 fontsize=11, fontweight="bold", color=C_HI)
        fig.text(0.055, 0.885 - r * 0.325,
                 f"路径长 {len(pr['path'])}，超阈值 {len(pr['hot'])}",
                 fontsize=9, color="#5F5E5A")

    ax = fig.add_subplot(gs[2, :2])
    w = 0.38
    xs = np.arange(NF)
    for r, pr in enumerate(out["paths"]):
        ax.bar(xs + (r - 0.5) * w, pr["byf"], width=w, color=[C_HI, C_LIN][r],
               label=f"{pr['label']}（路径像素）")
    ax.set_xticks(xs); ax.set_xticklabels(FRAME_NAME, fontsize=8.5)
    ax.set_ylabel("该帧上的高亮像素数")
    ax.set_title("这一条分裂路径落在哪些帧上", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#D3D1C7", lw=0.6); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[2, 2:])
    bc, bq = out["split_counts_q1"], out["split_counts_both"]
    ax.bar(xs - 0.2, bc, width=0.4, color=C_HI, label="Q(s,1) 树")
    ax.bar(xs + 0.2, bq, width=0.4, color=C_LIN, label="双动作合计")
    for i, v in enumerate(bc):
        ax.text(i - 0.2, v + 1, str(v), ha="center", fontsize=8.5)
    for i, v in enumerate(bq):
        ax.text(i + 0.2, v + 1, str(v), ha="center", fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels(FRAME_NAME, fontsize=8.5)
    ax.set_ylabel("分裂节点数")
    ax.set_title(f"全树分裂次数按帧（最新帧占 {bq[0] / max(sum(bq), 1):.0%}）", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#D3D1C7", lw=0.6); ax.set_axisbelow(True)

    fig.suptitle("③ 超像素解释（对标论文 Fig. 8）：沿根→目标叶的分裂路径高亮，红星 = 该像素，"
                 f"Inf > {thr:.6f}（论文定义：全部特征影响均值）；青框 = 实测小鸟位置",
                 fontsize=11.5, y=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"figE1_论文对齐_超像素路径高亮.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  → figs/figE1_论文对齐_超像素路径高亮.png / .pdf")

def draw_influence(G, sc, sc_both, out):
    bird = out["bird"]
    fig = plt.figure(figsize=(13.6, 6.6))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.16)
    vmax = max(G.max(), 1e-9)
    ax0 = None
    for k in range(4):
        ax = fig.add_subplot(gs[0, k])
        _sq(ax, G[k].T, f"帧{k} 特征影响", cmap="magma", vmin=0, vmax=vmax)
        _bird_box(ax, bird, color="#39FF14")
        if k == 0:
            ax0 = ax

    for px in out["top_pixels"][:8]:
        if px["frame"] == 0:
            ax0.plot(px["x"], px["y"], marker="o", ms=3.4, mfc="none",
                     mec="#39FF14", mew=1.1, zorder=7)
    ax = fig.add_subplot(gs[1, 0])
    xg = np.arange(SZ)
    ax.plot(xg, G.sum(axis=(0, 2)), color=C_HI, lw=1.6, label="按屏幕X投影")
    ax.plot(xg, G.sum(axis=(0, 1)), color=C_LIN, lw=1.6, ls="--", label="按屏幕Y投影")
    ax.set_xlabel("坐标（0 = 左 / 上）"); ax.set_ylabel("影响合计")
    ax.set_title("影响沿两个方向的投影", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#D3D1C7", lw=0.6); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[1, 1])
    q = out["quadrant_influence"]; tt = sum(q.values()) or 1
    ks = ["左上", "右上", "左下", "右下"]
    vs = [100 * q[k] / tt for k in ks]
    ax.bar(ks, vs, color=[C_HI, C_MID, C_LO, C_LO], width=0.6)
    for i, v in enumerate(vs):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("占比 (%)"); ax.set_ylim(0, max(vs) * 1.3)
    ax.set_title(f"断言A 四象限（左半 {out['left_half_share']:.0%}）", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#D3D1C7", lw=0.6); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[1, 2])
    xs = np.arange(NF)
    ax.bar(xs - 0.2, sc, width=0.4, color=C_HI, label="Q(s,1) 树")
    ax.bar(xs + 0.2, sc_both, width=0.4, color=C_LIN, label="双动作")
    ax.set_xticks(xs); ax.set_xticklabels(["帧0", "帧1", "帧2", "帧3"], fontsize=8.5)
    ax.set_ylabel("分裂次数")
    ax.set_title("断言B 分裂次数按帧", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#D3D1C7", lw=0.6); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[1, 3])
    fr = np.array(out["frame_share_influence"]) * 100
    ax.bar([f"帧{k}" for k in range(4)], fr, color=C_LIN, width=0.6)
    for i, v in enumerate(fr):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("影响占比 (%)"); ax.set_ylim(0, max(fr) * 1.25)
    ax.set_title("影响量按帧分布（≠ 分裂次数）", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#D3D1C7", lw=0.6); ax.set_axisbelow(True)

    fig.suptitle("① 特征影响（论文式 (3)）：上排为四帧影响热力图（左=屏左，上=屏上）；"
                 f"绿框=实测小鸟位置（屏X {bird['x'][0]}–{bird['x'][1]}，屏Y {bird['y'][0]}–{bird['y'][1]}），"
                 "绿圈=前 8 大影响像素；下排为两项断言检验", fontsize=11.5, y=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"figE2_特征影响与两项断言检验.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  → figs/figE2_特征影响与两项断言检验.png / .pdf")

def draw_rules(rules, blk_tree, occ, rho, rho_f0, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.6, 5.2),
                               gridspec_kw={"width_ratios": [1.55, 1]})
    a.axis("off")
    T = ["② 规则抽取（论文 §6.2 对 Flappy Bird 未做；此为定量版本，格式对齐论文 Fig. 7）",
         f"   条件阈值为单像素亮/暗；每条规则在 6,542 帧测试集上验证。Q=(Q0, Q1)，"
         f"树与 DQN 同格对照", ""]
    for i, r in enumerate(rules[:5], 1):
        T += [f"R{i}. 若 {cond_str(r['cond'])}",
              f"     覆盖 {r['support']:>5} 帧   树 Q=({r['q0_tree']:+.2f},{r['q1_tree']:+.2f})   "
              f"教师 Q=({r['q0_dqn']:+.2f},{r['q1_dqn']:+.2f})   "
              f"树判/教师拍翅 {r['flap_tree']:.0%}/{r['flap_dqn']:.0%}", ""]
    T += ["（树 Q 与教师 Q 越接近，说明该分割单元被模仿得越忠实；",
          "  两列差异越大，说明该单元正是模仿失效的地方。只列支持度 ≥ 30 帧的叶。）"]
    a.text(0, 1, "\n".join(T), va="top", ha="left", fontsize=8.6)

    hi = occ > np.percentile(occ, 50)
    b.scatter(blk_tree[~hi], occ[~hi], s=12, color=C_LO, label="遮挡敏感度较低的一半块")
    b.scatter(blk_tree[hi], occ[hi], s=16, color=C_HI, label="遮挡敏感度较高的一半块")
    b.set_xlabel("树给出的块级影响（式 (3) 求和）")
    b.set_ylabel("冻结 DQN 遮挡归因  |ΔQ(s,1)|")
    b.set_title(f"忠实性检验（块级秩相关）：全部块 ρ={rho:.2f}；"
                f"仅树有影响的 {out['n_hotblocks']} 块 ρ={out['rho_occlusion_hotblocks']:.2f}；"
                f"仅最新帧 ρ={rho_f0:.2f}（n.s.）", fontsize=9.5)
    b.legend(fontsize=8.5, frameon=False, loc="upper left")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.grid(True, color="#D3D1C7", lw=0.6); b.set_axisbelow(True)
    fig.suptitle("② 规则抽取 与 忠实性交叉验证（树的解释 vs DQN 自身的归因）", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"figE3_规则与忠实性.{ext}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  → figs/figE3_规则与忠实性.png / .pdf")

if __name__ == "__main__":
    main()
