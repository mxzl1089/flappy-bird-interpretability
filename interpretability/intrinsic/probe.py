import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dqn"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.chdir(ROOT)

from config import Config
from model import QNet
from utils import init_state, next_state

CKPT = os.path.join("runs", "ckpt-1890000.pt")
OUTDIR = os.path.join(HERE, "out")
LAM_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)
DEPTHS = (3, 4, 5, 6, 7, 8)
MIN_CHILD = 20
ON_LO, ON_HI = 0.01, 0.99
N_TRAIN_EP, N_TEST_EP, STRIDE = 10, 4, 4

def new_game_module():
    sys.path.append(Config.GAME_DIR)
    import wrapped_flappy_bird as G
    return G

def seed_env(G, seed):
    random.seed(seed)
    np.random.seed(seed)
    G.PLAYER_INDEX_GEN = itertools.cycle([0, 1, 2, 1])

def load_net():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = QNet(Config.NUM_ACTIONS).to(dev)
    net.load_state_dict(torch.load(os.path.join(ROOT, CKPT), map_location=dev)["policy"])
    net.eval()
    return net, dev

def q_batch(net, states, dev):
    x = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=dev)
    with torch.no_grad():
        return net(x).cpu().numpy()

def rollout(G, act_fn, seed, stride=None):
    seed_env(G, seed)
    env = G.GameState()
    a0 = np.zeros(2); a0[0] = 1
    frame, _, term = env.frame_step(a0, render=False)
    st, score, t = init_state(frame), 0, 0
    X, Q = [], []
    while not term:
        a, q = act_fn(st, t)
        if stride is not None and t % stride == 0:
            X.append(st.copy()); Q.append(q)
        av = np.zeros(2); av[a] = 1
        frame, r, term = env.frame_step(av, render=False)
        if r == 1.0:
            score += 1
        st = next_state(st, frame)
        t += 1
    return score, X, Q

def collect(net, dev, n_ep, seed0):
    G = new_game_module()

    def greedy(st, _t):
        q = q_batch(net, st[None], dev)[0]
        return int(q.argmax()), q

    X, Q, scores, eid = [], [], [], []
    for e in range(n_ep):
        sc, x, q = rollout(G, greedy, seed0 + e, stride=STRIDE)
        X += x; Q += q; scores.append(sc); eid += [e] * len(x)
    return (np.asarray(X, np.uint8), np.asarray(Q, np.float32), scores,
            np.asarray(eid, np.int32))

def play(G, act_fn, n_ep, seed0=9000):
    return [rollout(G, lambda st, t: (int(act_fn(st, t)), None), seed0 + e)[0]
            for e in range(n_ep)]

def ridge(F, t, lam, iters=80, tol=1e-8):
    n, d = F.shape
    mu, tm = F.mean(0), t.mean()
    Fc, tc = F - mu, t - tm
    bb = Fc.T @ tc
    pre = 1.0 / np.maximum((Fc * Fc).sum(0) + lam, 1e-12)
    w = np.zeros(d)
    r = bb.copy(); z = pre * r; p = z.copy(); rz = r @ z
    nb = float(bb @ bb) + 1e-30
    k = 0
    for k in range(iters):
        Ap = Fc.T @ (Fc @ p) + lam * p
        den = float(p @ Ap)
        if den <= 0:
            break
        a = rz / den
        w += a * p; r -= a * Ap
        if float(r @ r) < tol * nb:
            break
        z = pre * r; rz2 = float(r @ z)
        p = z + (rz2 / rz) * p; rz = rz2
    return w, tm - mu @ w, k + 1, float(np.sqrt((r @ r) / nb))

def best_split(F, t, min_child):
    n = len(t)
    if n < 2 * min_child:
        return None, None
    n1 = F.sum(0).astype(np.float64)
    s1 = (F.T @ t).astype(np.float64)
    q1 = (F.T @ (t * t)).astype(np.float64)
    n0 = n - n1
    s0 = float(t.sum()) - s1
    q0 = float((t * t).sum()) - q1
    ok = (n1 >= min_child) & (n0 >= min_child)
    if not ok.any():
        return None, None
    sse = np.where(ok, (q1 - s1 * s1 / np.maximum(n1, 1)) + (q0 - s0 * s0 / np.maximum(n0, 1)),
                   np.inf)
    j = int(np.argmin(sse))
    if not np.isfinite(sse[j]):
        return None, None
    return j, F[:, j] > 0.5

def build(F, t, idx, nodes, depth, max_depth, min_child):
    ni = len(nodes)
    nodes.append({"idx": idx, "j": None, "l": None, "r": None, "w": None, "b": None})
    if depth >= max_depth or len(idx) < 2 * min_child:
        return ni
    j, m = best_split(F[idx], t[idx], min_child)
    if j is None:
        return ni
    nodes[ni]["j"] = j
    nodes[ni]["l"] = build(F, t, idx[~m], nodes, depth + 1, max_depth, min_child)
    nodes[ni]["r"] = build(F, t, idx[m], nodes, depth + 1, max_depth, min_child)
    return ni

def fit_tree(F, t, max_depth, mode, lam):
    nodes = []
    root = build(F, t, np.arange(len(t)), nodes, 0, max_depth, MIN_CHILD)
    resid = []
    for nd in nodes:
        if nd["j"] is not None:
            continue
        i = nd["idx"]
        if mode == "const":
            nd["b"] = float(t[i].mean())
        else:
            nd["w"], nd["b"], _, rr = ridge(F[i], t[i], lam)
            resid.append(rr)
    return {"nodes": nodes, "root": root,
            "leaves": sum(1 for n in nodes if n["j"] is None),
            "cg_resid_max": max(resid) if resid else 0.0}

def predict(F, tree):
    nodes, out = tree["nodes"], np.zeros(len(F))
    stack = [(tree["root"], np.arange(len(F)))]
    while stack:
        ni, idx = stack.pop()
        nd = nodes[ni]
        if nd["j"] is None:
            out[idx] = (F[idx] @ nd["w"] + nd["b"]) if nd["w"] is not None else nd["b"]
        else:
            m = F[idx, nd["j"]] > 0.5
            stack.append((nd["l"], idx[~m]))
            stack.append((nd["r"], idx[m]))
    return out

def metrics(y, p):
    base = y.mean()
    return {"MAE": float(np.abs(p - y).mean()),
            "RMSE": float(np.sqrt(((p - y) ** 2).mean())),
            "RAE": float(np.abs(p - y).sum() / max(np.abs(base - y).sum(), 1e-9)),
            "RSE": float(((p - y) ** 2).sum() / max(((base - y) ** 2).sum(), 1e-9))}

def line(tag, m, tr):
    return (f"  {tag:<26} 叶{m['leaves']:>4}"
            f"  RAE {m['RAE']:.4f}(训练 {tr['RAE']:.4f})"
            f"  MAE {m['MAE']:.4f}(训练 {tr['MAE']:.4f})"
            f"  RMSE {m['RMSE']:.4f}(训练 {tr['RMSE']:.4f})")

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    net, dev = load_net()
    G = new_game_module()
    print(f"设备 {dev}    被模仿模型 {CKPT}（冻结，不训练）")

    CACHE = os.path.join(OUTDIR, "data.npz")
    if os.path.exists(CACHE) and "--nocache" not in sys.argv:
        z = np.load(CACHE)
        Xtr, Qtr, Xte, Qte = z["Xtr"], z["Qtr"], z["Xte"], z["Qte"]
        eid_tr, scr_tr, scr_te = z["eid_tr"], list(map(int, z["scr_tr"])), list(map(int, z["scr_te"]))
        print(f"复用缓存 {CACHE}（跳过采集；要重采加 --nocache）")
    else:
        Xtr, Qtr, scr_tr, eid_tr = collect(net, dev, N_TRAIN_EP, seed0=1000)
        Xte, Qte, scr_te, _eid_te = collect(net, dev, N_TEST_EP, seed0=5000)
        np.savez_compressed(CACHE, Xtr=Xtr, Qtr=Qtr, Xte=Xte, Qte=Qte,
                            eid_tr=eid_tr, scr_tr=np.asarray(scr_tr),
                            scr_te=np.asarray(scr_te))
    print(f"训练 {N_TRAIN_EP} 局 → {len(Xtr)} 条   每局得分 {scr_tr}")
    print(f"测试 {N_TEST_EP} 局 → {len(Xte)} 条   每局得分 {scr_te}"
          f"   （按局分层拆分，无跨集泄漏）")
    print(f"Q 尺度：mean {Qtr.mean():.3f}  std {Qtr.std():.3f}  min {Qtr.min():.3f}"
          f"  max {Qtr.max():.3f}")
    print(f"        Q(s,0) mean {Qtr[:,0].mean():.3f} std {Qtr[:,0].std():.3f}"
          f"   Q(s,1) mean {Qtr[:,1].mean():.3f} std {Qtr[:,1].std():.3f}")
    print(f"        DQN 拍翅率 {float((Qtr[:,1] > Qtr[:,0]).mean()):.2%}"
          f"  ⇒ 「永远不拍翅」的决策一致率就有 {float((Qtr[:,1] <= Qtr[:,0]).mean()):.2%}")

    Btr = (Xtr.reshape(len(Xtr), -1) > 0.5).astype(np.float32)
    Bte = (Xte.reshape(len(Xte), -1) > 0.5).astype(np.float32)

    perm = np.random.default_rng(0).permutation(len(Btr))
    vi, ti = perm[:len(Btr) // 5], perm[len(Btr) // 5:]

    on = Btr[ti].mean(0)
    cand = (on >= ON_LO) & (on <= ON_HI)

    varies_te = Bte.min(0) != Bte.max(0)
    lost = int((varies_te & ~cand).sum())
    print(f"\n候选像素 {int(cand.sum())} / 25600（on-rate ∈ [{ON_LO}, {ON_HI}]，仅在训练子集上筛）")
    print(f"  恒亮 {int((on >= ON_HI).sum())} 个（含地面带 4×1440=5760）"
          f"   恒暗 {int((on <= ON_LO).sum())} 个")
    print(f"  真正被误删（测试集里变化过但被筛掉）：{lost} 个（恒亮/恒暗不计，它们只进截距）")
    Ftr, Fte = Btr[:, cand], Bte[:, cand]

    res = {"ckpt": CKPT, "n_train": len(Xtr), "n_test": len(Xte), "stride": STRIDE,
           "train_scores": list(map(int, scr_tr)), "test_scores": list(map(int, scr_te)),
           "q_stats": {"mean": float(Qtr.mean()), "std": float(Qtr.std()),
                       "q0_mean": float(Qtr[:, 0].mean()), "q0_std": float(Qtr[:, 0].std()),
                       "q1_mean": float(Qtr[:, 1].mean()), "q1_std": float(Qtr[:, 1].std()),
                       "flap_rate_train": float((Qtr[:, 1] > Qtr[:, 0]).mean())},
           "n_cand": int(cand.sum()), "n_filtered_but_active_in_test": lost,
           "per_action": {}, "combined": {}}
    models, preds, tr_preds, lams, resid = {}, {}, {}, {}, []

    best_lam, best_val = LAM_GRID[0], np.inf
    for lam in LAM_GRID:
        w, b, _, _ = ridge(Ftr[ti], Qtr[ti, 1], lam)
        e = float(np.abs(Ftr[vi] @ w + b - Qtr[vi, 1]).mean())
        if e < best_val:
            best_lam, best_val = lam, e
    print(f"\nλ 由训练集内部验证集选出：λ = {best_lam:g}（验证 MAE {best_val:.4f}）")
    for lam in LAM_GRID:
        w, b, it, rr = ridge(Ftr[ti], Qtr[ti, 1], lam)
        print(f"    λ={lam:<9g} 验证 MAE {float(np.abs(Ftr[vi] @ w + b - Qtr[vi, 1]).mean()):.4f}"
              f"    PCG 迭代 {it}  相对残差 {rr:.2e}")

    for a in (0, 1):
        t_tr, t_te = Qtr[:, a], Qte[:, a]
        print(f"\n{'-' * 96}")
        print(f"动作 {a}：模仿 Q(s,{a})    训练集 mean {t_tr.mean():.3f} / std {t_tr.std():.3f}"
              f"    （RAE = 1.0 就是「永远输出均值」这条线）")
        print("-" * 96)
        pa = {"mean_baseline": metrics(t_te, np.full(len(t_te), t_tr.mean()))}
        pa["mean_baseline"]["leaves"] = 0
        pa["mean_baseline"]["train_RAE"] = 1.0
        print(f"  {'常数基线（永远输出均值）':<26} 叶   0  RAE 1.0000")

        w0, b0, it0, rr0 = ridge(Ftr[ti], t_tr[ti], best_lam)
        resid.append(rr0)
        pa["ridge_1leaf"] = {**metrics(t_te, Fte @ w0 + b0),
                             "leaves": 1, "lam": best_lam,
                             "train_RAE": metrics(t_tr[ti], Ftr[ti] @ w0 + b0)["RAE"],
                             "train_MAE": metrics(t_tr[ti], Ftr[ti] @ w0 + b0)["MAE"],
                             "train_RMSE": metrics(t_tr[ti], Ftr[ti] @ w0 + b0)["RMSE"],
                             "pcg_iters": it0, "pcg_resid": rr0}
        print(line("单叶线性模型", pa["ridge_1leaf"], {"RAE": pa["ridge_1leaf"]["train_RAE"],
                                                  "MAE": pa["ridge_1leaf"]["train_MAE"],
                                                  "RMSE": pa["ridge_1leaf"]["train_RMSE"]}))

        for mode in ("const", "linear"):
            cname = "常数叶 CART" if mode == "const" else "线性叶模型树"
            for d in DEPTHS:
                tree = fit_tree(Ftr[ti], t_tr[ti], d, mode, best_lam)
                p_te, p_tr = predict(Fte, tree), predict(Ftr[ti], tree)
                m = {**metrics(t_te, p_te), "leaves": tree["leaves"],
                     "train_RAE": metrics(t_tr[ti], p_tr)["RAE"],
                     "train_MAE": metrics(t_tr[ti], p_tr)["MAE"],
                     "train_RMSE": metrics(t_tr[ti], p_tr)["RMSE"],
                     "cg_resid_max": tree["cg_resid_max"]}
                models[(a, mode, d)] = tree
                preds[(a, mode, d)], tr_preds[(a, mode, d)] = p_te, p_tr
                pa[f"{mode}_depth{d}"] = m
                print(line(f"{cname} depth={d}", m,
                           {"RAE": m["train_RAE"], "MAE": m["train_MAE"], "RMSE": m["train_RMSE"]}))
        res["per_action"][f"Q{a}"] = pa

    print(f"\n{'-' * 96}")
    print("双动作合并 —— 决策一致率按类别拆开（总一致率会被类别不平衡骗）")
    print("-" * 96)
    gold = (Qte[:, 1] > Qte[:, 0]).astype(int)
    is_flap = gold == 1
    n_flap = int(is_flap.sum())
    print(f"  测试集 {len(gold)} 帧，其中 DQN 拍翅 {n_flap} 帧（{n_flap/len(gold):.2%}）")
    print(f"  平凡上游基线：永远不拍翅 总一致 {(1-is_flap.mean()):.2%} / 拍翅召回 0.00%"
          f"    永远拍翅 总一致 {is_flap.mean():.2%} / 拍翅召回 100.00%（但几乎必死）")
    for mode in ("const", "linear"):
        for d in DEPTHS:
            p0, p1 = preds[(0, mode, d)], preds[(1, mode, d)]
            m0, m1 = metrics(Qte[:, 0], p0), metrics(Qte[:, 1], p1)
            pred_flap = (p1 > p0).astype(int)
            m = {"leaves": models[(0, mode, d)]["leaves"] + models[(1, mode, d)]["leaves"],
                 "RAE": (m0["RAE"] + m1["RAE"]) / 2,
                 "MAE": (m0["MAE"] + m1["MAE"]) / 2,
                 "RMSE": (m0["RMSE"] + m1["RMSE"]) / 2,
                 "RSE": (m0["RSE"] + m1["RSE"]) / 2,
                 "agree": float((pred_flap == gold).mean()),
                 "recall_flap": float((pred_flap[is_flap] == 1).mean()),
                 "spec_noflap": float((pred_flap[~is_flap] == 0).mean())}
            res["combined"][f"{mode}_depth{d}"] = m
            cname = "常数叶 CART" if mode == "const" else "线性叶模型树"
            print(f"  {cname} depth={d}  叶{m['leaves']:>4}  RAE {m['RAE']:.4f}"
                  f"  MAE {m['MAE']:.4f}  总一致 {m['agree']:.2%}"
                  f"  拍翅召回 {m['recall_flap']:.2%}  不拍翅特异度 {m['spec_noflap']:.2%}")

    print(f"\n{'-' * 96}")
    print("数据量扫描（depth=8 线性叶，模仿 Q(s,1)）—— 测试 RAE 还在降 = 数据不够；平了 = 表示/维度不够")
    print("-" * 96)
    scale = []
    for k in (1, 2, 3, 5, 7, 10):
        if k > N_TRAIN_EP:
            break
        sub = ti[eid_tr[ti] < k]
        if len(sub) < 200:
            continue
        trk = fit_tree(Ftr[sub], Qtr[sub, 1], 8, "linear", best_lam)
        mm = metrics(Qte[:, 1], predict(Fte, trk))
        scale.append({"episodes": k, "n": int(len(sub)), "leaves": trk["leaves"],
                      "test_RAE": mm["RAE"], "test_MAE": mm["MAE"]})
        print(f"  {k:>2} 局（{len(sub):>5} 条）  叶{trk['leaves']:>3}"
              f"   测试 RAE {mm['RAE']:.4f}   MAE {mm['MAE']:.4f}")
    res["data_scaling"] = scale

    nl, dl = min(((res["combined"][f"linear_depth{d}"]["leaves"], d) for d in DEPTHS),
                 key=lambda z: abs(z[0] - 100))
    nc, dc = min(((res["combined"][f"const_depth{d}"]["leaves"], d) for d in DEPTHS),
                 key=lambda z: abs(z[0] - 100))
    rl = res["combined"][f"linear_depth{dl}"]["RAE"]
    rc = res["combined"][f"const_depth{dc}"]["RAE"]
    best_d = min(DEPTHS, key=lambda d: res["combined"][f"linear_depth{d}"]["RAE"])
    best = res["combined"][f"linear_depth{best_d}"]
    A = rl <= 0.80 * rc
    B = best["RAE"] <= 0.10 and best["recall_flap"] >= 0.50
    B2 = best["RAE"] <= 0.25

    print(f"\n{'=' * 96}")
    print("判据（跑之前写死，现在才看）")
    print("=" * 96)
    print(f"  A 方法前提：≈100 叶时  线性叶 RAE {rl:.4f}（{nl} 叶）  vs  "
          f"常数叶 RAE {rc:.4f}（{nc} 叶）   →  {'★成立' if A else '不成立'}"
          f"（要求线性叶低 ≥20%）")
    print(f"  B 可用性  ：最佳线性叶 depth={best_d}、{best['leaves']} 叶，RAE {best['RAE']:.4f}，"
          f"拍翅召回 {best['recall_flap']:.2%}   →  "
          f"{'★优秀，树吃得下这个输入' if B else ('勉强，需要先降维再上树' if B2 else '不合格，方向要换')}")

    m1 = res["per_action"]["Q1"]
    b1 = min(DEPTHS, key=lambda d: m1[f"linear_depth{d}"]["RAE"])
    c1 = min(DEPTHS, key=lambda d: m1[f"const_depth{d}"]["RAE"])
    r1, rc1 = m1[f"linear_depth{b1}"]["RAE"], m1[f"const_depth{c1}"]["RAE"]
    print(f"  ·补充口径：只看有效动作 Q(s,1)（目标方差大，RAE 才不会被放大）")
    print(f"     线性叶 depth={b1} RAE {r1:.4f}（{m1[f'linear_depth{b1}']['leaves']} 叶）"
          f"  vs  常数叶 depth={c1} RAE {rc1:.4f}（{m1[f'const_depth{c1}']['leaves']} 叶）"
          f"  → 线性叶低 {100 * (1 - r1 / rc1):.0f}%")
    supp = {"rae_linear": r1, "rae_const": rc1, "depth_linear": b1, "depth_const": c1}

    print(f"\n{'=' * 96}")
    print("端到端（判据 C）：让模仿器自己玩")
    print("=" * 96)
    dqn = play(G, lambda st, _t: int(q_batch(net, st[None], dev)[0].argmax()), 10)
    print(f"  DQN 本人            10 局   平均 {np.mean(dqn):.1f}   中位 {np.median(dqn):.1f}")
    ms = {}
    for mode, tag in (("const", "常数叶"), ("linear", "线性叶")):
        d = best_d if mode == "linear" else dc
        trees = [models[(a, mode, d)] for a in (0, 1)]

        def pol(st, _t, trees=trees):
            f = (st.reshape(1, -1) > 0.5).astype(np.float32)[:, cand]
            return int(np.argmax([predict(f, tr)[0] for tr in trees]))

        s = play(G, pol, 10)
        ms[mode] = {"scores": list(map(int, s)), "mean": float(np.mean(s)),
                    "median": float(np.median(s)), "depth": d,
                    "leaves": res["combined"][f"{mode}_depth{d}"]["leaves"]}
        print(f"  {tag}模型树 depth={d}  10 局   平均 {np.mean(s):.1f}   "
              f"中位 {np.median(s):.1f}   叶子 {ms[mode]['leaves']}")
    C = ms["linear"]["mean"] >= 0.60 * np.mean(dqn)
    print(f"  C 端到端：模仿器 {ms['linear']['mean']:.1f} / DQN {np.mean(dqn):.1f} = "
          f"{ms['linear']['mean'] / max(np.mean(dqn), 1e-9):.1%}   →  "
          f"{'★成立' if C else '不成立'}（要求 ≥60%）")

    print(f"\n  论文对照（Table 4/5）。他们的 Q 尺度与我们不同，MAE 不可直接比：")
    print(f"    Experience Training  LMUT  MAE 0.019 / RMSE 0.043 / 579 叶 / ARPE  97.62")
    print(f"    Active Play          LMUT  MAE 0.024 / RMSE 0.050 / 229 叶 / ARPE 103.32")
    print(f"    他们的 DQN baseline                                   ARPE 123.42")

    res["criteria"] = {"A_premise": bool(A),
                       "A_pair": {"rae_linear": rl, "leaves_linear": nl,
                                  "rae_const": rc, "leaves_const": nc},
                       "A_on_Q1": supp,
                       "B_usable": bool(B), "B_marginal": bool(B2),
                       "best_linear": {"depth": best_d, **best},
                       "C_end2end": bool(C), "dqn_scores": list(map(int, dqn)),
                       "dqn_mean": float(np.mean(dqn)), "mimic": ms,
                       "lam": best_lam, "pcg_resid_1leaf": resid,
                       "elapsed_sec": round(time.time() - t0, 1)}
    with open(os.path.join(OUTDIR, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n用时 {res['criteria']['elapsed_sec']} 秒    已写 interpretability/intrinsic/out/probe.json")

if __name__ == "__main__":
    main()
