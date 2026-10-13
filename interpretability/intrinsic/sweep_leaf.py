import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("probe", os.path.join(HERE, "probe.py"))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

OUT = os.path.join(HERE, "out")
LAM = 100.0
DEPTH = 10
MIN_CHILD = 8
TOPK = (32, 64, 128, 256, 512, 1024)

def build_path(F, t, idx, nodes, path, depth, max_depth, min_child):
    ni = len(nodes)
    nodes.append({"idx": idx, "j": None, "l": None, "r": None, "path": tuple(path)})
    if depth >= max_depth or len(idx) < 2 * min_child:
        return ni
    j, m = probe.best_split(F[idx], t[idx], min_child)
    if j is None:
        return ni
    nodes[ni]["j"] = j
    nodes[ni]["l"] = build_path(F, t, idx[~m], nodes, path + [j], depth + 1, max_depth, min_child)
    nodes[ni]["r"] = build_path(F, t, idx[m], nodes, path + [j], depth + 1, max_depth, min_child)
    return ni

def fit(F, t, mode, feat_global=None):
    nodes = []
    root = build_path(F, t, np.arange(len(t)), nodes, [], 0, DEPTH, MIN_CHILD)
    for nd in nodes:
        if nd["j"] is not None:
            continue
        i = nd["idx"]
        if mode == "full":
            feat = None
        elif mode == "topk":
            feat = feat_global
        else:
            feat = np.asarray(sorted(set(nd["path"])), dtype=np.int64)
        nd["feat"] = feat
        if feat is not None and len(feat) == 0:
            nd["feat"], nd["w"], nd["b"] = None, None, float(t[i].mean())
            continue
        Fi = F[i] if feat is None else F[i][:, feat]
        nd["w"], nd["b"], _, _ = probe.ridge(Fi, t[i], LAM)
    return {"nodes": nodes, "root": root,
            "leaves": sum(1 for n in nodes if n["j"] is None),
            "feat_used": sorted({int(f) for n in nodes if n["j"] is None and n["feat"] is not None
                                 for f in n["feat"]})}

def predict(F, tree):
    nodes, out = tree["nodes"], np.zeros(len(F))
    stack = [(tree["root"], np.arange(len(F)))]
    while stack:
        ni, idx = stack.pop()
        nd = nodes[ni]
        if nd["j"] is None:
            if nd["w"] is None:
                out[idx] = nd["b"]
            else:
                Fi = F[idx] if nd["feat"] is None else F[idx][:, nd["feat"]]
                out[idx] = Fi @ nd["w"] + nd["b"]
        else:
            m = F[idx, nd["j"]] > 0.5
            stack.append((nd["l"], idx[~m]))
            stack.append((nd["r"], idx[m]))
    return out

def screening(F, t):
    n = len(t)
    n1 = F.sum(0).astype(np.float64)
    s1 = (F.T @ t).astype(np.float64)
    ok = (n1 >= MIN_CHILD) & ((n - n1) >= MIN_CHILD)
    gain = np.where(ok, s1 * s1 / np.maximum(n1, 1) + (t.sum() - s1) ** 2 / np.maximum(n - n1, 1),
                    -np.inf)
    return np.argsort(-gain)

def main():
    z = np.load(os.path.join(OUT, "data.npz"))
    Xtr, Qtr, Xte, Qte = z["Xtr"], z["Qtr"], z["Xte"], z["Qte"]
    Btr = (Xtr.reshape(len(Xtr), -1) > 0.5).astype(np.float32)
    Bte = (Xte.reshape(len(Xte), -1) > 0.5).astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(Btr))
    vi, ti = perm[:len(Btr) // 5], perm[len(Btr) // 5:]
    on = Btr[ti].mean(0)
    cand = (on >= probe.ON_LO) & (on <= probe.ON_HI)
    Ftr, Fte = Btr[:, cand], Bte[:, cand]
    gold = (Qte[:, 1] > Qte[:, 0]).astype(int)
    is_flap = gold == 1
    print(f"训练 {len(ti)} / 测试 {len(Fte)}   候选维度 {Ftr.shape[1]}   depth={DEPTH} min_child={MIN_CHILD}")
    print(f"基线病根：叶子 ~300 时每叶约 {len(ti)//300} 个样本，却要拟合 {Ftr.shape[1]} 维线性模型\n")

    order = screening(Ftr[ti], Qtr[ti, 1])
    rows, models = [], {}

    def run(tag, mode, feat_global=None):
        trees = [fit(Ftr[ti], Qtr[ti, a], mode, feat_global) for a in (0, 1)]
        p = [predict(Fte, t) for t in trees]
        m0, m1 = probe.metrics(Qte[:, 0], p[0]), probe.metrics(Qte[:, 1], p[1])
        pf = (p[1] > p[0]).astype(int)
        nfeat = (Ftr.shape[1] if feat_global is None and mode == "full"
                 else (len(trees[0]["feat_used"]) if mode == "path" else len(feat_global)))
        row = {"tag": tag, "mode": mode, "n_feat": int(nfeat),
               "leaves": trees[0]["leaves"] + trees[1]["leaves"],
               "q1_RAE": m1["RAE"], "q1_MAE": m1["MAE"],
               "comb_RAE": (m0["RAE"] + m1["RAE"]) / 2,
               "agree": float((pf == gold).mean()),
               "recall_flap": float((pf[is_flap] == 1).mean())}
        rows.append(row)
        models[tag] = trees
        print(f"  {tag:<16} 特征 {row['n_feat']:>6}  叶{row['leaves']:>4}"
              f"  Q1_RAE {m1['RAE']:.4f}  MAE {m1['MAE']:.4f}"
              f"  合并RAE {row['comb_RAE']:.4f}  拍翅召回 {row['recall_flap']:.2%}")
        return row

    print(f"{'配置':<18}{'特征数':>8}{'叶':>6}{'Q1_RAE':>10}{'Q1_MAE':>10}{'合并RAE':>10}{'拍翅召回':>10}")
    print("-" * 76)
    run("full(基线)", "full")
    for k in TOPK:
        run(f"topk K={k}", "topk", order[:k])
    run("path(路径像素)", "path")

    best = min([r for r in rows if r["mode"] != "full"], key=lambda r: r["q1_RAE"])
    base = rows[0]
    print(f"\n最佳：{best['tag']}  特征 {best['n_feat']}（基线的 {best['n_feat']/base['n_feat']:.1%}）"
          f"  Q1_RAE {best['q1_RAE']:.4f}（基线 {base['q1_RAE']:.4f}，降 {100*(1-best['q1_RAE']/base['q1_RAE']):.0f}%）"
          f"  拍翅召回 {best['recall_flap']:.2%}")

    print("\n端到端（10 局）：")
    net, dev = probe.load_net()
    G = probe.new_game_module()
    dqn = probe.play(G, lambda st, _t: int(probe.q_batch(net, st[None], dev)[0].argmax()), 10)
    print(f"  DQN          平均 {np.mean(dqn):.1f}")
    e2e = {}
    for tag in (base["tag"], best["tag"]):
        trees = models[tag]

        def pol(st, _t, trees=trees):
            f = (st.reshape(1, -1) > 0.5).astype(np.float32)[:, cand]
            return int(np.argmax([predict(f, t)[0] for t in trees]))

        s = probe.play(G, pol, 10)
        e2e[tag] = {"scores": list(map(int, s)), "mean": float(np.mean(s)),
                    "ratio": float(np.mean(s) / max(np.mean(dqn), 1e-9))}
        print(f"  {tag:<14} 平均 {np.mean(s):.1f}  = DQN 的 "
              f"{np.mean(s)/max(np.mean(dqn),1e-9):.1%}")

    with open(os.path.join(OUT, "sweep_leaf.json"), "w", encoding="utf-8") as fh:
        json.dump({"depth": DEPTH, "min_child": MIN_CHILD, "lam": LAM, "rows": rows,
                   "best": best, "dqn_mean": float(np.mean(dqn)), "end2end": e2e,
                   "top_pixels_q1": [int(i) for i in order[:512]]}, fh, ensure_ascii=False, indent=1)
    print(f"\n已写 {os.path.join(OUT, 'sweep_leaf.json')}")

if __name__ == "__main__":
    main()
