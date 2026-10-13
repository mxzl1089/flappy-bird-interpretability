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
DEPTHS = (8, 9, 10)
MIN_CHILDS = (20, 8)

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
    print(f"训练 {len(ti)} 条 / 测试 {len(Fte)} 条   候选维度 {int(cand.sum())}"
          f"   测试集拍翅帧 {int(is_flap.sum())}（{is_flap.mean():.2%}）")

    rows, models = [], {}
    print(f"\n{'模式':<8}{'depth':>6}{'min_child':>10}{'叶':>6}{'Q1_RAE':>9}{'Q1_训练RAE':>11}"
          f"{'合并RAE':>9}{'总一致':>8}{'拍翅召回':>9}")
    print("-" * 84)
    for mode in ("const", "linear"):
        for mc in MIN_CHILDS:
            probe.MIN_CHILD = mc
            for d in DEPTHS:
                trees = [probe.fit_tree(Ftr[ti], Qtr[ti, a], d, mode, LAM) for a in (0, 1)]
                p = [probe.predict(Fte, t) for t in trees]
                ptr = [probe.predict(Ftr[ti], t) for t in trees]
                m1 = probe.metrics(Qte[:, 1], p[1])
                mt1 = probe.metrics(Qtr[ti, 1], ptr[1])
                m0 = probe.metrics(Qte[:, 0], p[0])
                pf = (p[1] > p[0]).astype(int)
                row = {"mode": mode, "depth": d, "min_child": mc,
                       "leaves": trees[0]["leaves"] + trees[1]["leaves"],
                       "leaves_q1": trees[1]["leaves"],
                       "q1_RAE": m1["RAE"], "q1_train_RAE": mt1["RAE"], "q1_MAE": m1["MAE"],
                       "comb_RAE": (m0["RAE"] + m1["RAE"]) / 2,
                       "agree": float((pf == gold).mean()),
                       "recall_flap": float((pf[is_flap] == 1).mean()),
                       "spec_noflap": float((pf[~is_flap] == 0).mean())}
                rows.append(row)
                models[(mode, mc, d)] = trees
                print(f"{mode:<8}{d:>6}{mc:>10}{row['leaves']:>6}{m1['RAE']:>9.4f}"
                      f"{mt1['RAE']:>11.4f}{row['comb_RAE']:>9.4f}{row['agree']:>8.2%}"
                      f"{row['recall_flap']:>9.2%}")

    lin = [r for r in rows if r["mode"] == "linear"]
    best = min(lin, key=lambda r: r["q1_RAE"])
    print(f"\n最佳线性叶：depth={best['depth']} min_child={best['min_child']} "
          f"{best['leaves']} 叶  Q1 RAE {best['q1_RAE']:.4f}（训练 {best['q1_train_RAE']:.4f}）"
          f"  拍翅召回 {best['recall_flap']:.2%}")

    print("\n端到端（10 局；种子与第一轮探针同源，可直接比）：")
    net, dev = probe.load_net()
    G = probe.new_game_module()
    dqn = probe.play(G, lambda st, _t: int(probe.q_batch(net, st[None], dev)[0].argmax()), 10)
    print(f"  DQN                 平均 {np.mean(dqn):.1f}  中位 {np.median(dqn):.1f}")
    e2e = {}
    for mode in ("const", "linear"):
        cands = [r for r in rows if r["mode"] == mode]
        b = (best if mode == "linear" else
             min(cands, key=lambda r: abs(r["leaves"] - best["leaves"])))
        trees = models[(mode, b["min_child"], b["depth"])]

        def pol(st, _t, trees=trees):
            f = (st.reshape(1, -1) > 0.5).astype(np.float32)[:, cand]
            return int(np.argmax([probe.predict(f, t)[0] for t in trees]))

        s = probe.play(G, pol, 10)
        e2e[mode] = {"scores": list(map(int, s)), "mean": float(np.mean(s)),
                     "median": float(np.median(s)), "leaves": b["leaves"],
                     "depth": b["depth"], "min_child": b["min_child"]}
        print(f"  {mode:<8}模型树 {b['leaves']} 叶  平均 {np.mean(s):.1f}  "
              f"中位 {np.median(s):.1f}  = DQN 的 {np.mean(s)/max(np.mean(dqn),1e-9):.1%}")

    with open(os.path.join(OUT, "sweep_capacity.json"), "w", encoding="utf-8") as fh:
        json.dump({"lam": LAM, "rows": rows, "best_linear": best,
                   "dqn_scores": list(map(int, dqn)), "dqn_mean": float(np.mean(dqn)),
                   "end2end": e2e}, fh, ensure_ascii=False, indent=1)
    print(f"\n已写 {os.path.join(OUT, 'sweep_capacity.json')}")

if __name__ == "__main__":
    main()
