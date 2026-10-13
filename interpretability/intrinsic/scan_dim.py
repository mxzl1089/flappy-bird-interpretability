import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("probe", os.path.join(HERE, "probe.py"))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

OUT = os.path.join(HERE, "out")
DEPTH, LAM = 8, 100.0
FACTORS = (1, 2, 4, 8)

def pool(X, f):
    n = len(X)
    if f == 1:
        return (X.reshape(n, -1) > 0.5).astype(np.float32)
    a = X.reshape(n, 4, 80, 80).astype(np.float32)
    a = a.reshape(n, 4, 80 // f, f, 80 // f, f).mean(axis=(3, 5))
    return (a.reshape(n, -1) > 0.5).astype(np.float32)

def main():
    z = np.load(os.path.join(OUT, "data.npz"))
    Xtr, Qtr, Xte, Qte = z["Xtr"], z["Qtr"], z["Xte"], z["Qte"]
    print(f"复用缓存：训练 {len(Xtr)} 条 / 测试 {len(Xte)} 条   被模仿 Q(s,1)  mean {Qtr[:,1].mean():.3f}"
          f"  std {Qtr[:,1].std():.3f}")

    rows = []
    print(f"\n{'f':>3} {'维度 d':>8} {'候选':>7} {'叶':>5} {'每叶样本':>8} {'样本/d':>7}"
          f" {'测试RAE':>8} {'训练RAE':>8} {'拍翅召回':>8}")
    print("-" * 78)
    for f in FACTORS:
        Btr, Bte = pool(Xtr, f), pool(Xte, f)
        d = Btr.shape[1]
        perm = np.random.default_rng(0).permutation(len(Btr))
        vi, ti = perm[:len(Btr) // 5], perm[len(Btr) // 5:]
        on = Btr[ti].mean(0)
        cand = (on >= probe.ON_LO) & (on <= probe.ON_HI)
        Ftr, Fte = Btr[:, cand], Bte[:, cand]

        tre = probe.fit_tree(Ftr[ti], Qtr[ti, 1], DEPTH, "linear", LAM)
        p_te, p_tr = probe.predict(Fte, tre), probe.predict(Ftr[ti], tre)
        m = probe.metrics(Qte[:, 1], p_te)
        mt = probe.metrics(Qtr[ti, 1], p_tr)
        nleaf = max(tre["leaves"], 1)
        per_leaf = len(ti) / nleaf
        rows.append({"f": f, "d": int(cand.sum()), "leaves": nleaf,
                     "samples_per_leaf": float(per_leaf), "samples_per_dim": float(len(ti) / cand.sum()),
                     "test_RAE": m["RAE"], "test_MAE": m["MAE"],
                     "train_RAE": mt["RAE"], "train_MAE": mt["MAE"],
                     "cg_resid_max": tre["cg_resid_max"]})
        print(f"{f:>3} {int(cand.sum()):>8} {int(cand.sum()):>7} {nleaf:>5} {per_leaf:>8.0f}"
              f" {len(ti)/cand.sum():>7.3f} {m['RAE']:>8.4f} {mt['RAE']:>8.4f}"
              f" {'—':>8}")

    print("\n判读：看『每叶样本 / 维度』这个比值 —— ≥1 才算线性叶良定。")
    for r in rows:
        ratio = r["samples_per_leaf"] / r["d"]
        flag = "良定" if ratio >= 1 else ("勉强" if ratio >= 0.2 else "严重欠定")
        print(f"  f={r['f']}（d={r['d']}）  每叶 {r['samples_per_leaf']:.0f} 条 / {r['d']} 维"
              f" = {ratio:.3f}  {flag}   测试 RAE {r['test_RAE']:.4f}")

    with open(os.path.join(OUT, "scan_dim.json"), "w", encoding="utf-8") as fh:
        json.dump({"depth": DEPTH, "lam": LAM, "target": "Q(s,1)", "rows": rows},
                  fh, ensure_ascii=False, indent=1)
    print(f"\n已写 {os.path.join(OUT, 'scan_dim.json')}")

if __name__ == "__main__":
    main()
