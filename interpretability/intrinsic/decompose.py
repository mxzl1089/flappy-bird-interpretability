import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dqn"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.chdir(ROOT)

import probe as P
from model import QNet

OUT = os.path.join(HERE, "out")
SEED, EPOCHS, BS, LR = 0, 300, 64, 1e-3
DEPTH, MINCHILD, LAM = 8, P.MIN_CHILD, 100.0
C_GRID = (1.0, 5.0, 20.0)
QTS = (5, 25, 50, 75, 95)

def rae(y, p):
    base = y.mean()
    return float(np.abs(p - y).sum() / max(np.abs(base - y).sum(), 1e-9))

def qt(y):
    return {f"p{k}": round(float(np.percentile(y, k)), 4) for k in QTS}

def evaluate(P1, P0, Qte, gt, m_te):
    pm = P1 - P0
    pred = (pm > 0).astype(int)
    e = pm - m_te
    ef, mm = e[gt == 1], m_te[gt == 1]
    fix = {}
    for d in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        fix[f"{d:.1f}"] = round(float((ef + d > -mm).mean()), 4)
    fix["debiased"] = round(float((ef - np.median(ef) > -mm).mean()), 4)
    return {
        "recall_flap": round(float((pred[gt == 1] == 1).mean()), 4),
        "spec_noflap": round(float((pred[gt == 0] == 0).mean()), 4),
        "agree": round(float((pred == gt).mean()), 4),
        "RAE_Q1": round(rae(Qte[:, 1], P1), 4),
        "MAE_Q1": round(float(np.abs(P1 - Qte[:, 1]).mean()), 4),
        "RAE_Q0": round(rae(Qte[:, 0], P0), 4),
        "MAE_Q0": round(float(np.abs(P0 - Qte[:, 0]).mean()), 4),
        "RAE_Q1_flap": round(rae(Qte[gt == 1, 1], P1[gt == 1]), 4),
        "RAE_Q1_noflap": round(rae(Qte[gt == 0, 1], P1[gt == 0]), 4),
        "err_signed_flap": qt(ef),
        "err_abs_flap_mean": round(float(np.abs(ef).mean()), 4),
        "frac_neg_err_flap": round(float((ef < 0).mean()), 4),
        "recall_if_debiased": fix,
    }

def train_cnn(x_tr, y_tr, w_tr, x_va, y_va, gva, dev, tag):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = QNet(2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best_rae, best_rec, st_rae, st_rec = np.inf, -1.0, None, None
    yva_np = y_va.numpy()
    for ep in range(EPOCHS):
        net.train()
        idx = torch.randperm(len(x_tr))
        for b in range(0, len(idx), BS):
            j = idx[b:b + BS]
            xb, yb, wb = x_tr[j].to(dev), y_tr[j].to(dev), w_tr[j].to(dev)
            opt.zero_grad()
            loss = ((net(xb) - yb) ** 2 * wb[:, None]).mean()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vp = net(x_va.to(dev)).cpu().numpy()
        r = rae(yva_np[:, 1], vp[:, 1])
        pr = (vp[:, 1] > vp[:, 0]).astype(int)
        rec = float((pr[gva == 1] == 1).mean())
        if r < best_rae:
            best_rae = r
            st_rae = {k: v.clone() for k, v in net.state_dict().items()}
        if rec > best_rec:
            best_rec = rec
            st_rec = {k: v.clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 100 == 0:
            print(f"    [{tag}] epoch {ep+1:>3}  验证 RAE(Q1) {r:.4f}  验证召回 {rec:.4f}")
    return st_rae, st_rec, best_rae, best_rec

def main():
    t0 = time.time()
    res = {}
    z = np.load(os.path.join(OUT, "data.npz"))
    Xtr, Qtr, Xte, Qte = z["Xtr"], z["Qtr"], z["Xte"], z["Qte"]

    Btr = (Xtr.reshape(len(Xtr), -1) > 0.5).astype(np.float32)
    Bte = (Xte.reshape(len(Xte), -1) > 0.5).astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(Btr))
    vi, ti = perm[:len(Btr) // 5], perm[len(Btr) // 5:]
    on = Btr[ti].mean(0)
    cand = (on >= P.ON_LO) & (on <= P.ON_HI)
    Ftr, Fte = Btr[:, cand], Bte[:, cand]

    gt = (Qte[:, 1] > Qte[:, 0]).astype(int)
    gtr = (Qtr[:, 1] > Qtr[:, 0]).astype(int)
    m_te = Qte[:, 1] - Qte[:, 0]
    print(f"候选像素 {int(cand.sum())} / 25600   训练子集 {len(ti)}   验证 {len(vi)}   测试 {len(Xte)}")
    print(f"测试集拍翅率 {gt.mean():.2%}  ⇒ 「永远不拍翅」的总一致率就有 {1 - gt.mean():.2%}")

    flap_m, noflap_m = m_te[gt == 1], -m_te[gt == 0]
    print("\n" + "=" * 96)
    print("Q1  教师决策余量谱（margin = Q(s,1) − Q(s,0)）")
    print("=" * 96)
    print(f"  Q(s,1) 测试 std {Qte[:,1].std():.4f}   Q(s,0) 测试 std {Qte[:,0].std():.4f}")
    print(f"  拍翅帧余量   {qt(flap_m)}")
    print(f"  不拍帧余量   {qt(noflap_m)}")
    for t in (0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"  余量 < {t:<4} 的拍翅帧占比 {float((flap_m < t).mean()):.2%}")
    res["teacher"] = {"flap_rate": float(gt.mean()), "q1_std": float(Qte[:, 1].std()),
                      "margin_flap": qt(flap_m), "margin_noflap": qt(noflap_m),
                      "frac_margin_below": {str(t): float((flap_m < t).mean())
                                            for t in (0.05, 0.1, 0.2, 0.3, 0.5)}}

    print("\n" + "=" * 96)
    print("Q2  树 depth=8 线性叶（配置与 probe.py 一致）")
    print("=" * 96)
    t1 = time.time()
    trees = {a: P.fit_tree(Ftr[ti], Qtr[ti, a], DEPTH, "linear", LAM) for a in (0, 1)}
    P1t, P0t = P.predict(Fte, trees[1]), P.predict(Fte, trees[0])
    m_tr = Qtr[ti, 1] - Qtr[ti, 0]
    gtr_ti = (Qtr[ti, 1] > Qtr[ti, 0]).astype(int)
    e_tr = (P.predict(Ftr[ti], trees[1]) - P.predict(Ftr[ti], trees[0])) - m_tr
    d = evaluate(P1t, P0t, Qte, gt, m_te)
    d["leaves"] = trees[1]["leaves"]
    d["RAE_Q1_train"] = round(rae(Qtr[ti, 1], P.predict(Ftr[ti], trees[1])), 4)
    d["recall_flap_train"] = round(float(((e_tr > -m_tr)[gtr_ti == 1]).mean()), 4)
    d["sec"] = round(time.time() - t1, 1)
    d["name"] = "tree_d8_linear"
    print(f"  叶 {d['leaves']}  测试 RAE(Q1) {d['RAE_Q1']}  MAE {d['MAE_Q1']}"
          f"  召回 {d['recall_flap']:.2%}  一致 {d['agree']:.2%}")
    print(f"  RAE(Q1) 拆解：拍翅帧 {d['RAE_Q1_flap']}   不拍翅帧 {d['RAE_Q1_noflap']}")
    print(f"  拍翅帧有符号 margin 误差 {d['err_signed_flap']}   负误差占比 {d['frac_neg_err_flap']:.2%}")
    print(f"  训练集 RAE(Q1) {d['RAE_Q1_train']}  训练集召回 {d['recall_flap_train']:.2%}（{d['sec']} 秒）")
    res["tree"] = d

    P.MIN_CHILD = 8
    trees_b = {a: P.fit_tree(Ftr[ti], Qtr[ti, a], 10, "linear", LAM) for a in (0, 1)}
    P.MIN_CHILD = MINCHILD
    P1b, P0b = P.predict(Fte, trees_b[1]), P.predict(Fte, trees_b[0])
    db = evaluate(P1b, P0b, Qte, gt, m_te)
    db["leaves"] = trees_b[1]["leaves"]
    db["RAE_Q1_train"] = round(rae(Qtr[ti, 1], P.predict(Ftr[ti], trees_b[1])), 4)
    db["name"] = "tree_d10_mc8_linear"
    print(f"  参考档 depth=10/mc=8（树最好的一档）：叶 {db['leaves']}  RAE(Q1) {db['RAE_Q1']}"
          f"  拍翅帧 {db['RAE_Q1_flap']} / 不拍 {db['RAE_Q1_noflap']}  召回 {db['recall_flap']:.2%}")
    res["tree_best"] = db

    print("\n" + "=" * 96)
    print("Q3  CNN 损失权重扫描（唯一变量 = 拍翅帧权重 c）")
    print("=" * 96)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x_tr = torch.as_tensor(Xtr[ti], dtype=torch.float32)
    y_tr = torch.as_tensor(Qtr[ti], dtype=torch.float32)
    x_va = torch.as_tensor(Xtr[vi], dtype=torch.float32)
    y_va = torch.as_tensor(Qtr[vi], dtype=torch.float32)
    x_te = torch.as_tensor(Xte, dtype=torch.float32)
    gva = (Qtr[vi, 1] > Qtr[vi, 0]).astype(int)
    res["cnn"] = {}
    for c in C_GRID:
        w = np.ones(len(ti), dtype=np.float32)
        w[gtr[ti] == 1] = c
        w = w / w.mean()
        w_tr = torch.as_tensor(w, dtype=torch.float32)
        print(f"\n  c = {c:g}（拍翅帧权重，均值归一后有效样本数 {len(ti)}）")
        st_rae, st_rec, vrae, vrec = train_cnn(x_tr, y_tr, w_tr, x_va, y_va, gva, dev, f"c={c:g}")
        tag = f"c{c:g}"
        res["cnn"][tag] = {"c": c, "val_RAE": round(float(vrae), 4), "val_recall": round(float(vrec), 4)}
        for nm, st in (("ckpt_bestRAE", st_rae), ("ckpt_bestRecall", st_rec)):
            torch.save(st, os.path.join(OUT, f"cnn_{tag}_{nm}.pt"))
            net = QNet(2).to(dev)
            net.load_state_dict(st)
            net.eval()
            with torch.no_grad():
                p = net(x_te.to(dev)).cpu().numpy()
            dd = evaluate(p[:, 1], p[:, 0], Qte, gt, m_te)
            dd["name"] = f"cnn_{tag}_{nm}"
            res["cnn"][tag][nm] = dd
            print(f"    {nm:<16} RAE(Q1) {dd['RAE_Q1']:.4f}（拍翅帧 {dd['RAE_Q1_flap']:.4f} /"
                  f" 不拍 {dd['RAE_Q1_noflap']:.4f}）  召回 {dd['recall_flap']:.2%}"
                  f"  一致 {dd['agree']:.2%}")

    print("\n" + "=" * 96)
    print("Q4  总表：RAE 更好 ≠ 更会玩，以及反向偏差处方")
    print("=" * 96)
    rows = [("tree_d8", res["tree"]), ("tree_d10_mc8", res["tree_best"])]
    for tag, v in res["cnn"].items():
        rows.append((f"cnn_{tag}_bestRAE", v["ckpt_bestRAE"]))
        rows.append((f"cnn_{tag}_bestRec", v["ckpt_bestRecall"]))
    print(f"  {'模型':<26}{'RAE(Q1)':>9}{'拍翅帧':>9}{'不拍帧':>9}{'召回':>9}{'一致':>9}{'去偏后召回':>11}")
    for nm, v in rows:
        print(f"  {nm:<26}{v['RAE_Q1']:>9.4f}{v['RAE_Q1_flap']:>9.4f}{v['RAE_Q1_noflap']:>9.4f}"
              f"{v['recall_flap']:>9.2%}{v['agree']:>9.2%}{v['recall_if_debiased']['debiased']:>11.2%}")
    res["sec"] = round(time.time() - t0, 1)
    res["epochs"] = EPOCHS
    with open(os.path.join(OUT, "decompose.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n用时 {res['sec']} 秒    已写 interpretability/intrinsic/out/decompose.json")

if __name__ == "__main__":
    main()
