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
N_EP, SEED0 = 10, 9000
QTS = (5, 25, 50, 75, 95)

def qt(y):
    return {f"p{k}": round(float(np.percentile(y, k)), 4) for k in QTS}

def main():
    t0 = time.time()
    z = np.load(os.path.join(OUT, "data.npz"))
    Xtr, Qtr, Xte, Qte = z["Xtr"], z["Qtr"], z["Xte"], z["Qte"]
    m_tr, m_te = Qtr[:, 1] - Qtr[:, 0], Qte[:, 1] - Qte[:, 0]

    perm = np.random.default_rng(0).permutation(len(Xtr))
    vi, ti = perm[:len(Xtr) // 5], perm[len(Xtr) // 5:]
    gt, gva = (m_te > 0).astype(int), (m_tr[vi] > 0).astype(int)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x_tr = torch.as_tensor(Xtr[ti], dtype=torch.float32)
    y_tr = torch.as_tensor(m_tr[ti, None], dtype=torch.float32)
    x_va = torch.as_tensor(Xtr[vi], dtype=torch.float32)
    y_va = torch.as_tensor(m_tr[vi, None], dtype=torch.float32)
    x_te = torch.as_tensor(Xte, dtype=torch.float32)
    print(f"设备 {dev}  训练 {len(ti)} / 验证 {len(vi)} / 测试 {len(Xte)}"
          f"   单头回归 margin（QNet(1)）")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = QNet(1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.MSELoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    def val_rae():
        net.eval()
        with torch.no_grad():
            p = net(x_va.to(dev)).cpu().numpy().ravel()
        base = m_tr[vi].mean()
        return float(np.abs(p - m_tr[vi]).sum() / max(np.abs(base - m_tr[vi]).sum(), 1e-9)), p

    best, st_best = np.inf, None
    best_rec, st_rec = -1.0, None
    for ep in range(EPOCHS):
        net.train()
        idx = torch.randperm(len(x_tr))
        for b in range(0, len(idx), BS):
            j = idx[b:b + BS]
            opt.zero_grad()
            loss = lossf(net(x_tr[j].to(dev)), y_tr[j].to(dev))
            loss.backward()
            opt.step()
        sched.step()
        r, p = val_rae()
        rec = float((((p > 0).astype(int))[gva == 1] == 1).mean())
        if r < best:
            best, st_best = r, {k: v.clone() for k, v in net.state_dict().items()}
        if rec > best_rec:
            best_rec, st_rec = rec, {k: v.clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 100 == 0:
            print(f"    epoch {ep+1:>3}  验证 RAE(margin) {r:.4f}  验证召回 {rec:.4f}")

    def report(nm, st):
        net.load_state_dict(st)
        net.eval()
        with torch.no_grad():
            mh = net(x_te.to(dev)).cpu().numpy().ravel()
        e = mh - m_te
        pred = (mh > 0).astype(int)
        base = m_te.mean()
        d = {"name": nm,
             "recall_flap": round(float((pred[gt == 1] == 1).mean()), 4),
             "spec_noflap": round(float((pred[gt == 0] == 0).mean()), 4),
             "agree": round(float((pred == gt).mean()), 4),
             "MAE_margin": round(float(np.abs(e).mean()), 4),
             "RAE_margin": round(float(np.abs(e).sum() / max(np.abs(base - m_te).sum(), 1e-9)), 4),
             "err_signed_flap": qt(e[gt == 1]),
             "err_signed_noflap": qt(e[gt == 0]),
             "frac_neg_err_flap": round(float((e[gt == 1] < 0).mean()), 4),
             "margin_std": round(float(m_te.std()), 4)}
        print(f"    {nm:<18} margin RAE {d['RAE_margin']:.4f}  MAE {d['MAE_margin']:.4f}"
              f"   召回 {d['recall_flap']:.2%}  一致 {d['agree']:.2%}")
        print(f"      拍翅帧有符号误差 {d['err_signed_flap']}   负误差占比 {d['frac_neg_err_flap']:.2%}")
        return d, mh

    print("\n  --- 两个检查点 ---")
    d_rae, mh_rae = report("bestRAE", st_best)
    d_rec, _ = report("bestRecall", st_rec)

    res = {"multilhead_vs_single": {
               "note": "多头=QNet(2) 分别回归 Q0/Q1（decompose.py）；单头=QNet(1) 直接回归 margin",
               "single_bestRAE": d_rae, "single_bestRecall": d_rec},
           "epochs": EPOCHS, "sec": round(time.time() - t0, 1)}

    torch.save(st_best, os.path.join(OUT, "marginhead_bestRAE.pt"))
    G = P.new_game_module()
    net.load_state_dict(st_best)
    net.eval()

    def policy(st, _t):
        with torch.no_grad():
            x = torch.as_tensor(st[None], dtype=torch.float32, device=dev)
            return int(net(x).item() > 0)

    scores = P.play(G, policy, N_EP, seed0=SEED0)
    with open(os.path.join(OUT, "probe.json"), encoding="utf-8") as f:
        ref = json.load(f)
    dqn_mean = float(ref["criteria"]["dqn_mean"])
    res["end2end"] = {"scores": [int(s) for s in scores], "mean": float(np.mean(scores)),
                      "median": float(np.median(scores)), "dqn_mean": dqn_mean,
                      "ratio_to_dqn": float(np.mean(scores)) / dqn_mean}
    print(f"\n  端到端（单头 margin 策略）平均 {np.mean(scores):.1f} / 中位 {np.median(scores):.1f}"
          f"   = DQN {dqn_mean} 的 {np.mean(scores) / dqn_mean:.1%}   逐局 {list(map(int, scores))}")

    res["sec"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "margin_head.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n用时 {res['sec']} 秒    已写 interpretability/intrinsic/out/margin_head.json")

if __name__ == "__main__":
    main()
