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
os.chdir(ROOT)

from model import QNet

OUT = os.path.join(HERE, "out")
EPOCHS, BS, LR = 300, 64, 1e-3
SEED = 0

def metrics(y, p):
    base = y.mean()
    return {"MAE": float(np.abs(p - y).mean()),
            "RAE": float(np.abs(p - y).sum() / max(np.abs(base - y).sum(), 1e-9)),
            "RSE": float(((p - y) ** 2).sum() / max(((base - y) ** 2).sum(), 1e-9))}

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    z = np.load(os.path.join(OUT, "data.npz"))
    Xtr, Qtr, Xte, Qte = z["Xtr"], z["Qtr"], z["Xte"], z["Qte"]
    perm = np.random.default_rng(0).permutation(len(Xtr))
    vi, ti = perm[:len(Xtr) // 5], perm[len(Xtr) // 5:]

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    x_tr = torch.as_tensor(Xtr[ti], dtype=torch.float32)
    y_tr = torch.as_tensor(Qtr[ti], dtype=torch.float32)
    x_va = torch.as_tensor(Xtr[vi], dtype=torch.float32)
    y_va = torch.as_tensor(Qtr[vi], dtype=torch.float32)
    x_te = torch.as_tensor(Xte, dtype=torch.float32)
    print(f"设备 {dev}  训练 {len(ti)} / 验证 {len(vi)} / 测试 {len(Xte)}")

    net = QNet(2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.MSELoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best, best_state = np.inf, None
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        idx = torch.randperm(len(x_tr))
        for b in range(0, len(idx), BS):
            j = idx[b:b + BS]
            xb, yb = x_tr[j].to(dev), y_tr[j].to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vp = net(x_va.to(dev)).cpu().numpy()
        r = metrics(y_va.numpy(), vp)["RAE"]
        if r < best:
            best = r
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:>3}  验证 RAE(Q1) {metrics(y_va.numpy(), vp)['RAE']:.4f}"
                  f"  Q1 {metrics(y_va.numpy()[:,1], vp[:,1])['RAE']:.4f}")
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p_tr = net(x_tr.to(dev)).cpu().numpy()
        p_te = net(x_te.to(dev)).cpu().numpy()
    res = {}
    for a, nm in ((0, "Q0"), (1, "Q1")):
        m_te, m_tr = metrics(Qte[:, a], p_te[:, a]), metrics(Qtr[ti][:, a], p_tr[:, a])
        res[nm] = {"test": m_te, "train": m_tr}
        print(f"  {nm}: 测试 RAE {m_te['RAE']:.4f} / MAE {m_te['MAE']:.4f}"
              f"    训练 RAE {m_tr['RAE']:.4f} / MAE {m_tr['MAE']:.4f}")
    g = (Qte[:, 1] > Qte[:, 0]).astype(int)
    pf = (p_te[:, 1] > p_te[:, 0]).astype(int)
    recall = float((pf[g == 1] == 1).mean())
    print(f"  决策：总一致 {float((pf == g).mean()):.2%}  拍翅召回 {recall:.2%}")
    res["decision"] = {"agree": float((pf == g).mean()), "recall_flap": recall}
    res["epochs"] = EPOCHS
    res["sec"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "ceiling_cnn.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"  （{res['sec']} 秒）已写 out/ceiling_cnn.json")

if __name__ == "__main__":
    main()
