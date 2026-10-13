import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dqn"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.chdir(ROOT)

import probe as P

OUT = os.path.join(HERE, "out")
K_LIST = [1, 2, 5, 10, 20]
SIG_LIST = [1.0, 3.0, 10.0]
N_REP = 5

def main():
    t0 = time.time()
    net, dev = P.load_net()
    z = np.load(os.path.join(OUT, "data.npz"))
    X = z["Xte"].astype(np.float32)
    N = len(X)
    q0 = P.q_batch(net, X, dev)
    a0 = q0.argmax(1)
    margin = q0[:, 1] - q0[:, 0]
    am = np.abs(margin)
    print(f"设备 {dev}；教师测试帧 {N} 帧")
    print("=" * 78)

    print("① 教师决策余量 |Q(s,1) − Q(s,0)| 的分布（全部测试帧）")
    base_flap = float((a0 == 1).mean())
    print(f"   贪心动作分布：拍翅 {base_flap:.2%} / 不拍 {1 - base_flap:.2%}")
    pcts = [5, 25, 50, 75, 90, 95, 99]
    pv = {f"p{p}": float(np.percentile(am, p)) for p in pcts}
    print("   |margin| 分位：" + "  ".join(f"p{p}={np.percentile(am, p):.4f}" for p in pcts))
    thr = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    fps = {f"{t}": float((am < t).mean()) for t in thr}
    print("   |margin| < δ 的比例：" + "  ".join(f"δ={t}:{fps[str(t)]:.1%}" for t in thr))
    print("   ⇒ 这一档就是「任何回归器误差超过 δ 就必然改判」的帧占比")
    print()

    rng = np.random.default_rng(0)
    print("② 极小输入扰动下的动作翻转率（与干净图 argmax 不同的比例）")
    flips = {"pixel": {}, "gauss": {}}
    for k in K_LIST:
        rates = []
        for _ in range(N_REP):
            Xp = X.copy()

            ii = rng.integers(0, 4, size=(N, k))
            jj = rng.integers(0, 80, size=(N, k))
            kk = rng.integers(0, 80, size=(N, k))
            rows = np.arange(N)[:, None]
            cur = Xp[rows, ii, jj, kk]
            Xp[rows, ii, jj, kk] = np.where(cur > 0.5, 0.0, 255.0)
            ap = P.q_batch(net, Xp, dev).argmax(1)
            rates.append(float((ap != a0).mean()))
        flips["pixel"][str(k)] = float(np.mean(rates))
        print(f"   翻转 {k:>2} 个像素 / 帧   → 动作改变 {np.mean(rates):.3%}")
    for s in SIG_LIST:
        rates = []
        for _ in range(N_REP):
            Xp = np.clip(X + rng.normal(0, s, X.shape), 0, 255).astype(np.float32)
            ap = P.q_batch(net, Xp, dev).argmax(1)
            rates.append(float((ap != a0).mean()))
        flips["gauss"][str(s)] = float(np.mean(rates))
        print(f"   高斯噪声 σ={s:<4} (0~255 尺度) → 动作改变 {np.mean(rates):.3%}")
    print()

    print("③ 按 |margin| 分箱看翻转率（1 个像素的扰动，重复 %d 次取均值）" % N_REP)
    Xp = X.copy()
    ii = rng.integers(0, 4, size=(N, 1)); jj = rng.integers(0, 80, size=(N, 1))
    kk = rng.integers(0, 80, size=(N, 1))
    rows = np.arange(N)[:, None]
    cur = Xp[rows, ii, jj, kk]
    Xp[rows, ii, jj, kk] = np.where(cur > 0.5, 0.0, 255.0)
    ap = P.q_batch(net, Xp, dev).argmax(1)
    changed = (ap != a0)
    edges = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1e9]
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (am >= a) & (am < b)
        if m.sum() == 0:
            continue
        bins.append({"lo": a, "hi": (None if b > 1e8 else b), "n": int(m.sum()),
                     "flip": float(changed[m].mean())})
        tag = f"[{a:g},{b:g})" if b < 1e8 else f"[{a:g},∞)"
        print("   |margin| %-12s 帧数 %5d   翻转率 %6.2f%%" % (tag, m.sum(), changed[m].mean() * 100))
    print()

    print("④ 交叉核对")
    print(f"   · 模仿器要够到论文量级，需把逐帧分歧压到 ≲0.5%（见 noise_tol：")
    print(f"     p=0.005 时教师自己只剩 11.9 分）")
    print(f"   · 本实验给出的地板：单像素扰动就改变 {flips['pixel']['1']:.2%} 帧的动作")
    print(f"   · 当前最好模仿器的实测分歧：4.75%（专家分布）")
    print(f"   ⇒ 若地板 ≳ 1%，则 0.5% 的分歧率在本目标上不可达。")
    print()

    res = {"n_test": int(N), "flap_rate": base_flap,
           "margin_percentiles": pv, "margin_below": fps,
           "q1_std": float(q0[:, 1].std()),
           "flip_pixel": flips["pixel"], "flip_gauss": flips["gauss"],
           "flip_bins": bins, "sec": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "policy_noise.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"用时 {res['sec']} 秒    已写 out/policy_noise.json")

if __name__ == "__main__":
    main()
