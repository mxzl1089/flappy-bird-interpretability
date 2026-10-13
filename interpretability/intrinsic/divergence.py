import importlib.util
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
from model import QNet

OUT = os.path.join(HERE, "out")
MIMIC_CKPT = "cnn_c5_ckpt_bestRAE.pt"
N_EP, SEED0 = 10, 9000
BINS = [(0, 100), (100, 300), (300, 700), (700, 1500), (1500, 10 ** 9)]

def load_mimic(dev):
    p = os.path.join(OUT, MIMIC_CKPT)
    if not os.path.exists(p):
        raise SystemExit(f"找不到 {p} —— 请先跑 interpretability/intrinsic/decompose.py")
    net = QNet(2).to(dev)
    net.load_state_dict(torch.load(p, map_location=dev))
    net.eval()
    return net

def play_pair(G, dqn, mimic, dev, n_ep, seed0):
    eps = []
    for e in range(n_ep):
        P.seed_env(G, seed0 + e)
        env = G.GameState()
        a0 = np.zeros(2); a0[0] = 1
        frame, _, term = env.frame_step(a0, render=False)
        st, score, t = P.init_state(frame), 0, 0
        n_agree = n = n_dqn_flap = n_miss_flap = 0
        per_t = []
        while not term:
            ad = int(P.q_batch(dqn, st[None], dev)[0].argmax())
            am = int(P.q_batch(mimic, st[None], dev)[0].argmax())
            n += 1
            ok = (ad == am)
            n_agree += ok
            if ad == 1:
                n_dqn_flap += 1
                if am != 1:
                    n_miss_flap += 1
            per_t.append(ok)
            av = np.zeros(2); av[am] = 1
            frame, r, term = env.frame_step(av, render=False)
            if r == 1.0:
                score += 1
            st = P.next_state(st, frame)
            t += 1
        eps.append({"ep": e, "seed": seed0 + e, "score": score, "frames": t,
                    "disagree": 1.0 - n_agree / max(n, 1),
                    "dqn_flap": n_dqn_flap, "miss_flap": n_miss_flap,
                    "miss_flap_rate": n_miss_flap / max(n_dqn_flap, 1),
                    "per_t": per_t})
    return eps

def expert_baseline(dqn, mimic, dev):
    z = np.load(os.path.join(OUT, "data.npz"))
    Xte = z["Xte"]
    ad = P.q_batch(dqn, Xte, dev).argmax(1)
    am = P.q_batch(mimic, Xte, dev).argmax(1)
    dis = float((ad != am).mean())
    flap = ad == 1
    return {"n": int(len(Xte)), "disagree": dis,
            "dqn_flap_rate": float(flap.mean()),
            "miss_flap_rate": float((am[flap] != 1).mean()),
            "false_flap_rate": float((am[~flap] == 1).mean())}

def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dqn, dev = P.load_net()
    mimic = load_mimic(dev)
    G = P.new_game_module()
    print(f"设备 {dev}；模仿器 {MIMIC_CKPT}；让**模仿器自己玩** {N_EP} 局，"
          f"逐帧与 DQN 对照")
    print("=" * 78)

    base = expert_baseline(dqn, mimic, dev)
    print("【参照】专家测试集 %d 帧（无漂移）：分歧率 %.2f%%；"
          "DQN 拍翅帧占 %.1f%%，其中模仿器漏拍 %.1f%%"
          % (base["n"], base["disagree"] * 100, base["dqn_flap_rate"] * 100,
             base["miss_flap_rate"] * 100))
    print()

    eps = play_pair(G, dqn, mimic, dev, N_EP, SEED0)
    sc = np.array([e["score"] for e in eps])
    fr = np.array([e["frames"] for e in eps])
    print("【实玩】逐局 分数 %s" % list(map(int, sc)))
    print("       平均分 %.1f   平均存活 %.0f 帧   逐局存活 %s"
          % (sc.mean(), fr.mean(), list(map(int, fr))))
    print()

    print("【分歧率按「已存活帧数」分箱】—— 上升 ⇒ 协变量漂移的直接证据")
    tot = [0] * len(BINS)
    dis = [0] * len(BINS)
    for e in eps:
        for t, ok in enumerate(e["per_t"]):
            for i, (a, b) in enumerate(BINS):
                if a <= t < b:
                    tot[i] += 1
                    dis[i] += (not ok)
                    break
    bins = []
    for i, (a, b) in enumerate(BINS):
        if tot[i] == 0:
            continue
        rr = dis[i] / tot[i]
        bins.append({"lo": a, "hi": (None if b > 10 ** 8 else b),
                     "n": tot[i], "disagree": float(rr)})
        tag = f"{a}–{b}" if b < 10 ** 8 else f"{a}+"
        print("  存活 %-10s  样本 %7d 帧   分歧率 %6.2f%%" % (tag, tot[i], rr * 100))
    print()
    print("【逐局分歧率 vs 分数】")
    for e in eps:
        print("  局%d seed%d  分数 %4d  存活 %5d 帧   分歧率 %5.2f%%   "
              "漏拍翅 %4d/%4d = %5.1f%%"
              % (e["ep"], e["seed"], e["score"], e["frames"], e["disagree"] * 100,
                 e["miss_flap"], e["dqn_flap"], e["miss_flap_rate"] * 100))

    res = {"mimic_ckpt": MIMIC_CKPT, "n_ep": N_EP, "seed0": SEED0,
           "expert_baseline": base,
           "scores": list(map(int, sc)), "score_mean": float(sc.mean()),
           "frames_mean": float(fr.mean()),
           "disagree_mean_play": float(np.mean([e["disagree"] for e in eps])),
           "bins": bins,
           "episodes": [{k: v for k, v in e.items() if k != "per_t"} for e in eps],
           "sec": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "divergence.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n用时 {res['sec']} 秒    已写 out/divergence.json")

if __name__ == "__main__":
    main()
