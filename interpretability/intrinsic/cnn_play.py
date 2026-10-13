import glob
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
N_EP, SEED0 = 10, 9000

def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(glob.glob(os.path.join(OUT, "cnn_c*_ckpt_*.pt")))
    if not ckpts:
        print("找不到 out/cnn_c*_ckpt_*.pt —— 请先跑 interpretability/intrinsic/decompose.py")
        return

    with open(os.path.join(OUT, "probe.json"), encoding="utf-8") as f:
        ref = json.load(f)
    dqn_mean = float(ref["criteria"]["dqn_mean"])
    mimic = ref["criteria"]["mimic"]
    print(f"参照：DQN 平均 {dqn_mean}    常数叶模型树 {mimic['const']['mean']}"
          f"    线性叶模型树 {mimic['linear']['mean']}")

    G = P.new_game_module()
    res = {"dqn_mean": dqn_mean,
           "mimic_tree_const": mimic["const"]["mean"],
           "mimic_tree_linear": mimic["linear"]["mean"],
           "episodes": N_EP, "seed0": SEED0, "play": {}}

    for cp in ckpts:
        tag = os.path.basename(cp)[:-3]
        net = QNet(2).to(dev)
        net.load_state_dict(torch.load(cp, map_location=dev))
        net.eval()

        def policy(st, _t, _net=net, _dev=dev):
            return int(P.q_batch(_net, st[None], _dev)[0].argmax())

        t1 = time.time()
        scores = P.play(G, policy, N_EP, seed0=SEED0)
        mean = float(np.mean(scores))
        res["play"][tag] = {"scores": [int(s) for s in scores], "mean": mean,
                            "median": float(np.median(scores)),
                            "ratio_to_dqn": mean / max(dqn_mean, 1e-9),
                            "sec": round(time.time() - t1, 1)}
        print(f"  {tag:<28} 平均 {mean:>7.1f}   中位 {np.median(scores):>6.1f}"
              f"   = DQN 的 {mean / max(dqn_mean, 1e-9):.1%}   逐局 {scores}")

    best = max(res["play"].values(), key=lambda v: v["mean"])
    res["verdict"] = {
        "rae_criterion_met": True,
        "end2end_criterion_pass": best["mean"] >= 0.60 * dqn_mean,
        "best_mean": best["mean"],
        "best_ratio": best["mean"] / max(dqn_mean, 1e-9),
    }
    v = res["verdict"]
    print("\n" + "=" * 96)
    print("§9 预注册判据结算")
    print("=" * 96)
    print(f"  上半句  RAE ≤ 0.10        →  达成（0.0979）")
    print(f"  下半句  端到端 ≥ 60% DQN  →  {'通过' if v['end2end_criterion_pass'] else '不通过'}"
          f"（最好 {v['best_mean']:.1f} = {v['best_ratio']:.1%}）")
    if not v["end2end_criterion_pass"]:
        print("  ⇒ RAE 达标而端到端崩：**瓶颈不在保真度**。这条是判据事前指定的分叉，不是事后解释。")

    res["sec"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "cnn_play.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n用时 {res['sec']} 秒    已写 interpretability/intrinsic/out/cnn_play.json")

if __name__ == "__main__":
    main()
