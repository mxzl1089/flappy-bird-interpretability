import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

def _imp(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

probe = _imp("probe")

P_LIST = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
PILOT_P = [0.0, 0.10]
N_EP = 30
SEED0 = 9000

DQN_REF = [2, 5, 22, 123, 76, 189, 134, 101, 375, 45]

def run_point(G, net, dev, p, n_ep):
    scores, lens = [], []
    for e in range(n_ep):
        rng = np.random.default_rng(7000 + e)
        probe.seed_env(G, SEED0 + e)
        env = G.GameState()
        a0 = np.zeros(2); a0[0] = 1
        frame, _, term = env.frame_step(a0, render=False)
        st, score, t = probe.init_state(frame), 0, 0
        while not term:
            q = probe.q_batch(net, st[None], dev)[0]
            a = int(q.argmax())
            if p > 0 and rng.random() < p:
                a = 1 - a
            av = np.zeros(2); av[a] = 1
            frame, r, term = env.frame_step(av, render=False)
            if r == 1.0:
                score += 1
            st = probe.next_state(st, frame)
            t += 1
        scores.append(score); lens.append(t)
    return scores, lens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--nep", type=int, default=N_EP)
    ap.add_argument("--ps", type=str, default="",
                    help="逗号分隔的 p 列表，覆盖默认（用于补测关键区间）")
    ap.add_argument("--tag", type=str, default="noise_tol",
                    help="输出文件名（不含 .json）")
    args = ap.parse_args()

    P = PILOT_P if args.pilot else P_LIST
    if args.ps:
        P = [float(x) for x in args.ps.split(",")]
    n_ep = 3 if args.pilot else args.nep
    out_json = os.path.join(OUT, args.tag + ".json")

    net, dev = probe.load_net()
    G = probe.new_game_module()
    print("设备 %s；p 扫描 %s；每点 %d 局；seed 配对（SEED0=%d）"
          % (dev, P, n_ep, SEED0))
    print("=" * 78)

    rows = []
    ok_selfcheck = None
    for p in P:
        t0 = time.time()
        sc, ln = run_point(G, net, dev, p, n_ep)
        sc, ln = np.array(sc), np.array(ln)
        if p == 0.0:

            got = list(map(int, sc[:len(DQN_REF)]))
            ok_selfcheck = (got == DQN_REF)
            print("★ 自检 p=0 复现 probe.py 的 DQN 逐局分数：%s" % ("一致" if ok_selfcheck else "不一致！"))
            print("   本脚本 %s" % got)
            print("   probe  %s" % DQN_REF)
            if not ok_selfcheck:
                print("   ⇒ harness 与 probe.py 不等价，后面的曲线不可用，先查这里")
            print()
        rows.append({"p": p, "n_ep": n_ep, "scores": list(map(int, sc)),
                     "score_mean": float(sc.mean()), "score_med": float(np.median(sc)),
                     "score_max": int(sc.max()), "score_min": int(sc.min()),
                     "len_mean": float(ln.mean()),
                     "n_dead_lt_300": int((ln < 300).sum()),
                     "secs": round(time.time() - t0, 1)})
        print("p=%5.3f  平均分 %7.2f  中位 %6.1f  [%d, %d]  平均存活 %6.0f 帧  "
              "早死(<300帧) %d/%d  %.0fs"
              % (p, sc.mean(), np.median(sc), sc.min(), sc.max(), ln.mean(),
                 rows[-1]["n_dead_lt_300"], n_ep, rows[-1]["secs"]))
        sys.stdout.flush()

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"p_list": P, "n_ep": n_ep, "seed0": SEED0,
                   "dqn_ref": DQN_REF, "selfcheck_p0": ok_selfcheck,
                   "rows": rows}, fh, ensure_ascii=False, indent=1)
    print("=" * 78)
    print("已写 " + os.path.basename(out_json))

if __name__ == "__main__":
    main()
