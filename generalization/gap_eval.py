import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, required=True, help="上下管之间的缝隙高度（px）")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--ckpt", type=str, default="runs/ckpt-1890000.pt")
    ap.add_argument("--cap", type=int, default=1000, help="单局封顶分数（>=cap 直接结束并记为 cap）")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    os.chdir(ROOT)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    sys.path.insert(0, str(ROOT / "dqn"))
    sys.path.insert(0, str(ROOT / "env"))

    import torch
    import wrapped_flappy_bird as game_env
    from config import Config
    from model import QNet
    from utils import init_state, next_state

    game_env.PIPEGAPSIZE = args.gap
    u, l = game_env.getRandomPipe()
    real_gap = l["y"] - (u["y"] + game_env.PIPE_HEIGHT)
    assert real_gap == args.gap, f"缝隙高度未生效: {real_gap} != {args.gap}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    net = QNet(Config.NUM_ACTIONS).to(device)
    ckpt_path = ROOT / args.ckpt
    sd = torch.load(str(ckpt_path), map_location=device)
    net.load_state_dict(sd["policy"] if isinstance(sd, dict) and "policy" in sd else sd)
    net.eval()

    tag = f"gap={args.gap}"
    print(f"[{tag}] ckpt={ckpt_path.name}  实测缝隙={real_gap}px  "
          f"episodes={args.episodes}  cap={args.cap}  seed={args.seed}", flush=True)

    env = game_env.GameState()
    scores = []
    t_all = time.time()
    for ep in range(args.episodes):
        t0 = time.time()
        do_nothing = np.zeros(Config.NUM_ACTIONS)
        do_nothing[0] = 1
        frame, _r, terminal = env.frame_step(do_nothing, render=False)
        state = init_state(frame)

        score = 0
        steps = 0
        while not terminal:
            s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(net(s).argmax(dim=1).item())
            a_vec = np.zeros(Config.NUM_ACTIONS)
            a_vec[action] = 1
            frame, reward, terminal = env.frame_step(a_vec, render=False)
            if reward == 1.0:
                score += 1
            state = next_state(state, frame)
            steps += 1
            if score >= args.cap:
                terminal = True
        scores.append(score)
        print(f"[{tag}] ep {ep+1:>2}/{args.episodes}  score={score:>4}  steps={steps:>6}  "
              f"ep_time={time.time()-t0:.1f}s", flush=True)

    arr = np.array(scores, dtype=float)
    stat = {
        "gap": args.gap,
        "ckpt": ckpt_path.name,
        "episodes": args.episodes,
        "cap": args.cap,
        "seed": args.seed,
        "scores": scores,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "n_capped": int((arr >= args.cap).sum()),
        "wall_seconds": round(time.time() - t_all, 1),
    }
    print(f"[{tag}] 均值={stat['mean']:.2f}  标准差={stat['std']:.2f}  中位数={stat['median']:.1f}  "
          f"最高={stat['max']:.0f}  最低={stat['min']:.0f}  "
          f"(封顶 {stat['n_capped']} 局)  用时={stat['wall_seconds']}s", flush=True)

    if args.out:
        outp = Path(args.out)
        if not outp.is_absolute():
            outp = ROOT / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(stat, f, ensure_ascii=False, indent=2)
        print(f"[{tag}] 结果已写入 {outp}", flush=True)

if __name__ == "__main__":
    main()
