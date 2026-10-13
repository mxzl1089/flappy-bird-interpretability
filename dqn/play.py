import argparse
import json
import os
import sys

import numpy as np
import torch

from config import Config
from model import QNet
from utils import init_state, next_state

@torch.no_grad()
def play(n_episodes: int = 10, render: bool = True):
    cfg = Config
    device = cfg.device()

    if not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    os.chdir(cfg.ROOT)
    sys.path.append(cfg.GAME_DIR)
    import wrapped_flappy_bird as game_env

    net = QNet(cfg.NUM_ACTIONS).to(device)
    if not os.path.exists(cfg.LATEST_JSON):
        raise FileNotFoundError(f"找不到存档指针 {cfg.LATEST_JSON}，请先运行 train.py 训练")
    with open(cfg.LATEST_JSON, "r", encoding="utf-8") as f:
        info = json.load(f)
    net.load_state_dict(torch.load(info["model_file"], map_location=device)["policy"])
    net.eval()
    print(f"已加载 {info['model_file']}（训练步数 {info['step']}）")
    print(f"模式：{'窗口渲染（慢）' if render else '无头快速评估'}")

    env = game_env.GameState()
    scores = []

    for ep in range(n_episodes):

        do_nothing = np.zeros(cfg.NUM_ACTIONS)
        do_nothing[0] = 1
        frame, _r, terminal = env.frame_step(do_nothing, render=render)
        state = init_state(frame)

        score = 0
        while not terminal:
            s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(net(s).argmax(dim=1).item())
            a_vec = np.zeros(cfg.NUM_ACTIONS)
            a_vec[action] = 1
            frame, reward, terminal = env.frame_step(a_vec, render=render)
            if reward == 1.0:
                score += 1
            state = next_state(state, frame)

        scores.append(score)
        print(f"第 {ep + 1:>2} 局：过了 {score} 根管")

    print(f"\n平均 {np.mean(scores):.2f} 根 / 最高 {max(scores)} 根 / 最低 {min(scores)} 根")
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加载模型看 AI 玩 Flappy Bird")
    parser.add_argument("--episodes", type=int, default=10, help="玩几局（默认 10）")

    parser.add_argument("--no-render", action="store_true",
                        help="关闭窗口渲染，无头快速评估（推荐大盘评估时用）")
    args = parser.parse_args()

    play(n_episodes=args.episodes, render=not args.no_render)
