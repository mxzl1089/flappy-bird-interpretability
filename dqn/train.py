import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from config import Config
from model import QNet
from replay import ReplayBuffer
from utils import init_state, next_state

def make_env(no_render: bool = False):

    os.chdir(Config.ROOT)
    if no_render:

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.append(Config.GAME_DIR)
    import wrapped_flappy_bird as game_env
    return game_env.GameState()

def select_action(policy_net, state, epsilon, num_actions, device) -> int:
    if random.random() < epsilon:
        return random.randrange(num_actions)

    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(s)
        return int(q_values.argmax(dim=1).item())

def get_epsilon(t, cfg):
    if t <= cfg.OBSERVE:
        return cfg.INIT_EPSILON
    progress = (t - cfg.OBSERVE) / cfg.EXPLORE
    progress = max(0.0, min(1.0, progress))
    return max(cfg.FINAL_EPSILON,
               cfg.INIT_EPSILON - (cfg.INIT_EPSILON - cfg.FINAL_EPSILON) * progress)

def train_step(policy_net, target_net, optimizer, memory, cfg, device) -> float:
    states, actions, rewards, next_states, dones = memory.sample(cfg.BATCH_SIZE, device)

    q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        targets = rewards + (1.0 - dones) * cfg.GAMMA * next_q

    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def save_ckpt(step, policy_net, target_net, optimizer, cfg) -> str:
    path = os.path.join(cfg.CKPT_DIR, f"ckpt-{step}.pt")
    torch.save({
        "policy": policy_net.state_dict(),
        "target": target_net.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
    }, path)
    with open(cfg.LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump({"model_file": path, "step": step}, f, indent=2)
    return path

def load_latest(policy_net, target_net, optimizer, cfg, device) -> int:
    if not os.path.exists(cfg.LATEST_JSON):
        return 0
    with open(cfg.LATEST_JSON, "r", encoding="utf-8") as f:
        info = json.load(f)
    if not os.path.exists(info.get("model_file", "")):
        return 0
    ck = torch.load(info["model_file"], map_location=device)
    policy_net.load_state_dict(ck["policy"])
    target_net.load_state_dict(ck["target"])
    optimizer.load_state_dict(ck["optim"])
    return int(info["step"])

def main():
    parser = argparse.ArgumentParser(description="训练 Flappy Bird DQN（PyTorch 版）")
    parser.add_argument("--fresh", action="store_true",
                        help="从零训练（忽略已有权重）")
    parser.add_argument("--no-render", action="store_true",
                        help="关闭窗口渲染，训练提速数十倍（强烈推荐）")
    parser.add_argument("--max-steps", type=int, default=Config.MAX_STEPS,
                        help="总训练步数上限")
    args = parser.parse_args()

    cfg = Config
    device = cfg.device()
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    render = not args.no_render

    writer = None
    if cfg.USE_TB and SummaryWriter is not None:
        os.makedirs(cfg.TB_DIR, exist_ok=True)
        writer = SummaryWriter(cfg.TB_DIR)
        print(f"TensorBoard 日志: {cfg.TB_DIR}")

    print(f"使用设备: {device}")
    env = make_env(args.no_render)

    policy_net = QNet(cfg.NUM_ACTIONS).to(device)
    target_net = QNet(cfg.NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.LR)
    memory = ReplayBuffer(cfg.REPLAY_SIZE)

    start_step = 0
    if not args.fresh:
        start_step = load_latest(policy_net, target_net, optimizer, cfg, device)
        if start_step > 0:
            print(f"续训：已从 step={start_step} 恢复，继续训练")
        else:
            print("没找到存档 → 从零开始训练（随机初始化权重）")
    else:
        print("从零开始训练（--fresh，忽略已有存档）")

    do_nothing = np.zeros(cfg.NUM_ACTIONS)
    do_nothing[0] = 1
    frame, _r, terminal = env.frame_step(do_nothing, render=render)
    state = init_state(frame)

    t = start_step
    episode_score = 0
    loss_acc, loss_cnt = 0.0, 0

    while t < args.max_steps:

        epsilon = get_epsilon(t, cfg)

        action = select_action(policy_net, state, epsilon, cfg.NUM_ACTIONS, device)
        a_vec = np.zeros(cfg.NUM_ACTIONS)
        a_vec[action] = 1

        frame, reward, terminal = env.frame_step(a_vec, render=render)
        nxt = next_state(state, frame)
        if reward == 1.0:
            episode_score += 1

        memory.push(state, action, reward, nxt, float(terminal))

        loss, trained = 0.0, False
        if t > cfg.OBSERVE and len(memory) >= cfg.BATCH_SIZE:
            loss = train_step(policy_net, target_net, optimizer, memory, cfg, device)
            trained = True

            if cfg.USE_TARGET_NET and t % cfg.TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())

        state = nxt
        t += 1

        if trained:
            loss_acc += loss
            loss_cnt += 1
        if t % cfg.LOG_EVERY == 0:
            phase = ("观察期" if t <= cfg.OBSERVE
                     else "探索期" if epsilon > cfg.FINAL_EPSILON else "训练期")
            avg_loss = loss_acc / max(1, loss_cnt)
            print(f"步数 {t} / {phase} / ε {epsilon:.6f} / 动作 {action} "
                  f"/ 奖励 {reward} / 本局得分 {episode_score} / 平均loss {avg_loss:.6f}")
            if writer is not None:

                with torch.no_grad():
                    q_max = float(policy_net(
                        torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    ).max())
                writer.add_scalar("step/reward", reward, t)
                writer.add_scalar("step/avg_loss", avg_loss, t)
                writer.add_scalar("step/epsilon", epsilon, t)
                writer.add_scalar("step/q_max", q_max, t)
            loss_acc, loss_cnt = 0.0, 0

        if terminal:
            if writer is not None:
                writer.add_scalar("episode/score", episode_score, t)
            episode_score = 0

        if t % cfg.SAVE_EVERY == 0:
            path = save_ckpt(t, policy_net, target_net, optimizer, cfg)
            print(f"  → 已存档 {path}")

    path = save_ckpt(t, policy_net, target_net, optimizer, cfg)
    if writer is not None:
        writer.close()
    print(f"训练结束，最终模型已保存到 {path}")

if __name__ == "__main__":
    main()
