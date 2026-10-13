import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from config import Config
from model import ActorCritic
from rollout import RolloutBuffer
from utils import init_state, next_state

def make_env(no_render: bool = False):
    os.chdir(Config.PROJECT_ROOT)
    if no_render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.append(Config.GAME_DIR)
    import wrapped_flappy_bird as game_env
    return game_env.GameState()

def linear_schedule(start: float, end: float, progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return end + (start - end) * (1.0 - progress)

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y == 0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)

def collect_rollout(env, net, buffer, state, cfg, device, render, episode_score):
    buffer.reset()
    finished = []
    reward_sum = entropy_sum = value_sum = 0.0
    flap_count = 0

    for _ in range(cfg.N_STEPS):

        s_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a_t, logp_t, ent_t, v_t = net.get_action_and_value(s_t)
        action = int(a_t.item())
        log_prob = float(logp_t.item())
        entropy = float(ent_t.item())
        value = float(v_t.item())

        a_vec = np.zeros(cfg.NUM_ACTIONS)
        a_vec[action] = 1
        frame, reward, terminal = env.frame_step(a_vec, render=render)
        if reward == 1.0:
            episode_score += 1

        nxt = init_state(frame) if terminal else next_state(state, frame)

        buffer.add(state, action, float(reward), float(terminal), log_prob, value)

        state = nxt
        reward_sum += reward
        entropy_sum += entropy
        value_sum += value
        flap_count += action

        if terminal:
            finished.append(episode_score)
            episode_score = 0

    stats = {
        "reward_mean": reward_sum / cfg.N_STEPS,
        "entropy_mean": entropy_sum / cfg.N_STEPS,
        "value_mean": value_sum / cfg.N_STEPS,
        "flap_ratio": flap_count / cfg.N_STEPS,
        "n_episodes": len(finished),
        "score_mean": float(np.mean(finished)) if finished else float("nan"),
    }
    return state, episode_score, finished, stats

def update_policy(net, optimizer, buffer, cfg, device, ent_coef):
    n_mb = 0
    pg_sum = v_sum = e_sum = tot_sum = kl_sum = ok_sum = cf_sum = 0.0
    early_stopped = False

    for epoch in range(cfg.N_EPOCHS):
        for mb in buffer.get_minibatches(cfg.BATCH_SIZE, n_epochs=1, device=device):

            logprobs, entropy, values = net.evaluate_actions(mb.states, mb.actions)

            advantages = mb.advantages
            if cfg.NORMALIZE_ADV and cfg.NORM_ADV_PER_MINIBATCH:

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(logprobs - mb.old_logprobs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_RANGE, 1.0 + cfg.CLIP_RANGE) * advantages
            pg_loss = -torch.min(surr1, surr2).mean()

            if cfg.CLIP_VLOSS:

                v_loss_unclipped = (values - mb.returns) ** 2
                v_clipped = mb.old_values + torch.clamp(
                    values - mb.old_values, -cfg.CLIP_RANGE, cfg.CLIP_RANGE)
                v_loss_clipped = (v_clipped - mb.returns) ** 2
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = F.mse_loss(values, mb.returns)

            entropy_loss = -entropy.mean()

            loss = pg_loss + cfg.VF_COEF * v_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                logratio = logprobs - mb.old_logprobs

                approx_kl = ((ratio - 1.0) - logratio).mean()

                old_approx_kl = (-logratio).mean()

                clip_frac = ((ratio - 1.0).abs() > cfg.CLIP_RANGE).float().mean()

            n_mb += 1
            pg_sum += pg_loss.item()
            v_sum += v_loss.item()
            e_sum += entropy.mean().item()
            tot_sum += loss.item()
            kl_sum += approx_kl.item()
            ok_sum += old_approx_kl.item()
            cf_sum += clip_frac.item()

        if cfg.TARGET_KL is not None and (kl_sum / max(1, n_mb)) > cfg.TARGET_KL:
            early_stopped = True
            break

    return {
        "policy_loss": pg_sum / n_mb,
        "value_loss": v_sum / n_mb,
        "entropy": e_sum / n_mb,
        "total_loss": tot_sum / n_mb,
        "approx_kl": kl_sum / n_mb,
        "old_approx_kl": ok_sum / n_mb,
        "clip_fraction": cf_sum / n_mb,
        "early_stopped": early_stopped,
        "explained_variance": explained_variance(buffer.values, buffer.returns),
    }

def save_ckpt(step, net, optimizer, cfg) -> str:
    path = os.path.join(cfg.CKPT_DIR, f"ckpt-{step}.pt")
    torch.save({
        "net": net.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
    }, path)
    with open(cfg.LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump({"model_file": path, "step": step}, f, indent=2)
    return path

def load_latest(net, optimizer, cfg, device) -> int:
    if not os.path.exists(cfg.LATEST_JSON):
        return 0
    with open(cfg.LATEST_JSON, "r", encoding="utf-8") as f:
        info = json.load(f)
    if not os.path.exists(info.get("model_file", "")):
        return 0
    ck = torch.load(info["model_file"], map_location=device)
    net.load_state_dict(ck["net"])
    optimizer.load_state_dict(ck["optim"])
    return int(info["step"])

def main():
    parser = argparse.ArgumentParser(description="训练 Flappy Bird PPO（PyTorch 版）")
    parser.add_argument("--fresh", action="store_true", help="从零训练（忽略已有权重）")
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

    net = ActorCritic(cfg.NUM_ACTIONS).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.LR, eps=1e-5)
    buffer = RolloutBuffer(cfg.N_STEPS, state_shape=(cfg.STACK_LEN,
                                                     cfg.FRAME_SIZE, cfg.FRAME_SIZE))

    start_step = 0
    if not args.fresh:
        start_step = load_latest(net, optimizer, cfg, device)
        if start_step > 0:
            print(f"续训：已从 step={start_step} 恢复，继续训练")
        else:
            print("没找到存档 → 从零开始训练（随机初始化权重）")
    else:
        print("从零开始训练（--fresh，忽略已有存档）")

    do_nothing = np.zeros(cfg.NUM_ACTIONS)
    do_nothing[0] = 1
    frame, _r, _term = env.frame_step(do_nothing, render=render)
    state = init_state(frame)

    net.train()
    t = start_step
    update_idx = 0
    episode_score = 0
    next_save = ((start_step // cfg.SAVE_EVERY) + 1) * cfg.SAVE_EVERY
    recent_scores = deque(maxlen=100)

    print(f"\n配置：N_STEPS={cfg.N_STEPS} / BATCH={cfg.BATCH_SIZE} / "
          f"EPOCHS={cfg.N_EPOCHS} / LR={cfg.LR} / CLIP={cfg.CLIP_RANGE} / ENT={cfg.ENT_COEF}\n")

    while t < args.max_steps:

        progress = t / float(args.max_steps)
        lr = linear_schedule(cfg.LR, 0.0, progress) if cfg.LR_DECAY else cfg.LR
        for g in optimizer.param_groups:
            g["lr"] = lr

        state, episode_score, finished, rstats = collect_rollout(
            env, net, buffer, state, cfg, device, render, episode_score)
        t += cfg.N_STEPS
        recent_scores.extend(finished)

        with torch.no_grad():
            s_last = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = float(net.get_action_and_value(s_last)[3].item())

        buffer.compute_returns_and_advantages(
            last_value, cfg.GAMMA, cfg.GAE_LAMBDA,
            normalize=cfg.NORMALIZE_ADV and not cfg.NORM_ADV_PER_MINIBATCH)

        losses = update_policy(net, optimizer, buffer, cfg, device, cfg.ENT_COEF)
        update_idx += 1

        ma100 = float(np.mean(recent_scores)) if recent_scores else float("nan")
        if update_idx % cfg.LOG_EVERY_UPDATES == 0:
            print(f"步数 {t:>7} / 更新 {update_idx:>4} / lr {lr:.2e} "
                  f"/ 局数 {rstats['n_episodes']:>2} / 均分 {rstats['score_mean']:.2f} "
                  f"/ 近100局 {ma100:.2f} / r均值 {rstats['reward_mean']:.3f} "
                  f"/ 拍翅率 {rstats['flap_ratio']:.2f}")
            print(f"           loss {losses['total_loss']:.4f} "
                  f"(策略 {losses['policy_loss']:.4f} / 价值 {losses['value_loss']:.4f}) "
                  f"/ 熵 {losses['entropy']:.3f} / KL {losses['approx_kl']:.4f} "
                  f"/ 裁剪率 {losses['clip_fraction']:.3f} "
                  f"/ 解释方差 {losses['explained_variance']:.3f}"
                  + ("  [KL早停]" if losses["early_stopped"] else ""))

        if writer is not None:
            for sc in finished:
                writer.add_scalar("episode/score", sc, t)
            writer.add_scalar("episode/score_ma100", ma100, t)
            writer.add_scalar("rollout/reward_mean", rstats["reward_mean"], t)
            writer.add_scalar("rollout/flap_ratio", rstats["flap_ratio"], t)
            writer.add_scalar("loss/total", losses["total_loss"], t)
            writer.add_scalar("loss/policy", losses["policy_loss"], t)
            writer.add_scalar("loss/value", losses["value_loss"], t)
            writer.add_scalar("train/entropy", losses["entropy"], t)
            writer.add_scalar("train/approx_kl", losses["approx_kl"], t)
            writer.add_scalar("train/old_approx_kl", losses["old_approx_kl"], t)
            writer.add_scalar("train/clip_fraction", losses["clip_fraction"], t)
            writer.add_scalar("train/explained_variance", losses["explained_variance"], t)
            writer.add_scalar("train/learning_rate", lr, t)

        if t >= next_save:
            path = save_ckpt(t, net, optimizer, cfg)
            print(f"  → 已存档 {path}")
            next_save += cfg.SAVE_EVERY

    path = save_ckpt(t, net, optimizer, cfg)
    if writer is not None:
        writer.close()
    print(f"训练结束，最终模型已保存到 {path}")

if __name__ == "__main__":
    main()
