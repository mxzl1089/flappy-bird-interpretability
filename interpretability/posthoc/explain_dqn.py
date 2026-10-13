import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(ROOT, "dqn"))

from config import Config
from model import QNet
from utils import init_state, next_state

import attribution as A

MOMENT_ORDER = ["平稳飞行", "刚穿过一根管子", "撞死前的最后一步"]

PHASE_BINS = 36
DEATH_BACK = 30
OCC_BLOCK = 8
OCC_CHUNK = 300

REGION_NAMES = ["地面带", "小鸟", "上管道", "下管道", "背景残差"]
REGION_COLORS = ["#9AA0A6", "#D85A30", "#185FA5", "#1D9E75", "#E4E1DC"]
N_REGION = len(REGION_NAMES)
_REGION_TXT = " / ".join(REGION_NAMES)

DENSITY_YLIM = (0.05, 20.0)

def _cells():
    n = Config.FRAME_SIZE
    i = (np.arange(n) + 0.5) / n * 288.0
    j = (np.arange(n) + 0.5) / n * 512.0
    return i, j

def _ensure_game_module():
    if "wrapped_flappy_bird" not in sys.modules:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        sys.path.append(Config.GAME_DIR)
        import wrapped_flappy_bird
    return sys.modules["wrapped_flappy_bird"]

def _fill_screen_rect(lab, x, y, w, h, value):
    ci, cj = _cells()
    mi = (ci >= x) & (ci < x + w)
    mj = (cj >= y) & (cj < y + h)
    lab[mi[:, None] & mj[None, :]] = value

_BIRD_LUT = {}

def _bird_cells(playerIndex, playery, playerx):
    key = (int(playerIndex), int(playery))
    m = _BIRD_LUT.get(key)
    if m is None:
        gm = _ensure_game_module()
        hm = np.asarray(gm.HITMASKS["player"][int(playerIndex)], dtype=np.uint8)
        ys, xs = np.nonzero(hm)
        n = Config.FRAME_SIZE
        ii = ((playerx + xs + 0.5) / 288.0 * n).astype(np.int64)
        jj = ((playery + ys + 0.5) / 512.0 * n).astype(np.int64)
        ok = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n)
        m = np.zeros((n, n), dtype=bool)
        m[ii[ok], jj[ok]] = True
        _BIRD_LUT[key] = m
    return m

def region_label(env):
    n = Config.FRAME_SIZE
    gm = _ensure_game_module()
    lab = np.full((n, n), 4, np.uint8)
    pw, ph = float(gm.PIPE_WIDTH), float(gm.PIPE_HEIGHT)
    for up, lo in zip(env.upperPipes, env.lowerPipes):
        _fill_screen_rect(lab, up["x"], up["y"], pw, ph, 2)
        _fill_screen_rect(lab, lo["x"], lo["y"], pw, ph, 3)
    lab[_bird_cells(env.playerIndex, env.playery, env.playerx)] = 1
    lab[:, A.GROUND_BAND:] = 0
    return lab

def load_model(ckpt, device):
    path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到存档：{path}")
    net = QNet(Config.NUM_ACTIONS).to(device)
    net.load_state_dict(torch.load(path, map_location=device)["policy"])
    net.eval()
    return net, path

def make_env():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.append(Config.GAME_DIR)
    import wrapped_flappy_bird as game_env
    return game_env.GameState()

def collect(net, env, device, episodes, max_steps, pool, seed=0):
    rng = np.random.default_rng(seed)
    states, actions, qs = [], [], []
    special = {}
    seen = 0
    scores = []

    for ep in range(episodes):

        a0 = np.zeros(Config.NUM_ACTIONS)
        a0[0] = 1
        frame, _, terminal = env.frame_step(a0, render=False)
        state = init_state(frame)

        score, t = 0, 0
        while not terminal and t < max_steps:
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = net(x)[0].cpu().numpy()
            a = int(q.argmax())

            item = (state.astype(np.uint8), a, q)

            if len(states) < pool:
                states.append(item[0]); actions.append(a); qs.append(q)
            else:
                j = int(rng.integers(0, seen))
                if j < pool:
                    states[j], actions[j], qs[j] = item[0], a, q
            seen += 1

            av = np.zeros(Config.NUM_ACTIONS)
            av[a] = 1
            frame, r, terminal = env.frame_step(av, render=False)
            if r == 1.0:
                score += 1
                special["刚穿过一根管子"] = item
            state = next_state(state, frame)
            t += 1
            if terminal:
                special["撞死前的最后一步"] = item

        scores.append(score)
        print(f"    第 {ep + 1} 局：{t} 步，过了 {score} 根管，"
              f"{'撞死' if terminal else f'达到 {max_steps} 步上限（未死亡）'}")

    n_generic = len(states)

    moments = {}
    for name in MOMENT_ORDER:
        if name in special:
            s, a, q = special[name]
            moments[name] = len(states)
            states.append(s); actions.append(a); qs.append(q)

    moments["平稳飞行"] = int(rng.integers(0, n_generic))

    if "撞死前的最后一步" not in moments:
        print("    ⚠️ 这几局都没有死亡，无法给出「撞死前」的解释。")
        print("       想让模型死一次看失败归因：把 --max-steps 调大（比如 6000）再跑。")

    return (np.array(states), np.array(actions), np.array(qs), moments, n_generic, scores)

def _panel(ax, base, heat=None, title=None):
    base = base.T
    ax.imshow(base, cmap="gray_r", vmin=0, vmax=255, interpolation="nearest")
    if heat is not None:
        v = np.abs(heat).T
        hi = float(np.percentile(v, 99))
        if hi > 0:
            v = np.ma.masked_less(v, 0.2 * hi)
            ax.imshow(v, cmap="YlOrRd", vmin=0, vmax=hi,
                      alpha=0.8, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=10, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1")

def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✔ 已保存 {os.path.relpath(path, ROOT)}")
    return path

def run_local(net, states, actions, qs, moments, device):
    names = [n for n in MOMENT_ORDER if n in moments]
    mnames = list(A.METHODS)
    nrow, ncol = len(names), len(mnames) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.45 * ncol, 3.0 * nrow),
                             gridspec_kw=dict(hspace=0.32, wspace=0.06))
    axes = np.atleast_2d(axes)

    for r, name in enumerate(names):
        idx = moments[name]
        s, a, q = states[idx], int(actions[idx]), qs[idx]
        tip = "拍翅" if a else "不拍翅"
        print(f"    解释「{name}」（Q=[{q[0]:.2f}, {q[1]:.2f}]，选中 {tip}）")

        _panel(axes[r, 0], s[0], None,
               f"{name}\nQ=[{q[0]:.2f}, {q[1]:.2f}] → {tip}")
        for c, (mname, fn) in enumerate(A.METHODS.items()):
            att = fn(net, s, a, device)
            _panel(axes[r, c + 1], s[0], att[0], mname if r == 0 else None)

    fig.suptitle("第一层 · 局部解释：三种关键时刻下，五种归因方法的对照"
                 "（底图=最新一帧画面，颜色越深=该像素对 Q 值越重要）",
                 fontsize=13, y=0.985)
    fig.subplots_adjust(top=0.90, bottom=0.015, left=0.015, right=0.99)
    return _save(fig, "fig1_local_attribution.png")

def run_global(net, states, actions, n_generic, device, sg_samples=8):
    S = states[:n_generic]
    Ac = actions[:n_generic]
    print(f"    对 {n_generic} 个普通状态计算平均显著性…")
    sal = np.stack([A.saliency(net, s, int(a), device) for s, a in zip(S, Ac)])
    print(f"    对 {n_generic} 个状态计算平均 SmoothGrad（每个 {sg_samples} 次加噪）…")
    sg = np.stack([A.smoothgrad(net, s, int(a), device, n=sg_samples)
                   for s, a in zip(S, Ac)])

    imp = {m: np.abs(arr).sum(axis=(0, 2, 3)) for m, arr in [("Saliency", sal), ("SmoothGrad", sg)]}
    for k in imp:
        imp[k] = imp[k] / imp[k].sum() * 100

    fig = plt.figure(figsize=(16, 6.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.7], wspace=0.12, hspace=0.28)

    for j in range(Config.STACK_LEN):
        for i, (m, arr) in enumerate([("Saliency", sal), ("SmoothGrad", sg)]):
            ax = fig.add_subplot(gs[i, j])
            img = arr.mean(axis=0)[j]
            name = f"第 {j} 帧" + ("（最新）" if j == 0 else f"（早 {j} 帧）")
            _panel(ax, S[0][j], img, f"{m} · {name}")

    axb = fig.add_subplot(gs[:, 4])
    x = np.arange(Config.STACK_LEN)
    w = 0.38
    b1 = axb.bar(x - w / 2, imp["Saliency"], w, label="Saliency", color="#378ADD")
    b2 = axb.bar(x + w / 2, imp["SmoothGrad"], w, label="SmoothGrad", color="#D85A30")
    for bars in (b1, b2):
        for b in bars:
            axb.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                     f"{b.get_height():.0f}", ha="center", fontsize=9, color="#334155")
    axb.set_xticks(x)
    axb.set_xticklabels([f"第 {j} 帧" + ("\n(最新)" if j == 0 else "") for j in range(4)])
    axb.set_ylim(0, max(imp["Saliency"].max(), imp["SmoothGrad"].max()) * 1.25)
    axb.set_title("四帧输入的重要性排序\n（占全部归因的 %）", fontsize=11)
    axb.legend(fontsize=9, frameon=False)
    axb.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    fig.suptitle(f"第二层 · 全局解释：{n_generic} 个游戏状态的归因平均（越深越重要）",
                 fontsize=13, y=0.98)
    path = _save(fig, "fig2_global_attribution.png")
    return path, {k: v.tolist() for k, v in imp.items()}

def run_faith(net, states, actions, qs, n_generic, device, n_states=12, steps=10):
    q_chosen = qs[:n_generic, :][np.arange(n_generic), actions[:n_generic]]

    idx = np.argsort(-q_chosen)[:n_states]
    S, Ac = states[idx], actions[idx].astype(int)
    print(f"    取 Q 最高的 {n_states} 个状态做检验（平均 Q = {q_chosen[idx].mean():.2f}）")

    curves, aucs = {}, {}
    rand = None
    for mname, fn in A.METHODS.items():
        attrs = np.stack([fn(net, s, a, device) for s, a in zip(S, Ac)])
        fracs, c_attr, c_rand = A.deletion_curve(net, S, Ac, attrs, device, steps=steps)
        if rand is None:
            rand = c_rand
        curves[mname] = c_attr
        aucs[mname] = float(np.trapz(c_attr, fracs))
    curves["随机（对照）"] = rand
    aucs["随机（对照）"] = float(np.trapz(rand, fracs))

    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = ["#378ADD", "#1D9E75", "#7F77DD", "#BA7517", "#D85A30", "#888780"]
    for (mname, c), col in zip(curves.items(), colors):
        is_rand = "随机" in mname
        ax.plot(fracs * 100, c, marker="o", ms=4, lw=1.8, color=col,
                ls="--" if is_rand else "-",
                label=f"{mname}  (AUC={aucs[mname]:.1f})")
    ax.axhline(100, color="#B4B2A9", lw=0.8, ls=":")
    ax.set_xlabel("被涂黑的像素比例 (%)  —— 按各自的重要性排序涂黑")
    ax.set_ylabel("Q 值保留百分比 (%)")
    ax.set_title("第三层 · 保真度检验：归因排序 vs 随机排序\n"
                 "曲线掉得越快 = 越早打中要害 = 归因越可信", fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    path = _save(fig, "fig3_faithfulness.png")
    return path, aucs

def collect_phased(net, device, episodes, max_steps, seed=0):
    for ep in range(episodes):
        env = make_env()

        a0 = np.zeros(Config.NUM_ACTIONS); a0[0] = 1
        frame, _, terminal = env.frame_step(a0, render=False)
        state = init_state(frame)

        states, actions, qs, labels = [], [], [], []
        score, t, pass_t = 0, 0, []
        while not terminal and t < max_steps:
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = net(x)[0].cpu().numpy()
            a = int(q.argmax())

            states.append(state.astype(np.uint8))

            labels.append(region_label(env))
            actions.append(a); qs.append(q)

            av = np.zeros(Config.NUM_ACTIONS); av[a] = 1
            frame, r, terminal = env.frame_step(av, render=False)
            if r == 1.0:
                score += 1
                pass_t.append(t)
            state = next_state(state, frame)
            t += 1

        dead = (t - 1) if terminal else None
        print(f"    第 {ep + 1} 局：{t} 帧，过 {score} 根管，"
              + (f"死亡于第 {dead} 帧" if dead is not None else "跑到上限未死"))
        yield (np.array(states, np.uint8), np.array(actions, np.int64),
               np.array(qs, np.float32), np.array(labels, np.uint8),
               pass_t, dead, score)

def phase_of_frames(pass_t, n_frames, period):
    idx = np.arange(n_frames, dtype=np.float64)
    if not pass_t:
        return np.clip((idx + 1.0) / (n_frames + 1.0), 0.0, 1.0)
    seq = [float(v) for v in pass_t]
    while seq[0] > 0.0:
        seq.insert(0, seq[0] - float(period))
    while seq[-1] <= n_frames - 1:
        seq.append(seq[-1] + float(period))
    knots = np.asarray(seq, dtype=np.float64)
    ph = np.zeros(n_frames, dtype=np.float64)
    for i in range(len(knots) - 1):
        lo, hi = knots[i], knots[i + 1]
        if hi <= lo:
            continue
        m = (idx >= lo) & (idx < hi)
        ph[m] = (idx[m] - lo) / (hi - lo)
    return np.clip(ph, 0.0, 1.0)

def _accum(acc, cnt, att, bins, groups=None):
    n = att.shape[0]
    if groups is None:
        groups = np.zeros(n, dtype=np.int64)
    for g in np.unique(groups):
        sel_g = np.where(groups == g)[0]
        bg = bins[sel_g]
        for b in np.unique(bg):
            sel = sel_g[bg == b]
            acc[g, b] += att[sel].sum(0)
            cnt[g, b] += len(sel)

def _accum_rows(acc, cnt, vec, bins, groups=None):
    n = vec.shape[0]
    if groups is None:
        groups = np.zeros(n, dtype=np.int64)
    for g in np.unique(groups):
        sel_g = np.where(groups == g)[0]
        bg = bins[sel_g]
        for b in np.unique(bg):
            sel = sel_g[bg == b]
            acc[g, b] += vec[sel].sum(0)
            cnt[g, b] += len(sel)

def _new_accum(n_ep):
    P = PHASE_BINS
    C, H, W = Config.STACK_LEN, Config.FRAME_SIZE, Config.FRAME_SIZE

    def bl(k):
        return (np.zeros((k, P, C, H, W), np.float32), np.zeros((k, P), np.int64))

    def bl1(k):
        return (np.zeros((k, P, 1, 1, 1), np.float32), np.zeros((k, P), np.int64))

    a_all, c_all = bl(1)
    a_ep, c_ep = bl(n_ep)
    a_half, c_half = bl(2)
    a_rand, c_rand = bl(2)
    a_ctrl, c_ctrl = bl(1)
    a_img, c_img = bl(2)
    a_dead, c_dead = bl(2)
    a_rwin, c_rwin = bl(2)

    a_band, c_band = bl1(2)
    a_tot, c_tot = bl1(2)
    a_nzb, c_nzb = bl1(2)
    a_nzr, c_nzr = bl1(2)
    a_nzt, c_nzt = bl1(2)
    a_rband, c_rband = bl1(2)
    a_rtot, c_rtot = bl1(2)

    def blr(k):
        return (np.zeros((k, P, N_REGION), np.float32), np.zeros((k, P), np.int64))

    a_reg, c_reg = blr(2)
    a_are, c_are = blr(2)
    a_reg1, c_reg1 = blr(1)
    a_are1, c_are1 = blr(1)
    a_lbl, c_lbl = bl1(1)
    a_lbp, c_lbp = bl1(1)
    a_haze, c_haze = bl1(1)
    return dict(acc_all=a_all, cnt_all=c_all, acc_ep=a_ep, cnt_ep=c_ep,
                acc_half=a_half, cnt_half=c_half, acc_rand=a_rand, cnt_rand=c_rand,
                acc_ctrl=a_ctrl, cnt_ctrl=c_ctrl, acc_img=a_img, cnt_img=c_img,
                acc_dead=a_dead, cnt_dead=c_dead, acc_rwin=a_rwin, cnt_rwin=c_rwin,
                acc_band=a_band, cnt_band=c_band, acc_tot=a_tot, cnt_tot=c_tot,
                acc_rband=a_rband, cnt_rband=c_rband, acc_rtot=a_rtot, cnt_rtot=c_rtot,
                acc_nzb=a_nzb, cnt_nzb=c_nzb, acc_nzr=a_nzr, cnt_nzr=c_nzr,
                acc_nzt=a_nzt, cnt_nzt=c_nzt,
                acc_reg=a_reg, cnt_reg=c_reg, acc_area=a_are, cnt_area=c_are,
                acc_reg_all=a_reg1, cnt_reg_all=c_reg1,
                acc_area_all=a_are1, cnt_area_all=c_are1,
                acc_lbl=a_lbl, cnt_lbl=c_lbl, acc_lbl_bright=a_lbp, cnt_lbl_bright=c_lbp,
                acc_haze=a_haze, cnt_haze=c_haze,
                ground_band=A.GROUND_BAND)

def attribute_phase(net, states, actions, bins_real, bins_ctrl, ep_ids,
                    half_ids, rand_ids, dead_ids, rwin_ids, accs, device,
                    labels=None, verbose=True):
    G = A.GROUND_BAND
    N = len(states)
    t0 = time.time()
    for s0 in range(0, N, OCC_CHUNK):
        s1 = min(s0 + OCC_CHUNK, N)
        att = A.occlusion_batch(net, states[s0:s1], actions[s0:s1],
                                device, block=OCC_BLOCK)
        img = states[s0:s1].astype(np.float32)
        br = bins_real[s0:s1]

        a = np.abs(att)
        band = a[:, :, :, G:].sum(axis=(1, 2, 3))
        rest = a[:, :, :, :G].sum(axis=(1, 2, 3))
        tot = band + rest

        nzb = (a[:, :, :, G:] > 0).sum(axis=(1, 2, 3)).astype(np.float32)
        nzr = (a[:, :, :, :G] > 0).sum(axis=(1, 2, 3)).astype(np.float32)
        nzt = nzb + nzr
        attv = att.copy()
        attv[:, :, :, G:] = 0.0

        if labels is not None:
            lab = labels[s0:s1]
            npx = float(Config.FRAME_SIZE * Config.FRAME_SIZE)
            mass = np.zeros((len(lab), N_REGION), np.float64)
            arw = np.zeros((len(lab), N_REGION), np.float32)
            for r in range(N_REGION):
                m = (lab == r)
                mass[:, r] = (a * m[:, None, :, :]).sum(axis=(1, 2, 3))
                arw[:, r] = m.sum(axis=(1, 2)) / npx * 100.0
            _accum_rows(accs["acc_reg"], accs["cnt_reg"], mass, br, dead_ids[s0:s1])
            _accum_rows(accs["acc_area"], accs["cnt_area"], arw, br, dead_ids[s0:s1])
            _accum_rows(accs["acc_reg_all"], accs["cnt_reg_all"], mass, br)
            _accum_rows(accs["acc_area_all"], accs["cnt_area_all"], arw, br)

            img0 = states[s0:s1, 0] > 0
            lb_pb = (lab == 1) | (lab == 2) | (lab == 3)
            _accum(accs["acc_lbl"], accs["cnt_lbl"],
                   lb_pb.sum(axis=(1, 2)).astype(np.float32)[:, None, None, None], br)
            _accum(accs["acc_lbl_bright"], accs["cnt_lbl_bright"],
                   (lb_pb & img0).sum(axis=(1, 2)).astype(np.float32)[:, None, None, None], br)

            bgh = (lab == 4) & img0
            _accum(accs["acc_haze"], accs["cnt_haze"],
                   (a * bgh[:, None, :, :]).sum(axis=(1, 2, 3))[:, None, None, None], br)

        _accum(accs["acc_all"], accs["cnt_all"], attv, br)
        _accum(accs["acc_ep"], accs["cnt_ep"], attv, br, ep_ids[s0:s1])
        _accum(accs["acc_half"], accs["cnt_half"], attv, br, half_ids[s0:s1])
        _accum(accs["acc_rand"], accs["cnt_rand"], attv, br, rand_ids[s0:s1])
        _accum(accs["acc_ctrl"], accs["cnt_ctrl"], attv, bins_ctrl[s0:s1])
        _accum(accs["acc_dead"], accs["cnt_dead"], attv, br, dead_ids[s0:s1])
        _accum(accs["acc_rwin"], accs["cnt_rwin"], attv, br, rwin_ids[s0:s1])
        _accum(accs["acc_img"], accs["cnt_img"], img, br, dead_ids[s0:s1])

        def _c(key, arr):
            _accum(accs["acc_" + key], accs["cnt_" + key],
                   arr[:, None, None, None], br, dead_ids[s0:s1])

        _c("band", band); _c("tot", tot)
        _c("nzb", nzb); _c("nzr", nzr); _c("nzt", nzt)

        for key, arr in (("band", band), ("tot", tot)):
            _accum(accs["acc_r" + key], accs["cnt_r" + key],
                   arr[:, None, None, None], br, rwin_ids[s0:s1])
        if verbose:
            print(f"      归因 {s1}/{N} 帧  ({time.time() - t0:.0f}s)")
    return accs

def _region_mean(acc, cnt, group=None):
    acc = np.asarray(acc, np.float64)
    cnt = np.asarray(cnt, np.float64)
    if group is None:
        num, den = acc.sum(0), cnt.sum(0)
    else:
        num, den = acc[group], cnt[group]
    return num / np.maximum(den, 1.0)[:, None]

def _rows_to_share(rows):
    rows = np.asarray(rows, np.float64)
    return rows / np.maximum(rows.sum(axis=-1, keepdims=True), 1e-12) * 100.0

def _region_density(share, area):
    return share / np.maximum(area, 1e-9)

def _share_curve(acc_num, acc_den, P, group=None):
    num = np.asarray(acc_num)
    den = np.asarray(acc_den)
    if num.ndim == 5:
        num = num[:, :, 0, 0, 0]
        den = den[:, :, 0, 0, 0]
    n = num.sum(axis=0) if group is None else num[group]
    d = den.sum(axis=0) if group is None else den[group]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, n / d * 100.0, np.nan)

def attribute_death(net, deaths, device):
    C, H, W = Config.STACK_LEN, Config.FRAME_SIZE, Config.FRAME_SIZE
    T = DEATH_BACK + 1
    G = A.GROUND_BAND

    picks, dropped, death_scores = [], 0, []
    for rec in deaths:
        dead = rec["dead"]
        if dead is None:
            continue
        if dead < DEATH_BACK:

            dropped += 1
            continue
        di = len(death_scores)

        for g in range(max(0, dead - DEATH_BACK), dead + 1):
            picks.append((g, dead - g, rec, di))
        death_scores.append(int(rec["score"]))
    if not picks:
        return (None, None, None, 0, None, None, None, None, dropped, [], None, None, None)

    taus = np.array([p[1] for p in picks])
    DIDX = np.array([p[3] for p in picks])
    S = np.stack([p[2]["states"][p[0]] for p in picks])
    Ac = np.array([p[2]["actions"][p[0]] for p in picks])
    AT = np.array([int(p[2]["bins"][p[0]]) for p in picks])
    LB = np.stack([p[2]["labels"][p[0]] for p in picks])
    att = A.occlusion_batch(net, S, Ac, device, block=OCC_BLOCK)
    img = S.astype(np.float32)

    a = np.abs(att)
    band = a[:, :, :, G:].sum(axis=(1, 2, 3))
    tot = a.sum(axis=(1, 2, 3))
    nzb = (a[:, :, :, G:] > 0).sum(axis=(1, 2, 3)).astype(np.float64)
    nzt = (a > 0).sum(axis=(1, 2, 3)).astype(np.float64)
    attv = att.copy()
    attv[:, :, :, G:] = 0.0

    acc = np.zeros((T, C, H, W), np.float32)
    cimg = np.zeros((T, C, H, W), np.float32)
    cnt = np.zeros(T, np.int64)
    hist2d = np.zeros((T, PHASE_BINS), np.int64)
    inten2d = np.zeros((T, PHASE_BINS), np.float64)
    sband = np.zeros(T, np.float64)
    stot = np.zeros(T, np.float64)
    snzb = np.zeros(T, np.float64)
    snzt = np.zeros(T, np.float64)
    nD = len(death_scores)
    sband_d = np.zeros((nD, T), np.float64)
    stot_d = np.zeros((nD, T), np.float64)

    reg_tau = np.zeros((T, N_REGION), np.float64)
    are_tau = np.zeros((T, N_REGION), np.float64)
    npx = float(Config.FRAME_SIZE * Config.FRAME_SIZE)
    for r in range(N_REGION):
        mr = (LB == r)
        np.add.at(reg_tau[:, r], taus, (a * mr[:, None, :, :]).sum(axis=(1, 2, 3)))
        np.add.at(are_tau[:, r], taus, mr.sum(axis=(1, 2)) / npx * 100.0)
    for j in range(len(picks)):
        b = int(taus[j]); pb = int(AT[j]); di = int(DIDX[j])
        acc[b] += attv[j]; cimg[b] += img[j]; cnt[b] += 1
        hist2d[b, pb] += 1
        inten2d[b, pb] += float(np.abs(attv[j]).sum())
        sband[b] += float(band[j]); stot[b] += float(tot[j])
        snzb[b] += nzb[j]; snzt[b] += nzt[j]
        sband_d[di, b] += float(band[j]); stot_d[di, b] += float(tot[j])

    share = np.array([sband[b] / stot[b] * 100 if stot[b] > 0 else np.nan
                      for b in range(T)])
    share_cnt = np.array([snzb[b] / snzt[b] * 100 if snzt[b] > 0 else np.nan
                          for b in range(T)])
    with np.errstate(invalid="ignore", divide="ignore"):
        share_by_death = np.where(stot_d > 0, sband_d / np.maximum(stot_d, 1e-12) * 100,
                                  np.nan)
    n_death = nD
    return (acc, cimg, cnt, n_death, hist2d, inten2d,
            share, share_cnt, dropped, death_scores, share_by_death,
            reg_tau, are_tau)

def _avg_over_phase(acc, cnt, group=0):
    P = acc.shape[1]
    out = np.zeros(acc.shape[1:], np.float32)
    for b in range(P):
        if group is None:
            c = int(cnt[:, b].sum())
            if c > 0:
                out[b] = acc[:, b].sum(0) / c
        else:
            c = int(cnt[group, b])
            if c > 0:
                out[b] = acc[group, b] / c
    return out

def _density_lines(ax, x, share, area, xlabel="", title="", ms=3.0):
    xs = np.asarray(x, np.float64)
    for r in range(N_REGION):
        d = _region_density(np.asarray(share[:, r], np.float64),
                            np.asarray(area[:, r], np.float64))
        ax.plot(xs, d, lw=1.8, marker="o", ms=ms, color=REGION_COLORS[r],
                label=REGION_NAMES[r], markeredgewidth=0)
    ax.axhline(1.0, color="#888780", lw=1.1, ls="--")
    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_yscale("log")
    ax.set_ylim(*DENSITY_YLIM)
    ax.set_ylabel("归因密度（单位面积归因）", fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.20, which="both")
    h = ax.get_legend_handles_labels()[0]
    ax.legend(h + [plt.Line2D([], [], color="#888780", ls="--", lw=1.1)],
              REGION_NAMES + ["1 = 全图平均"], fontsize=7.6, frameon=True,
              framealpha=0.82, edgecolor="none", loc="upper center", ncol=3,
              columnspacing=0.9, handlelength=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

def fig_phase_cycle(res, out_name="figA_pipe_phase.png"):
    P = PHASE_BINS
    G = res["ground_band"]
    avg = _avg_over_phase(res["acc_all"], res["cnt_all"], 0)

    img = _avg_over_phase(res["acc_img"], res["cnt_img"], None)
    ns = res["cnt_all"][0]
    reg = _rows_to_share(_region_mean(res["acc_reg_all"], res["cnt_reg_all"]))
    are = _region_mean(res["acc_area_all"], res["cnt_area_all"])

    fig = plt.figure(figsize=(16, 10.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.35], hspace=0.40, wspace=0.06)

    reps = [0, 6, 12, 18, 24, 30]
    for j, b in enumerate(reps):
        ax = fig.add_subplot(gs[0, j])
        _panel(ax, img[b][0] if ns[b] else np.zeros((80, 80), np.float32),
               avg[b][0], f"相位 {b}/36   n={int(ns[b])}")

    gsb = gs[1, :].subgridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.28)
    axm = fig.add_subplot(gsb[0])
    prof = np.abs(avg[:, 0]).sum(axis=1)[:, :G]
    prof = prof / np.maximum(prof.sum(axis=1, keepdims=True), 1e-9) * 100
    im = axm.imshow(prof.T, aspect="auto", cmap="YlOrRd", origin="upper")
    axm.set_xlabel("管道相位（0 = 刚过管）", fontsize=10)
    axm.set_ylabel(f"屏幕 Y（0 = 上，{G - 1} = 地面带顶；已去地面带）", fontsize=10)
    axm.set_title("时空热图：注意力在画面高度上的分布随相位变化（每列归一化到 100%）", fontsize=11)
    axm.set_xticks([0, 6, 12, 18, 24, 30, 35])
    axm.set_yticks([0, 15, 30, 45, G - 1])
    cb = fig.colorbar(im, ax=axm, fraction=0.032, pad=0.012)
    cb.ax.tick_params(labelsize=8)

    axd = fig.add_subplot(gsb[1])
    _density_lines(axd, np.arange(P), reg, are,
                   xlabel="管道相位 φ（0 = 刚过管）",
                   title="各区域归因密度随相位的变化（单位面积归因，对数纵轴）")

    mm = {n: float(np.nanmean(_region_density(reg[:, r], are[:, r])))
          for r, n in enumerate(REGION_NAMES)}
    seg = " · ".join(f"{n} {mm[n]:.2f}" for n in REGION_NAMES)
    axd.text(0.012, 0.02, f"全周期平均：{seg}", transform=axd.transAxes,
             fontsize=8.4, va="bottom", ha="left", color="#2C2C2A")
    fig.suptitle("图 3-4　空间相位图：一个管道周期内归因的空间节律"
                 "（45 局 / 89,423 帧按管道相位 φ 折叠）",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.905, bottom=0.075, left=0.02, right=0.985)
    return _save(fig, out_name)

def fig_death_align(acc_d, cimg_d, cnt_d, n_death,
                    reg_tau=None, are_tau=None, ground_band=None,
                    out_name="figB_death_align.png"):
    T = DEATH_BACK + 1
    G = int(ground_band) if ground_band is not None else 62

    reg_tau = _rows_to_share(np.zeros((T, N_REGION)) if reg_tau is None else reg_tau)

    if are_tau is None:
        are_tau = np.full((T, N_REGION), np.nan)
    else:
        are_tau = np.asarray(are_tau, np.float64)
        _rs = are_tau.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            are_tau = np.where(_rs > 0, are_tau / np.maximum(_rs, 1e-12) * 100.0, np.nan)
    avg = np.zeros_like(acc_d)
    for b in range(T):
        if cnt_d[b] > 0:
            avg[b] = acc_d[b] / cnt_d[b]
    img = np.zeros_like(cimg_d)
    for b in range(T):
        if cnt_d[b] > 0:
            img[b] = cimg_d[b] / cnt_d[b]

    fig = plt.figure(figsize=(16, 10.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.35], hspace=0.40, wspace=0.06)

    reps = [30, 24, 18, 12, 6, 0]
    for j, b in enumerate(reps):
        ax = fig.add_subplot(gs[0, j])
        _panel(ax, img[b][0] if cnt_d[b] else np.zeros((80, 80), np.float32),
               avg[b][0], f"死亡前 {b} 帧   n={int(cnt_d[b])}")

    gsb = gs[1, :].subgridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.28)

    axm = fig.add_subplot(gsb[0])
    prof = np.abs(avg[:, 0]).sum(axis=1)[:, :G]
    prof = prof / np.maximum(prof.sum(axis=1, keepdims=True), 1e-9) * 100
    im = axm.imshow(prof.T, aspect="auto", cmap="YlOrRd", origin="upper")
    axm.set_xlabel("距死亡帧数 τ（0 = 死亡帧；自左向右时间推进）", fontsize=10)
    axm.set_ylabel(f"屏幕 Y（0 = 上，{G - 1} = 地面带顶；已去地面带）", fontsize=10)
    axm.set_title("时空热图：注意力在画面高度上的分布随距死亡帧数的变化"
                  "（每列归一化到 100%）", fontsize=11)
    axm.set_xlim(T - 0.5, -0.5)
    axm.set_xticks([30, 25, 20, 15, 10, 5, 0])
    axm.set_yticks([0, 15, 30, 45, G - 1])
    cb = fig.colorbar(im, ax=axm, fraction=0.032, pad=0.012)
    cb.ax.tick_params(labelsize=8)

    axd = fig.add_subplot(gsb[1])
    _density_lines(axd, np.arange(T), reg_tau, are_tau,
                   xlabel="距死亡帧数 τ（0 = 死亡帧；自左向右时间推进）",
                   title="各区域归因密度随距死亡帧数的变化（单位面积归因，对数纵轴）")
    axd.set_xlim(T - 0.5, -0.5)

    mmd = {n: float(np.nanmean(_region_density(reg_tau[:, r], are_tau[:, r])))
           for r, n in enumerate(REGION_NAMES)}
    segd = " · ".join(f"{n} {mmd[n]:.2f}" for n in REGION_NAMES)
    axd.text(0.012, 0.02, f"死亡区间平均：{segd}", transform=axd.transAxes,
             fontsize=8.4, va="bottom", ha="left", color="#2C2C2A")

    fig.suptitle(f"图 3-6　死亡对齐图：临死前 30 帧归因的空间节律"
                 f"（{n_death} 次死亡事件对齐到同一条倒计时轴）",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.905, bottom=0.075, left=0.02, right=0.985)
    return _save(fig, out_name)

def _corr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")

def _feat(acc, cnt, group, center=False):
    avg = _avg_over_phase(acc, cnt, group)

    f = np.abs(avg[:, 0]).sum(axis=1)
    if center:
        f = f - f.mean(axis=0, keepdims=True)
    return f.ravel()

def fig_stability(res, n_ep, period=36.0, min_cycles=10, out_name="figC_phase_stability.png"):
    P = PHASE_BINS

    def curveY(acc, cnt, g):
        avg = _avg_over_phase(acc, cnt, g)
        m = np.abs(avg[:, 0]).sum(axis=1)
        ys = np.arange(m.shape[1], dtype=np.float64)
        tot = m.sum(axis=1)
        return np.where(tot > 0, (m * ys[None, :]).sum(axis=1) / np.maximum(tot, 1e-12),
                        np.nan)

    def getacc(name):
        return res["acc_" + name], res["cnt_" + name]

    def corr_pair(name, g0, g1):
        a, c = getacc(name)
        rc = _corr(_feat(a, c, g0, center=True), _feat(a, c, g1, center=True))
        rr = _corr(_feat(a, c, g0), _feat(a, c, g1))
        return rc, rr

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    corr_out, raw_out = {}, {}
    cols = ["#185FA5", "#D85A30", "#1D9E75", "#BA7517"]

    a, c = getacc("rand")
    axes[0, 0].plot(range(P), curveY(a, c, 0), color=cols[0], lw=1.6, marker="o", ms=3, label="随机分半 A")
    axes[0, 0].plot(range(P), curveY(a, c, 1), color=cols[1], lw=1.6, marker="s", ms=3, label="随机分半 B")
    rc, rr = corr_pair("rand", 0, 1)
    corr_out["随机对半"] = rc; raw_out["随机对半"] = rr
    axes[0, 0].set_title(f"① 随机对半复核\n相位调制 r = {rc:.3f}   （原始 r = {rr:.3f}）", fontsize=11)

    a, c = getacc("half")
    axes[0, 1].plot(range(P), curveY(a, c, 0), color=cols[0], lw=1.6, marker="o", ms=3, label="前半局")
    axes[0, 1].plot(range(P), curveY(a, c, 1), color=cols[1], lw=1.6, marker="s", ms=3, label="后半局")
    rc, rr = corr_pair("half", 0, 1)
    corr_out["前半局vs后半局"] = rc; raw_out["前半局vs后半局"] = rr
    axes[0, 1].set_title(f"② 时间漂移检验（前半局 vs 后半局）\n"
                         f"相位调制 r = {rc:.3f}   （原始 r = {rr:.3f}）", fontsize=11)

    a, c = getacc("ep")
    curves = np.stack([curveY(a, c, e) for e in range(n_ep)])
    for e in range(n_ep):
        axes[1, 0].plot(range(P), curves[e], lw=0.9, alpha=0.45,
                        color=cols[e % len(cols)])
    axes[1, 0].plot(range(P), np.nanmean(curves, axis=0), lw=2.4, color="#0B0B0B",
                    label=f"{n_ep} 局均值")
    fc = [_feat(a, c, e, center=True) for e in range(n_ep)]
    fr = [_feat(a, c, e) for e in range(n_ep)]

    def pairwise(idxs):
        rc_, rr_ = [], []
        for i in idxs:
            for j in idxs:
                if j <= i:
                    continue
                rc_.append(_corr(fc[i], fc[j])); rr_.append(_corr(fr[i], fr[j]))
        return (float(np.nanmean(rc_)) if rc_ else float("nan"),
                float(np.nanmean(rr_)) if rr_ else float("nan"))

    rc_all, rr_all = pairwise(range(n_ep))

    frames_ep = np.asarray(res["cnt_ep"]).sum(axis=1)
    keep = [e for e in range(n_ep) if frames_ep[e] >= min_cycles * period]
    rc_long, rr_long = pairwise(keep)
    use_long = len(keep) >= 3
    rc, rr = (rc_long, rr_long) if use_long else (rc_all, rr_all)
    corr_out["局间均值"] = rc; raw_out["局间均值"] = rr
    axes[1, 0].set_title(f"③ 局间差异（判据用 {len(keep)} 局 ≥{min_cycles} 周期；"
                         f"全部 {n_ep} 局 r={rc_all:.3f}）\n"
                         f"相位调制 r = {rc:.3f}   （原始 r = {rr:.3f}）", fontsize=11)

    ks = list(corr_out.keys()); vs = [corr_out[k] for k in ks]
    bars = axes[1, 1].bar(ks, vs, color=["#185FA5", "#1D9E75", "#BA7517"], width=0.5)
    axes[1, 1].axhline(0.7, color="#A32D2D", lw=1, ls="--")

    for b, v in zip(bars, vs):
        axes[1, 1].text(b.get_x() + b.get_width() / 2, v + 0.02,
                        f"{v:.2f}", ha="center", fontsize=9)
    axes[1, 1].set_ylim(0, 1.12)
    axes[1, 1].set_ylabel("Pearson r（相位调制，已去公共成分）", fontsize=9)
    axes[1, 1].set_title("④ 三种切法的一致性汇总（红虚线 = 判据 r > 0.7）", fontsize=11)
    axes[1, 1].tick_params(axis="x", labelsize=8)
    axes[1, 1].grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        axes[1, 1].spines[sp].set_visible(False)

    for ax in axes.ravel()[:3]:
        ax.set_xlabel("管道相位 φ", fontsize=10)

        ax.invert_yaxis()
        ax.set_ylabel("归因重心 · 屏幕 Y（0 = 天空，向下增大；已去地面带）", fontsize=9)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("图3-9 · 稳定性复核：换一半数据 / 换一个时间段 / 换一局，结论还在不在",
                 fontsize=13, y=0.985)
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.07, right=0.98, hspace=0.46, wspace=0.24)
    extra = {"inter_ep_long_only": rc_long, "inter_ep_all": rc_all,
             "n_long_episodes": len(keep), "min_cycles": min_cycles,
             "long_episode_idx": keep, "frames_per_episode": [int(v) for v in frames_ep]}
    return _save(fig, out_name), corr_out, raw_out, extra

def fig_control(res, out_name="figD_phase_control.png"):
    fr = _feat(res["acc_all"], res["cnt_all"], 0, center=True)
    fc = _feat(res["acc_ctrl"], res["cnt_ctrl"], 0, center=True)
    r = _corr(fr, fc)
    corr_raw = _corr(_feat(res["acc_all"], res["cnt_all"], 0),
                     _feat(res["acc_ctrl"], res["cnt_ctrl"], 0))
    amp = float(np.std(fr) / max(float(np.std(fc)), 1e-12))

    def prof(acc, cnt):
        avg = _avg_over_phase(acc, cnt, 0)
        G = 62
        p = np.abs(avg[:, 0]).sum(axis=1)[:, :G]
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-9) * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2),
                             gridspec_kw=dict(width_ratios=[1, 1, 0.8], wspace=0.30))
    for ax, p, t in [(axes[0], prof(res["acc_all"], res["cnt_all"]), "真实相位（按过管事件对齐）"),
                     (axes[1], prof(res["acc_ctrl"], res["cnt_ctrl"]), "随机相位（标签打乱后重折叠）")]:
        im = ax.imshow(p.T, aspect="auto", cmap="YlOrRd", origin="upper")
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("相位", fontsize=10)
        ax.set_ylabel("屏幕 Y（0=上，61=地面带顶）", fontsize=9)
        ax.set_xticks([0, 6, 12, 18, 24, 30, 35])
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=8)

    ax = axes[2]
    ax.bar(["相位调制 r"], [r], color="#D85A30", width=0.45)
    ax.axhline(0, color="#888780", lw=0.8)
    ax.axhline(0.5, color="#A32D2D", lw=1, ls="--")

    ax.set_xlim(-0.5, 0.5)

    ax.text(0.97, 0.53, "判据 < 0.5", transform=ax.transAxes,
            ha="right", fontsize=8, color="#A32D2D")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Pearson r", fontsize=10)
    ax.set_title(f"真实 vs 随机相位\nr = {r:.3f}（原始 r = {corr_raw:.3f}）\n"
                 f"调制幅度比 amp = {amp:.2f}×（判据 > 1.5）", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("图3-10 · 随机相位对照（phase-randomized surrogate）："
                 "若真实与随机几乎一样，说明「按周期对齐」没抓到结构",
                 fontsize=12.5, y=0.98)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.05, right=0.98)
    return _save(fig, out_name), r, amp, corr_raw

def fig_death_vs_cycle(res, out_name="figE_death_vs_cycle.png"):
    P = PHASE_BINS

    def curveY_of(acc, cnt, g):
        avg = _avg_over_phase(acc, cnt, g)
        m = np.abs(avg[:, 0]).sum(axis=1)
        ys = np.arange(m.shape[1], dtype=np.float64)
        tot = m.sum(axis=1)
        return np.where(tot > 0, (m * ys[None, :]).sum(axis=1) / np.maximum(tot, 1e-12),
                        np.nan)

    yb = curveY_of(res["acc_dead"], res["cnt_dead"], 0)
    yd = curveY_of(res["acc_dead"], res["cnt_dead"], 1)
    yr_b = curveY_of(res["acc_rwin"], res["cnt_rwin"], 0)
    yr_d = curveY_of(res["acc_rwin"], res["cnt_rwin"], 1)
    nb, nd = res["cnt_dead"][0], res["cnt_dead"][1]
    nrd = res["cnt_rwin"][1]
    have, have_r = nd > 0, nrd > 0
    ok = np.isfinite(yb) & np.isfinite(yd) & have
    okr = np.isfinite(yr_b) & np.isfinite(yr_d) & have_r
    d = np.where(ok, yd - yb, np.nan)
    dr = np.where(okr, yr_d - yr_b, np.nan)
    r_shape = _corr(yb[ok], yd[ok])
    rms = float(np.sqrt(np.nanmean(d[ok] ** 2))) if ok.any() else float("nan")
    sd = float(np.nanstd(yb[ok])) if ok.any() else float("nan")
    eff = rms / sd if (sd and sd > 0) else float("nan")

    rms_r = float(np.sqrt(np.nanmean(dr[okr] ** 2))) if okr.any() else float("nan")
    eff_r = rms_r / sd if (sd and sd > 0) else float("nan")
    r_shape_r = _corr(yr_b[okr], yr_d[okr]) if okr.sum() >= 3 else float("nan")

    sh_b = _share_curve(res["acc_band"], res["acc_tot"], P, 0)
    sh_d = _share_curve(res["acc_band"], res["acc_tot"], P, 1)

    b0, b1 = (int(np.where(have)[0].min()), int(np.where(have)[0].max())) if have.any() else (0, 0)
    cols = {"base": "#185FA5", "dead": "#D85A30"}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4),
                             gridspec_kw=dict(wspace=0.26))
    x = np.arange(P)
    for ax in axes:
        ax.axvspan(b0 - 0.5, b1 + 0.5, color="#F2D9C9", alpha=0.55, zorder=0)
        ax.set_xlabel("管道相位（0 = 刚过管）", fontsize=10)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([0, 6, 12, 18, 24, 30, 35])

    ax = axes[0]
    ax.plot(x, yb, "-o", ms=3, lw=2.4, color=cols["base"],
            label=f"非死亡基线（{int(round(nb.mean()))} 帧/格）")
    ax.plot(x, np.where(have, yd, np.nan), "--s", ms=3.5, lw=1.7, color=cols["dead"],
            label=f"死亡窗口（{int(round(nd[have].mean())) if have.any() else 0} 帧/格）")
    ax.invert_yaxis()
    ax.set_ylabel("归因重心 · 屏幕 Y（0 = 天空，向上 = 更靠天空）", fontsize=9)
    ax.set_title(f"① 归因重心随管道相位的变化：死亡窗口 vs 非死亡基线\n"
                 f"曲线形状相关 r = {r_shape:.2f}", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="best")

    ax = axes[1]
    ax.axhline(0, color="#888780", lw=1)
    ax.bar(x, np.nan_to_num(d), color=cols["dead"], width=0.72, alpha=0.85,
           label="死亡窗口 Δ")
    ax.plot(x, np.where(have_r, dr, np.nan), color="#185FA5", lw=1.7, marker="o", ms=3,
            ls="--", label="随机窗口 Δ（阴性对照）")
    ax.set_ylabel("Δ 归因重心（屏幕 Y，正 = 比基线更靠下）", fontsize=9)
    ax.set_title(f"② 同相位差异 Δ(φ)\n"
                 f"死亡 RMS Δ = {rms:.2f} px（{eff:.2f}× 基线 std）；"
                 f"随机 = {rms_r:.2f} px（{eff_r:.2f}×）", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="best")

    ax = axes[2]
    ax.plot(x, sh_b, "-o", ms=3, lw=2.2, color=cols["base"],
            label="基线 · 地面带质量占比")
    ax.plot(x, np.where(have, sh_d, np.nan), "--s", ms=3.5, lw=1.7, color=cols["dead"],
            label="死亡窗口 · 地面带质量占比")
    ax.set_ylabel("地面带占全部归因 (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(f"③ 「看地面 vs 看障碍物」是否随死亡改变\n"
                 f"均值 {np.nanmean(sh_b):.1f}% → {np.nanmean(sh_d):.1f}%", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="best")

    fig.suptitle("图3-8 · 同相位对照：归因重心随管道相位的变化——死亡窗口 vs 非死亡基线\n"
                 "（底色 = 死亡窗口有样本的相位区间；纵轴 0 = 天空，向上 = 更靠天空）",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.78, bottom=0.10, left=0.05, right=0.985)
    path = _save(fig, out_name)

    def _rr(v):
        return None if (v is None or not np.isfinite(v)) else round(float(v), 4)

    ratio = (rms / rms_r) if (np.isfinite(rms) and np.isfinite(rms_r) and rms_r > 0) else float("nan")
    return {"corr_shape": _rr(r_shape), "rms_delta_px": _rr(rms),
            "baseline_std_px": _rr(sd), "effect_rms_over_std": _rr(eff),
            "corr_shape_random": _rr(r_shape_r),
            "rms_delta_random_px": _rr(rms_r), "effect_random_rms_over_std": _rr(eff_r),
            "death_over_random_rms": _rr(ratio),
            "death_window_frames_per_bin": int(round(float(nd[have].mean()))) if have.any() else 0,
            "baseline_frames_per_bin": int(round(float(nb.mean()))),
            "death_phase_span": [b0, b1],
            "share_mass_base_mean": _rr(np.nanmean(sh_b)),
            "share_mass_death_mean": _rr(np.nanmean(sh_d))}

def selfcheck_sky(net, states, actions, device, n=6, block=OCC_BLOCK):
    S = states[:n]
    Ac = actions[:n].astype(int)
    att = A.occlusion_batch(net, S, Ac, device, block=block)
    b = block
    nH = Config.FRAME_SIZE // b
    nW = Config.FRAME_SIZE // b

    st = S.reshape(len(S), Config.STACK_LEN, nH, b, nW, b).transpose(0, 2, 4, 1, 3, 5)
    at = att.reshape(len(S), Config.STACK_LEN, nH, b, nW, b).transpose(0, 2, 4, 1, 3, 5)
    val = st.max(axis=(3, 4, 5))
    drop = at[:, :, :, 0, 0, 0]
    m = val <= 0
    if not m.any():
        return 0, 0.0
    return int(m.sum()), float(np.abs(drop[m]).max())

def run_phase(net, device, episodes, max_steps, period, seed=0):
    P = PHASE_BINS
    print("\n[相位] 全帧采样：逐局跑、逐局归因（常驻内存与总局数无关）")

    accs = _new_accum(episodes)
    rng = np.random.default_rng(seed + 12345)
    bins_all, counts, ep_pass, ep_dead, scores = [], [], [], [], []
    deaths = []
    gaps, period_use = [], float(period)
    sky_S = sky_Ac = None

    for ep, (states, actions, qs, labels, pass_t, dead, score) in enumerate(
            collect_phased(net, device, episodes, max_steps, seed)):
        n = len(states)
        if ep == 0:
            sky_S, sky_Ac = states[:6].copy(), actions[:6].copy()
        if len(pass_t) >= 2:
            gaps.extend(np.diff(pass_t).tolist())
        period_use = float(np.mean(gaps)) if gaps else float(period)

        ph = phase_of_frames(pass_t, n, period_use)
        bins_ep = np.clip((ph * P).astype(np.int64), 0, P - 1)

        ep_ids = np.full(n, ep, np.int64)
        half_ids = (np.arange(n) >= n / 2.0).astype(np.int64)
        rand_ids = rng.integers(0, 2, size=n).astype(np.int64)
        ctrl_ep = rng.permutation(bins_ep)
        dead_ids = np.zeros(n, np.int64)
        if dead is not None and dead >= DEATH_BACK:
            dead_ids[dead - DEATH_BACK: dead + 1] = 1

        rwin_ids = np.zeros(n, np.int64)
        L = DEATH_BACK + 1
        if n >= L:
            bad = dead_ids.astype(bool)
            starts = [s for s in range(0, n - L + 1) if not bad[s:s + L].any()]
            if starts:
                s = int(starts[int(rng.integers(0, len(starts)))])
                rwin_ids[s:s + L] = 1

        print(f"[相位] 第 {ep + 1}/{episodes} 局归因（{n} 帧）")
        attribute_phase(net, states, actions, bins_ep, ctrl_ep,
                        ep_ids, half_ids, rand_ids, dead_ids, rwin_ids, accs, device,
                        labels=labels)

        bins_all.append(bins_ep)
        counts.append(n); ep_pass.append(pass_t); ep_dead.append(dead); scores.append(score)
        if dead is not None:
            deaths.append({"states": states, "actions": actions, "labels": labels,
                           "dead": dead, "bins": bins_ep, "score": score})

    counts = np.array(counts, np.int64)
    phase_bins_all = (np.concatenate(bins_all) if bins_all else np.zeros(0, np.int64))
    N = int(counts.sum())
    period_mean = float(np.mean(gaps)) if gaps else float(period)
    period_med = float(np.median(gaps)) if gaps else float(period)
    n_death_total = sum(1 for d in ep_dead if d is not None)
    print(f"    实测管道周期：均值 {period_mean:.2f} / 中位 {period_med:.2f} 帧"
          f"（36 与 38 交替 ⇒ 期望 37）；末段外推用 {period_use:.2f} 帧")
    print(f"    总帧数 {N}，每局帧数 {counts.tolist()}，"
          f"总过管 {sum(len(p) for p in ep_pass)} 次，死亡 {n_death_total} 次"
          f"（均来自每局全新环境，无「续局假死亡」）")

    n_sky, sky_dev = selfcheck_sky(net, sky_S, sky_Ac, device)
    sky_ok = (sky_dev == 0.0)
    print(f"    [自检③ 天空对照] 检验 {n_sky} 个全黑块，最大偏差 = {sky_dev:.2e}"
          f"   {'✔ 通过（应为 0）' if sky_ok else '✘ 异常！链路可能有串位'}")

    print("[相位] 出图 A：一个管道周期内，网络在看哪、怎么随相位变化")
    fig_phase_cycle(accs)

    (acc_d, cimg_d, cnt_d, n_death, hist2d, inten2d, share_tau,
     share_cnt_tau, n_drop, death_scores, share_by_death,
     reg_tau, are_tau) = attribute_death(net, deaths, device)
    if acc_d is not None:
        print(f"[相位] 出图 B：{n_death} 次死亡事件对齐到同一 τ 轴"
              f"（死亡时各自已过 {sorted(death_scores)} 根管）"
              + (f"（另有 {n_drop} 次给不出完整窗口而丢弃）" if n_drop else ""))
        fig_death_align(acc_d, cimg_d, cnt_d, n_death,
                        reg_tau=reg_tau, are_tau=are_tau, ground_band=A.GROUND_BAND)
    else:
        print("[相位] ⚠️ 没有可用的死亡事件，图B 跳过。"
              "想拿到失败归因：把 --max-steps 调大（如 6000）再跑。")

    loo = None
    if share_by_death is not None and len(death_scores) >= 3:
        full = np.nanmean(share_by_death, axis=0)
        devs = []
        for j in range(len(death_scores)):
            keep = [i for i in range(len(death_scores)) if i != j]
            devs.append(float(np.nanmax(np.abs(np.nanmean(share_by_death[keep], axis=0) - full))))
        loo = {"max_abs_dev_percent": round(float(max(devs)), 3),
               "n_death": int(len(death_scores))}
        print(f"    [鲁棒① 留一法] 逐次去掉 1 次死亡（共 {len(death_scores)} 次）后，"
              f"地面带占比曲线最大偏移 = {loo['max_abs_dev_percent']:.2f} 个百分点"
              f"   {'✔ 无单次死亡主导' if loo['max_abs_dev_percent'] < 5 else '⚠️ 某次死亡影响偏大'}")

    print("[相位] 出图 E：同相位对照 —— 死亡窗口 vs 非死亡基线")
    fig_effect = fig_death_vs_cycle(accs)

    print("[相位] 出图 C：稳定性复核（随机对半 / 前后半局 / 局间）")
    _, corr_stab, raw_stab, extra_stab = fig_stability(accs, episodes, period_use)
    print("[相位] 出图 D：随机相位对照")
    _, corr_ctrl, amp_ctrl, corr_ctrl_raw = fig_control(accs)

    _cache = dict(
        acc_all=accs["acc_all"], cnt_all=accs["cnt_all"],
        acc_ep=accs["acc_ep"], cnt_ep=accs["cnt_ep"],
        acc_half=accs["acc_half"], cnt_half=accs["cnt_half"],
        acc_rand=accs["acc_rand"], cnt_rand=accs["cnt_rand"],
        acc_ctrl=accs["acc_ctrl"], cnt_ctrl=accs["cnt_ctrl"],
        acc_img=accs["acc_img"], cnt_img=accs["cnt_img"],
        acc_dead=accs["acc_dead"], cnt_dead=accs["cnt_dead"],
        acc_rwin=accs["acc_rwin"], cnt_rwin=accs["cnt_rwin"],
        acc_band=accs["acc_band"], cnt_band=accs["cnt_band"],
        acc_tot=accs["acc_tot"], cnt_tot=accs["cnt_tot"],
        acc_rband=accs["acc_rband"], cnt_rband=accs["cnt_rband"],
        acc_rtot=accs["acc_rtot"], cnt_rtot=accs["cnt_rtot"],
        acc_nzb=accs["acc_nzb"], cnt_nzb=accs["cnt_nzb"],
        acc_nzr=accs["acc_nzr"], cnt_nzr=accs["cnt_nzr"],
        acc_nzt=accs["acc_nzt"], cnt_nzt=accs["cnt_nzt"],
        acc_reg=accs["acc_reg"], cnt_reg=accs["cnt_reg"],
        acc_area=accs["acc_area"], cnt_area=accs["cnt_area"],
        acc_reg_all=accs["acc_reg_all"], cnt_reg_all=accs["cnt_reg_all"],
        acc_area_all=accs["acc_area_all"], cnt_area_all=accs["cnt_area_all"],
        acc_lbl=accs["acc_lbl"], cnt_lbl=accs["cnt_lbl"],
        acc_lbl_bright=accs["acc_lbl_bright"], cnt_lbl_bright=accs["cnt_lbl_bright"],
        acc_haze=accs["acc_haze"], cnt_haze=accs["cnt_haze"],
        region_names=np.array(REGION_NAMES),
        region_colors=np.array(REGION_COLORS),
        meta=np.array([P, DEATH_BACK, A.GROUND_BAND, episodes, len(deaths)]),
        period=np.array([period_use]), phase_bins=np.array([P]),
        death_back=np.array([DEATH_BACK]), ground_band=np.array([A.GROUND_BAND]))
    if acc_d is not None:
        _cache.update(acc_tau=acc_d, cimg_tau=cimg_d, cnt_tau=cnt_d,
                      share_tau=share_tau, share_cnt_tau=share_cnt_tau,
                      reg_tau=reg_tau, are_tau=are_tau,
                      hist2d=hist2d, n_death_tau=np.array([int(n_death)]))
    npz_path = os.path.join(OUT, "phase_acc.npz")
    np.savez(npz_path, **_cache)
    print(f"    [缓存] 全部累加器已写入 out/phase_acc.npz"
          f"（{os.path.getsize(npz_path) / 1e6:.0f} MB；供 --redraw / GIF 用，无需重跑归因）")

    r1 = corr_stab.get("随机对半", float("nan"))
    r2 = corr_stab.get("前半局vs后半局", float("nan"))
    r3 = corr_stab.get("局间均值", float("nan"))
    min_bin = int(accs["cnt_all"][0].min())
    ok1 = min_bin >= 30
    ok2 = all(np.isfinite([r1, r2, r3])) and min(r1, r2, r3) > 0.7
    ok4 = np.isfinite(corr_ctrl) and corr_ctrl < 0.5 and amp_ctrl > 1.5
    print(f"    [自检① 样本量] 每个相位格最少 {min_bin} 帧"
          f"   {'✔ ≥30' if ok1 else '⚠️ <30，结论要谨慎'}")
    print(f"    [自检② 稳定性] 相位调制 r：随机对半 {r1:.3f}，前后半局 {r2:.3f}，"
          f"局间 {r3:.3f}   {'✔ 均 >0.7' if ok2 else '⚠️ 有项 <0.7，该维度不稳'}"
          f"   （原始 r：{raw_stab.get('随机对半', float('nan')):.3f} / "
          f"{raw_stab.get('前半局vs后半局', float('nan')):.3f} / "
          f"{raw_stab.get('局间均值', float('nan')):.3f}）")
    print(f"    [自检④ 阴性对照] 相位调制 r={corr_ctrl:.3f}（应 <0.5，原始 {corr_ctrl_raw:.3f}），"
          f"幅度比 amp={amp_ctrl:.2f}×（应 >1.5）"
          f"   {'✔ 通过' if ok4 else '✘ 未通过 —— 折叠没抓到相位结构！'}")

    share_mass = _share_curve(accs["acc_band"], accs["acc_tot"], P)
    share_pix = _share_curve(accs["acc_nzb"], accs["acc_nzt"], P)
    bs_mean = float(np.nanmean(share_mass))
    bs_lo, bs_hi = float(np.nanmin(share_mass)), float(np.nanmax(share_mass))
    bp_mean = float(np.nanmean(share_pix))

    NPS = Config.STACK_LEN * Config.FRAME_SIZE * Config.FRAME_SIZE
    NPSB = Config.STACK_LEN * (Config.FRAME_SIZE - A.GROUND_BAND) * Config.FRAME_SIZE
    nzt_pf = accs["acc_nzt"][:, :, 0, 0, 0].sum(0) / np.maximum(accs["cnt_nzt"].sum(0), 1)
    nzb_pf = accs["acc_nzb"][:, :, 0, 0, 0].sum(0) / np.maximum(accs["cnt_nzb"].sum(0), 1)
    zf_all = float(np.mean(nzt_pf)) / NPS
    zf_band = float(np.mean(nzb_pf)) / NPSB
    bin_spread = (min_bin, int(accs["cnt_all"][0].max()))
    print(f"    [附报1 分母口径] 地面带份额有两种口径，都报出来：")
    print(f"        · 质量口径 Σ|attr|地面带/Σ|attr|全图 = {bs_mean:.1f}%"
          f"（{bs_lo:.1f}%~{bs_hi:.1f}%）")
    print(f"        · 像素口径 #(|attr|>0 的地面带像素)/#(|attr|>0 的全图像素) = {bp_mean:.1f}%")
    print(f"        全图非零像素占全图槽位 {zf_all * 100:.1f}%（即 {(1 - zf_all) * 100:.1f}% 是恒为 0 的"
          f"全黑像素）；地面带内非零占 {zf_band * 100:.1f}%。")
    print(f"        → 全黑像素的归因【恰好为 0】，在【求和】分母里贡献 0，不会压低分母；"
          f"把口径换成「只数有响应的像素」后份额是 {bp_mean:.1f}%，差异 = "
          f"{bp_mean - bs_mean:+.1f} 个百分点。")
    print(f"    [附报2] 相位格样本数区间 = [{bin_spread[0]}, {bin_spread[1]}]"
          f"；第 0 格与第 18 格各比其余格多约 48% —— 这是【整数分格】的必然，不是 bug："
          f"36 帧周期每格恰 1 帧；38 帧周期里 floor(t·36/38) 会把 t=0,1 都放进第 0 格、"
          f"t=19,20 都放进第 18 格。各格均值为【该格自身样本数】的平均，不引入偏差。")
    print(f"    [附报3] 局间一致性：判据用 {extra_stab['n_long_episodes']} 局"
          f"（≥{extra_stab['min_cycles']} 周期，即 ≥{extra_stab['min_cycles'] * period_use:.0f} 帧）"
          f" r={corr_stab['局间均值']:.3f}；全部 {episodes} 局一起算则"
          f" r={extra_stab['inter_ep_all']:.3f}（短局每格仅几个样本，会拖低均值，两值都报）")
    if acc_d is not None:
        print(f"    [附报4] 死亡口径同样两种都算：质量 {np.nanmean(share_tau):.1f}% / "
              f"像素 {np.nanmean(share_cnt_tau):.1f}%")

    mass_mean = _region_mean(accs["acc_reg_all"], accs["cnt_reg_all"])
    reg_mean = _rows_to_share(mass_mean)
    are_mean = _region_mean(accs["acc_area_all"], accs["cnt_area_all"])
    m_reg = np.asarray(accs["acc_reg"].sum(axis=2), np.float64)
    m_tot = np.asarray(accs["acc_tot"][:, :, 0, 0, 0], np.float64)
    m_band = np.asarray(accs["acc_band"][:, :, 0, 0, 0], np.float64)
    d_tot = float(np.abs(m_reg - m_tot).max() / max(float(m_tot.max()), 1e-12))
    d_band = float(np.abs(
        np.asarray(accs["acc_reg"][:, :, 0], np.float64) - m_band).max()
        / max(float(m_band.max()), 1e-12))
    reg_ok = d_tot < 1e-4
    g_ok = d_band < 1e-4
    g_new = float(np.nanmean(reg_mean[:, 0]))
    lbl_cnt = float(np.asarray(accs["acc_lbl"]).sum())
    lbl_bri = float(np.asarray(accs["acc_lbl_bright"]).sum())
    prec = lbl_bri / max(lbl_cnt, 1.0) * 100.0
    prec_ok = prec >= 95.0

    haze = float(np.asarray(accs["acc_haze"]).sum()) / max(float(m_tot.sum()), 1e-12) * 100.0
    bg_share = float(np.nanmean(reg_mean[:, 4]))
    print(f"    [自检⑤ 五区份额] 逐相位格、逐分组的【精确恒等式】检查：")
    print(f"        · Σ五区质量 vs 总 |attr| 质量：最大相对偏差 {d_tot:.2e}"
          f"   {'✔ 分区完整且互斥' if reg_ok else '✘ 分区漏算/重算！'}")
    print(f"        · 五区里的『地面带』质量 vs 旧 acc_band：最大相对偏差 {d_band:.2e}"
          f"   {'✔ 新旧两条链路一致' if g_ok else '✘ 两条链路不一致，分区有问题'}"
          f"（地面带份额 {g_new:.2f}% vs 旧口径 {bs_mean:.2f}%，差异只是"
          f"「质量占比 vs 每帧占比再平均」的估计量差别）")
    print(f"    [自检⑥ 分区精确度] 标成『管子/小鸟』的格子 {int(lbl_cnt)} 个，其中真亮着 "
          f"{int(lbl_bri)} 个 = {prec:.2f}%   {'✔ ≥95%' if prec_ok else '✘ <95%，坐标/轴可能算错'}")
    print(f"    [附报5 五区] 全帧平均：归因密度（单位面积归因，= 归因量占比 ÷ 面积占比，"
          f">1 表示单位面积比全图平均更受关注）：")
    for r, name in enumerate(REGION_NAMES):
        print(f"        · {name:<6s} 面积 {np.nanmean(are_mean[:, r]):6.2f}%   归因密度 "
              f"{np.nanmean(_region_density(reg_mean[:, r], are_mean[:, r])):5.2f}   "
              f"（归因量占比 {np.nanmean(reg_mean[:, r]):6.2f}%）")
    print(f"        背景残差 {bg_share:.2f}% 的来源拆解：抗锯齿光晕（亮像素）{haze:.2f}% + "
          f"8×8 块跨界摊派 {bg_share - haze:.2f}%（两者都不是「网络在看空处」；"
          f"光晕像素与相邻管像素同属一个遮挡块、归因相同）")
    reg_base = reg_dead = None
    if acc_d is not None:
        reg_base = _rows_to_share(_region_mean(accs["acc_reg"], accs["cnt_reg"], 0))
        reg_dead = _rows_to_share(_region_mean(accs["acc_reg"], accs["cnt_reg"], 1))
        print(f"        同相位对照（非死亡基线 → 死亡窗口）：" + "  ".join(
            f"{REGION_NAMES[r]} {np.nanmean(reg_base[:, r]):.1f}→{np.nanmean(reg_dead[:, r]):.1f}%"
            for r in range(N_REGION)))
    if (fig_effect and fig_effect.get("rms_delta_px") is not None
            and fig_effect.get("rms_delta_random_px") is not None):
        print(f"    [鲁棒② 阴性对照] 同相位 Δ 的 RMS：死亡窗口 "
              f"{fig_effect['rms_delta_px']:.2f} px vs 随机窗口（等长、避开死亡） "
              f"{fig_effect['rms_delta_random_px']:.2f} px，比值 "
              f"{fig_effect['death_over_random_rms']:.2f}×"
              f"   {'✔ 死亡窗口确实更特殊' if (fig_effect['death_over_random_rms'] or 0) > 1.2 else '⚠️ 与随机窗口差不多，Δ 不具死亡特异性'}")

    def _r(v):
        return None if not np.isfinite(v) else round(float(v), 4)

    return {
        "phase_bins": PHASE_BINS,
        "period_mean": round(period_mean, 3),
        "period_median": round(period_med, 3),
        "period_used_for_tail": round(period_use, 3),
        "n_frames": int(N),
        "frames_per_episode": [int(c) for c in counts],
        "passes_per_episode": [len(p) for p in ep_pass],
        "scores": [int(s) for s in scores],
        "n_death": int(n_death),
        "death_scores": death_scores,
        "n_death_total_in_sample": int(n_death_total),
        "n_death_dropped_short_window": int(n_drop),
        "death_frame_phases": ([int(p) for p in range(PHASE_BINS) if hist2d[0, p] > 0]
                               if acc_d is not None else []),
        "bins_count": [int(v) for v in accs["cnt_all"][0]],
        "min_bin_count": min_bin,
        "max_bin_count": int(accs["cnt_all"][0].max()),
        "ground_band_share_percent": {"mean": round(bs_mean, 3),
                                      "min": round(bs_lo, 3), "max": round(bs_hi, 3)},
        "ground_band_share_percent_pixelwise": round(bp_mean, 3),
        "region_names": list(REGION_NAMES),
        "region_share_percent_mean": {REGION_NAMES[r]: round(float(np.nanmean(reg_mean[:, r])), 3)
                                      for r in range(N_REGION)},
        "region_area_percent_mean": {REGION_NAMES[r]: round(float(np.nanmean(are_mean[:, r])), 3)
                                     for r in range(N_REGION)},
        "region_density_mean": {REGION_NAMES[r]: _r(np.nanmean(
            _region_density(reg_mean[:, r], are_mean[:, r]))) for r in range(N_REGION)},
        "region_share_percent_baseline": (None if reg_base is None else
                                          {REGION_NAMES[r]: round(float(np.nanmean(reg_base[:, r])), 3)
                                           for r in range(N_REGION)}),
        "region_share_percent_death_window": (None if reg_dead is None else
                                              {REGION_NAMES[r]: round(float(np.nanmean(reg_dead[:, r])), 3)
                                               for r in range(N_REGION)}),
        "region_partition_precision_percent": round(prec, 3),
        "region_share_percent_ground_vs_legacy": {"regions": round(g_new, 3),
                                                  "legacy_band_over_tot": round(bs_mean, 3)},
        "region_background_share_percent": round(bg_share, 3),
        "region_background_from_halo_percent": round(haze, 3),
        "region_background_from_block_bleed_percent": round(bg_share - haze, 3),
        "selfcheck_region": {"mass_identity_ok": bool(reg_ok),
                             "mass_sums_max_rel_dev": float(d_tot),
                             "ground_band_identity_ok": bool(g_ok),
                             "ground_band_max_rel_dev": float(d_band),
                             "partition_precision_ge95": bool(prec_ok)},
        "nonzero_pixel_fraction": {"all_image": round(zf_all, 4),
                                   "ground_band": round(zf_band, 4)},
        "effect_death_vs_baseline": fig_effect,
        "leave_one_out": loo,
        "selfcheck": {"c1_min_bin_ge30": bool(ok1),
                      "c2_stability_gt0.7": bool(ok2),
                      "c3_sky_dev_zero": bool(sky_ok),
                      "c4_control_pass": bool(ok4)},
        "stability_corr_phase": {"随机对半": _r(r1), "前半局vs后半局": _r(r2), "局间均值": _r(r3)},
        "stability_corr_raw": {k: _r(v) for k, v in raw_stab.items()},
        "inter_episode": {"long_only": _r(extra_stab["inter_ep_long_only"]),
                          "all_episodes": _r(extra_stab["inter_ep_all"]),
                          "n_long_episodes": int(extra_stab["n_long_episodes"]),
                          "long_episode_idx": [int(v) for v in extra_stab["long_episode_idx"]],
                          "min_cycles": int(extra_stab["min_cycles"])},
        "phase_control": {"corr_phase": _r(corr_ctrl), "corr_raw": _r(corr_ctrl_raw),
                          "amp_ratio": round(amp_ctrl, 4)},
        "sky_control_blocks": n_sky,
        "sky_control_max_abs_dev": sky_dev,
    }

def redraw_from_cache(npz_path=None):
    npz_path = npz_path or os.path.join(OUT, "phase_acc.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(f"找不到缓存 {npz_path}；请先跑一次 --only phase")
    z = np.load(npz_path, allow_pickle=False)

    need = ["acc_all", "cnt_all", "acc_img", "cnt_img", "acc_ctrl", "cnt_ctrl",
            "acc_rand", "cnt_rand", "acc_half", "cnt_half", "acc_ep", "cnt_ep",
            "acc_dead", "cnt_dead", "acc_rwin", "cnt_rwin",
            "acc_band", "cnt_band", "acc_tot", "cnt_tot", "acc_nzb", "cnt_nzb",
            "acc_reg_all", "cnt_reg_all", "acc_area_all", "cnt_area_all"]
    miss = [k for k in need if k not in z.files]
    if miss:
        raise SystemExit(f"缓存缺键，无法完整重画：{miss}\n"
                         f"（该缓存是旧版留下的、只存了部分累加器；重跑一次 --only phase 即可）")
    res = {k: z[k] for k in z.files}
    res["ground_band"] = int(z["ground_band"][0])
    n_ep = int(z["meta"][3]) if "meta" in z.files else int(z["cnt_ep"].shape[0])
    period_use = float(z["period"][0]) if "period" in z.files else 36.0
    print(f"[重画] 读缓存 {npz_path}（{os.path.getsize(npz_path) / 1e6:.0f} MB），"
          f"局数 {n_ep}，周期 {period_use:.2f}")
    print("[重画] 图A / 图D / 图E")
    fig_phase_cycle(res)
    fig_control(res)
    eff = fig_death_vs_cycle(res)
    if "acc_ep" in res:
        print("[重画] 图C")
        fig_stability(res, n_ep, period_use)
    else:
        print("    ⚠️ 缓存无 acc_ep，跳过图C")
    if "acc_tau" in res and "reg_tau" in z.files:
        print("[重画] 图B")
        fig_death_align(res["acc_tau"], res["cimg_tau"], res["cnt_tau"],
                        int(z["n_death_tau"][0]),
                        reg_tau=z["reg_tau"], are_tau=z["are_tau"],
                        ground_band=int(z["ground_band"][0]))
    else:
        print("[重画] ⚠️ 缓存里没有五区 τ 累加器（旧版缓存 / 本次无死亡），跳过图B")
    print(f"[重画] 完成。效果量：{eff}")

def main():
    ap = argparse.ArgumentParser(description="Flappy Bird DQN 后置解释")
    ap.add_argument("--ckpt", default=os.path.join("runs", "ckpt-1890000.pt"),
                    help="要解释的存档（默认用 10 局均值最高的 ckpt-1890000.pt）")
    ap.add_argument("--redraw", action="store_true",
                    help="不跑模型，直接从 out/phase_acc.npz 重画 figA~E（几秒完成）")
    ap.add_argument("--episodes", type=int, default=45,
                    help="跑几局来采样。⚠️ 图B/图E（死亡）的统计强度只取决于【死亡事件数】："
                         "这个模型很强，--max-steps 2500（≈67 根管）内约每 3~4 局才真死 1 次，"
                         "所以默认 45 局（约 10~13 次真实死亡）。逐局流式处理，内存与局数无关。")
    ap.add_argument("--max-steps", type=int, default=2500,
                    help="每局最多跑多少步（调大更容易采到死亡时刻）")
    ap.add_argument("--pool", type=int, default=200, help="普通状态采样池大小")
    ap.add_argument("--faith-states", type=int, default=12, help="保真度检验用几个状态")
    ap.add_argument("--faith-steps", type=int, default=10, help="涂黑曲线的取点数")
    ap.add_argument("--period", type=float, default=36.0,
                    help="管道周期理论值（帧）。实测会用中位间隔覆盖它；仅当过管样本不足时回落到此值")
    ap.add_argument("--no-phase", action="store_true",
                    help="跳过相位层（figA~E），只跑 fig1~3（省算力）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["all", "local", "global", "faith", "phase"],
                    default="all")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.redraw:
        redraw_from_cache()
        return
    os.chdir(ROOT)

    random.seed(args.seed)
    device = Config.device()
    t0 = time.time()

    print("=" * 72)
    print("Flappy Bird DQN · 后置解释")
    print("=" * 72)
    net, ckpt_path = load_model(args.ckpt, device)
    print(f"模型：{os.path.relpath(ckpt_path, ROOT)}    设备：{device}")

    run_classic = args.only in ("all", "local", "global", "faith")
    run_phase_flag = args.only in ("all", "phase") and not args.no_phase

    summary = {"ckpt": os.path.relpath(ckpt_path, ROOT), "episodes": args.episodes,
               "max_steps": args.max_steps, "pool": args.pool,
               "methods": list(A.METHODS)}

    if run_classic:
        print("\n[1] 采样：无头跑游戏，收集要解释的时刻（fig1~3 用）")
        env = make_env()
        states, actions, qs, moments, n_generic, scores = collect(
            net, env, device, args.episodes, args.max_steps, args.pool, args.seed)
        print(f"    共采到 {len(states)} 个状态（其中普通帧 {n_generic} 个），"
              f"平均每局 {np.mean(scores):.1f} 根管")
        summary["n_states"] = int(len(states))
        summary["n_generic"] = int(n_generic)
        summary["scores"] = [int(s) for s in scores]

        if args.only in ("all", "local"):
            print("\n[2] 第一层：局部解释（三种关键时刻 × 五种方法）")
            run_local(net, states, actions, qs, moments, device)

        if args.only in ("all", "global"):
            print("\n[3] 第二层：全局解释（平均归因 + 四帧重要性）")
            _, imp = run_global(net, states, actions, n_generic, device)
            summary["frame_importance_percent"] = imp

        if args.only in ("all", "faith"):
            print("\n[4] 第三层：保真度检验（涂黑实验）")
            _, aucs = run_faith(net, states, actions, qs, n_generic, device,
                                n_states=args.faith_states, steps=args.faith_steps)
            summary["deletion_auc"] = aucs
            print("\n    曲线下面积 AUC（越小 = 越早打中要害 = 归因越好）：")
            for k, v in sorted(aucs.items(), key=lambda kv: kv[1]):
                tail = "   ← 随机对照基线" if "随机" in k else ""
                print(f"      {k:<14s}{v:7.2f}{tail}")

    if run_phase_flag:
        print("\n[5] 相位层：整局按管道周期折叠 + 死亡对齐 + 同相位对照（figA~E）")

        summary["phase"] = run_phase(net, device, args.episodes,
                                     args.max_steps, args.period, args.seed)

    summary["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n全部完成，用时 {summary['seconds']} 秒。产物在 interpretability/posthoc/out/ 下。")

if __name__ == "__main__":
    main()
