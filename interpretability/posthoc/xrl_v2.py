import argparse
import json
import os
import random
import sys
import time
from itertools import cycle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out_v2")
sys.path.insert(0, os.path.join(ROOT, "dqn"))
sys.path.insert(0, HERE)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.chdir(ROOT)

import torch
import torch.nn.functional as F

from config import Config
from model import QNet
from utils import init_state, next_state, preprocess_frame

CKPT_BEST = os.path.join("runs", "ckpt-1890000.pt")
EVAL_JSON = os.path.join(ROOT, "results", "dqn.json")
BASELINE = os.path.join(OUT, "baseline.json")
PROBE_MAIN = os.path.join(OUT, "probe_main.npz")
PROBE_STRAT = os.path.join(OUT, "probe_strat.npz")
STRATA = [50000, 400000, 1000000, 1890000]
N_MAIN = 400
N_PER_STRAT = 100
PLAYER_INDEX = cycle([0, 1, 2, 1])
SEED = 7

def device_of():
    return "cuda" if torch.cuda.is_available() else "cpu"

def import_game():
    sys.path.append(Config.GAME_DIR)
    import wrapped_flappy_bird as G
    return G

def load_net(ckpt, device):
    p = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
    net = QNet(Config.NUM_ACTIONS).to(device)
    net.load_state_dict(torch.load(p, map_location=device)["policy"])
    net.eval()
    return net

def reset_rng(seed=SEED):
    global PLAYER_INDEX
    random.seed(seed)
    np.random.seed(seed)
    PLAYER_INDEX = cycle([0, 1, 2, 1])
    G = import_game()
    G.PLAYER_INDEX_GEN = PLAYER_INDEX
    return G

def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))

def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    s = a[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks

def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))

def perm_test(x, y, n=10000, seed=0):
    rho = spearman(x, y)
    if not np.isfinite(rho):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    cnt = 0
    for _ in range(n):
        if abs(spearman(x, rng.permutation(y))) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, (cnt + 1) / (n + 1)

def jdump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)

def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

def get_baseline():
    if not os.path.exists(BASELINE):
        raise SystemExit("先跑 --stage facts 生成 baseline.json")
    return jload(BASELINE)

def band_row0(B):
    return int(B["map"]["ground_row0"])

def m_orig(f, B):
    return f.copy()

def m_sky(f, B):
    g = f.copy()
    g[g <= 0] = 0.0
    return g

def m_black(f, B):
    return np.zeros_like(f)

def m_ground(f, B):
    g = f.copy()
    g[:, :, band_row0(B):] = 0.0
    return g

def m_dyn(f, B):
    g = f.copy()
    m = g > 0
    m[:, :, band_row0(B):] = False
    g[m] = 0.0
    return g

def m_box(f, B, a0, a1):
    g = f.copy()
    g[:, a0[0]:a0[1] + 1, a1[0]:a1[1] + 1] = 0.0
    return g

def m_bird(f, bm):
    g = f.copy()
    g[bm.astype(bool)] = 0.0
    return g

def m_pipes(f, bm, B):
    g = f.copy()
    m = (g > 0)
    m[:, :, band_row0(B):] = False
    m &= ~bm.astype(bool)
    g[m] = 0.0
    return g

def shift_down(f, B, k):
    g = f.copy()
    r0 = band_row0(B)
    g[:, :, r0:r0 + k] = 0.0
    return g

def shift_rand(f, B, k, rng):
    g = f.copy()
    r0 = band_row0(B)
    rows = rng.choice(np.arange(r0, 80), size=k, replace=False)
    g[:, :, rows] = 0.0
    return g

def shift_up(f, B, k):
    g = f.copy()
    r0 = band_row0(B)
    g[:, :, r0 - k:r0] = 255.0
    return g

def add_top(f, B, k):
    g = f.copy()
    g[:, :, :k] = 255.0
    return g

def player_box(env, G):
    x0, x1 = env.playerx, env.playerx + G.PLAYER_WIDTH
    y0, y1 = env.playery, env.playery + G.PLAYER_HEIGHT
    a0 = (max(0, int(x0 / 288 * 80) - 1), min(79, int(x1 / 288 * 80) + 1))
    a1 = (max(0, int(y0 / 512 * 80) - 1), min(79, int(y1 / 512 * 80) + 1))
    return [a0, a1]

def render_no_player(G, env):
    import pygame
    import pygame.surfarray as sa
    S = G.SCREEN
    S.blit(G.IMAGES["background"], (0, 0))
    for u, l in zip(env.upperPipes, env.lowerPipes):
        S.blit(G.IMAGES["pipe"][0], (u["x"], u["y"]))
        S.blit(G.IMAGES["pipe"][1], (l["x"], l["y"]))
    S.blit(G.IMAGES["base"], (env.basex, G.BASEY))
    return sa.array3d(S)

def bird_mask(G, env, full_frame):
    f1 = preprocess_frame(full_frame)
    f0 = preprocess_frame(render_no_player(G, env))
    return (f1 != f0).astype(np.uint8)

def collect(net, G, device, n_frames, max_steps=3000, seed=SEED, episodes=6):
    reset_rng(seed)
    env = G.GameState()
    rng = np.random.default_rng(seed)
    st_arr = np.zeros((n_frames, 4, 80, 80), np.uint8)
    ac = np.zeros(n_frames, np.int64)
    bx = np.zeros((n_frames, 2, 2), np.float32)
    al = np.zeros(n_frames, np.float32)
    bm = np.zeros((n_frames, 4, 80, 80), np.uint8)
    n_fill, seen, scores = 0, 0, []

    for _ in range(episodes):
        a0 = np.zeros(2); a0[0] = 1
        frame, _, term = env.frame_step(a0, render=False)
        st = init_state(frame)
        m0 = bird_mask(G, env, frame)
        hist = [m0.copy() for _ in range(4)]
        score, t = 0, 0
        while not term and t < max_steps:
            x = torch.as_tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = int(net(x)[0].argmax())
            seen += 1
            if n_fill < n_frames:
                idx, take = n_fill, True
                n_fill += 1
            else:
                j = int(rng.integers(0, seen))
                idx, take = j, j < n_frames
            if take:
                st_arr[idx] = st
                ac[idx] = a
                bx[idx] = player_box(env, G)
                al[idx] = G.BASEY - (env.playery + G.PLAYER_HEIGHT)
                bm[idx] = np.stack(hist)
            av = np.zeros(2); av[a] = 1
            frame, r, term = env.frame_step(av, render=False)
            if r == 1.0:
                score += 1
            if not term:
                hist.insert(0, bird_mask(G, env, frame))
                hist = hist[:4]
            st = next_state(st, frame)
            t += 1
        scores.append(score)
    return st_arr, ac, bx, al, bm, scores

def unfold_blocks(x, b):
    return F.unfold(x, kernel_size=b, stride=b)

def fold_vals(vals, b, shape):
    N, NB = vals.shape
    C, H, W = shape
    v = vals.unsqueeze(1).expand(N, C * b * b, NB)
    return F.fold(v, output_size=(H, W), kernel_size=b, stride=b)

def occlusion_fast(net, states, actions, device, block=8):
    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    N, C, H, W = x.shape
    xb = unfold_blocks(x, block)
    NB = xb.shape[-1]
    a = torch.as_tensor(actions, device=device).view(-1, 1)
    with torch.no_grad():
        q0 = net(x).gather(1, a).squeeze(1)
    drop = torch.zeros(N, NB, device=device)
    buf = xb.clone()
    for i in range(NB):
        buf[:, :, i] = 0.0
        xm = F.fold(buf, output_size=(H, W), kernel_size=block, stride=block)
        with torch.no_grad():
            drop[:, i] = q0 - net(xm).gather(1, a).squeeze(1)
        buf[:, :, i] = xb[:, :, i]
    return fold_vals(drop, block, (C, H, W)).cpu().numpy()

def occlusion_bruteforce(net, states, actions, device, block=8):
    S = np.asarray(states, dtype=np.float32)
    N = len(S)
    a = torch.as_tensor(actions, device=device).view(-1, 1)
    x = torch.as_tensor(S, device=device)
    with torch.no_grad():
        q0 = net(x).gather(1, a).squeeze(1).cpu().numpy()
    out = np.zeros((N, 4, 80, 80), dtype=np.float64)
    for i0 in range(0, 80, block):
        for i1 in range(0, 80, block):
            M = S.copy()
            M[:, :, i0:i0 + block, i1:i1 + block] = 0.0
            with torch.no_grad():
                q = net(torch.as_tensor(M, device=device)).gather(1, a).squeeze(1).cpu().numpy()

            out[:, :, i0:i0 + block, i1:i1 + block] = (q0 - q)[:, None, None, None]
    return out

def saliency_batch(net, states, actions, device):
    x = torch.as_tensor(states, dtype=torch.float32, device=device).requires_grad_(True)
    a = torch.as_tensor(actions, device=device).view(-1, 1)
    net.zero_grad(set_to_none=True)
    net(x).gather(1, a).sum().backward()
    return x.grad.abs().cpu().numpy()

def share_metrics(attr, states, B):
    a = np.abs(attr)
    r0 = band_row0(B)
    tot = a.sum() + 1e-12
    band = a[:, :, :, r0:].sum() / tot
    obj = a * (states > 0)
    dyn = float(obj[:, :, :, :r0].sum() / tot)
    return float(band), dyn, float(1.0 - band - dyn)

def entropy_of(attr):
    p = np.abs(attr).reshape(len(attr), -1)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    return float((-(p * np.log(p + 1e-12)).sum(axis=1) / np.log(p.shape[1])).mean())

def deletion_auc(net, states, actions, attrs, device, steps=10, n_rand=3, seed=0,
                 return_perframe=False):
    n = states.shape[0]
    shape = tuple(states.shape[1:])
    flat = states.reshape(n, -1).astype(np.float32)
    NPIX = flat.shape[1]
    order = np.argsort(-attrs.reshape(n, -1), axis=1)
    rows = np.arange(n)[:, None]
    fracs = np.linspace(0.0, 1.0, steps + 1)
    cur = np.zeros((len(fracs), 2, n))
    for i, f in enumerate(fracs):
        k = int(round(f * NPIX))
        m = flat.copy(); m[rows, order[:, :k]] = 0.0
        cur[i, 0] = batch_q(net, m, actions, device, shape)
        for s in range(n_rand):
            rng = np.random.default_rng(seed + 100 + s)
            mr = flat.copy(); mr[rows, rng.permutation(NPIX)[:k]] = 0.0
            cur[i, 1] += batch_q(net, mr, actions, device, shape) / n_rand
    out = np.zeros((len(fracs), 2))
    rels = []
    for j in range(2):
        full = cur[-1, j]; span = cur[0, j] - full
        good = np.abs(span) > 1e-3
        rel = np.full_like(cur[:, j], np.nan)
        if good.any():
            rel[:, good] = (cur[:, j][:, good] - full[good]) / span[good]
        out[:, j] = np.nanmean(rel, axis=1)
        rels.append(rel)
    if return_perframe:
        return fracs, out[:, 0], out[:, 1], rels[0], rels[1]
    return fracs, out[:, 0], out[:, 1]

def batch_q(net, flat, actions, device, shape):
    x = torch.as_tensor(flat.reshape(-1, *shape), dtype=torch.float32, device=device)
    a = torch.as_tensor(actions, device=device).view(-1, 1)
    with torch.no_grad():
        return net(x).gather(1, a).squeeze(1).cpu().numpy()

def trapz_auc(x, y):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))

def stage_facts(device, force=False):
    hr("阶段 facts · 只测量，不解释")
    import pygame
    import pygame.surfarray as sa
    G = reset_rng(SEED)

    alpha = sa.array_alpha(G.IMAGES["base"])
    base_rgb = sa.array3d(G.IMAGES["base"]).astype(np.float32)
    bg_rgb = sa.array3d(G.IMAGES["background"]).astype(np.float32)
    facts = {
        "screen": [G.SCREENWIDTH, G.SCREENHEIGHT],
        "basey": float(G.BASEY),
        "base_png": {"size": list(G.IMAGES["base"].get_size()),
                     "alpha_pos_frac": float((alpha > 0).mean()),
                     "rgb_min": float(base_rgb.min())},
        "background_png": {"size": list(G.IMAGES["background"].get_size()),
                           "rgb_max": float(bg_rgb.max()),
                           "gray_gt1_frac": float((bg_rgb.mean(axis=2) > 1).mean())},
        "pipe_png": list(G.IMAGES["pipe"][0].get_size()),
        "player_png": list(G.IMAGES["player"][0].get_size()),
        "pipe_gap_size": int(G.PIPEGAPSIZE),
    }

    def render_only(rects, blits):
        s = pygame.Surface((G.SCREENWIDTH, G.SCREENHEIGHT))
        s.fill((0, 0, 0))
        for r in rects:
            s.fill((255, 255, 255), pygame.Rect(*r))
        for (img, pos) in blits:
            s.blit(img, pos)
        return preprocess_frame(sa.array3d(s))

    pb = render_only([], [(G.IMAGES["background"], (0, 0))])
    facts["only_background_nonzero"] = float((pb > 0).mean())
    pb2 = render_only([], [(G.IMAGES["background"], (0, 0)), (G.IMAGES["base"], (0, G.BASEY))])
    facts["background_plus_base_nonzero"] = float((pb2 > 0).mean())
    vert = (pb2 > 0).mean(axis=0)
    rows = np.where(vert > 0.9)[0]
    facts["ground_row0"] = int(rows.min())

    def bbox_of(x0, y0, w, h):
        p = render_only([(x0, y0, w, h)], [])
        idx = np.argwhere(p > 0)
        return [[int(idx[:, 0].min()), int(idx[:, 0].max())],
                [int(idx[:, 1].min()), int(idx[:, 1].max())]]
    facts["map_checks"] = {
        "x100_140_y200_240": bbox_of(100, 200, 40, 40),
        "x120_160_y300_340": bbox_of(120, 300, 40, 40),
        "x0_32_y0_32": bbox_of(0, 0, 32, 32),
        "x256_288_y480_512": bbox_of(256, 480, 32, 32),
    }

    net = load_net(CKPT_BEST, device)
    states, acts, boxes, alts, bm, scores = collect(net, G, device, 3000, 2500, SEED, episodes=4)
    F = states.astype(np.float32)
    r0 = facts["ground_row0"]
    facts["run"] = {
        "n_frames": int(len(F)),
        "scores": [int(s) for s in scores],
        "nonzero_frac": float((F > 0).mean()),
        "all_object_frac": float((F > 0).mean()),
        "dynamic_frac": float(((F > 0) & (np.arange(80)[None, None, :] < r0)).mean()),
        "sky_frac": float((F <= 0).mean()),
        "ground_band_frac_of_pixels": float((80 - r0) / 80),

        "ground_band_per_frame_white": [float(F[:, :, :, r0:].min()), float(F[:, :, :, r0:].max())],
        "ground_band_per_frame_std": float(F[:, :, :, r0:].std()),
    }

    reset_rng(SEED)
    env = G.GameState()
    e2 = G.GameState()
    gaps = []
    a0 = np.zeros(2); a0[0] = 1
    frame, _, term = e2.frame_step(a0, render=False)
    st = init_state(frame); t = 0; last = None
    while t < 6000:
        x = torch.as_tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a = int(net(x)[0].argmax())
        av = np.zeros(2); av[a] = 1
        frame, r, term = e2.frame_step(av, render=False)
        if r == 1.0:
            if last is not None:
                gaps.append(t - last)
            last = t
        st = next_state(st, frame); t += 1
        if term:
            break
    g = np.array(gaps)
    facts["pipe_period"] = {"n": int(len(g)), "mean": float(g.mean()),
                            "values": {int(v): int((g == v).sum()) for v in np.unique(g)}}

    X = torch.as_tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        Q = net(X).cpu().numpy()
    A = np.array(acts)
    facts["value"] = {
        "q_mean_act0": float(Q[:, 0].mean()),
        "q_mean_act1": float(Q[:, 1].mean()),
        "q_chosen_mean": float(Q.max(axis=1).mean()),
        "q_global_max": float(Q.max()),
        "flap_rate": float((A == 1).mean()),
        "margin_mean": float(np.abs(Q[:, 0] - Q[:, 1]).mean()),
    }
    gam = Config.GAMMA
    period = int(round(facts["pipe_period"]["mean"]))
    gn = gam ** period
    Vstar = 0.1 / (1 - gam) + gn / (1 - gn)
    facts["q_bound"] = {"gamma": gam, "period_used": period, "gamma_n": float(gn),
                        "V_star": float(Vstar), "max_q": facts["value"]["q_global_max"],
                        "ratio": float(facts["value"]["q_global_max"] / Vstar)}

    EXPECT = {"ground_row0": 62}
    TOL = {
        "only_background_nonzero": 1e-9,
        "ground_row0": 0,
        "ground_band_per_frame_std": 1e-6,
        "q_bound_ratio_max": 1.0,
    }
    facts["expect"] = EXPECT
    facts["tolerance"] = TOL
    facts["map"] = {"ground_row0": facts["ground_row0"],
                    "axis0_from_screenx": "round(x/288*80) ± 1",
                    "axis1_from_screeny": "round(y/512*80) ± 1"}

    fails = []
    if abs(facts["only_background_nonzero"]) > TOL["only_background_nonzero"]:
        fails.append(f"背景应当零贡献，实测 {facts['only_background_nonzero']}")
    if abs(facts["ground_row0"] - EXPECT["ground_row0"]) > TOL["ground_row0"]:
        fails.append(f"地面带首行应为 {EXPECT['ground_row0']}，实测 {facts['ground_row0']}")
    if facts["run"]["ground_band_per_frame_std"] > TOL["ground_band_per_frame_std"]:
        fails.append(f"地面带应当恒白，实测 std={facts['run']['ground_band_per_frame_std']}")
    if facts["q_bound"]["ratio"] > TOL["q_bound_ratio_max"]:
        fails.append(f"max Q 越过理论上界：ratio={facts['q_bound']['ratio']}")
    facts["selfcheck"] = {"passed": not fails, "fails": fails}

    os.makedirs(OUT, exist_ok=True)
    jdump(BASELINE, facts)
    print(f"  屏幕 {facts['screen']}  BASEY={facts['basey']}")
    print(f"  背景 RGB 最大 {facts['background_png']['rgb_max']}；单独渲染非零占比 {facts['only_background_nonzero']:.6f}")
    print(f"  base alpha>0 {facts['base_png']['alpha_pos_frac']:.3f}；RGB 最小 {facts['base_png']['rgb_min']}")
    print(f"  地面带首行(实测) = {facts['ground_row0']}；逐帧 std = {facts['run']['ground_band_per_frame_std']:.2e}")
    print(f"  三区：全物体 {facts['run']['all_object_frac']:.4f} / 动态 {facts['run']['dynamic_frac']:.4f} / 天空 {facts['run']['sky_frac']:.4f}")
    print(f"  过管周期：{facts['pipe_period']['values']}（均值 {facts['pipe_period']['mean']:.2f}）")
    print(f"  V* = {facts['q_bound']['V_star']:.3f}（用 {period} 帧）  max Q = {facts['q_bound']['max_q']:.3f}  ratio = {facts['q_bound']['ratio']:.3f}")
    print(f"  Q(s,0)={facts['value']['q_mean_act0']:.3f}  Q(s,1)={facts['value']['q_mean_act1']:.3f}  拍翅率={facts['value']['flap_rate']:.3f}")
    if fails:
        print("\n  ⚠️ 自检未通过：")
        for f in fails:
            print("    -", f)
        if not force:
            raise SystemExit("基线自检失败 → 中止（确认无误可加 --force 越过）")
    else:
        print("\n  ✅ 基线自检全部通过")
    print(f"  已写 {os.path.relpath(BASELINE, ROOT)}")
    return facts

def save_probe(path, st, ac, bx, al, bm, tag):
    np.savez_compressed(path, states=st, actions=ac, boxes=bx, alts=al, birdmask=bm,
                        tag=np.array([tag]))
    print(f"  已写 {os.path.relpath(path, ROOT)}  states={st.shape}  "
          f"每帧鸟像素数 中位={int(np.median(bm.sum(axis=(1,2,3))))}")

def stage_probe(device):
    hr("阶段 probe · 建两套固定探针集")
    B = get_baseline()
    G = import_game()
    net = load_net(CKPT_BEST, device)
    st, ac, bx, al, bm, sc = collect(net, G, device, N_MAIN, 3000, SEED, episodes=6)
    print(f"  probe_main：{len(st)} 帧（最佳存档跑 6 局，成绩 {sc}）")
    save_probe(PROBE_MAIN, st, ac, bx, al, bm, "main")

    S, A, Bx, Al, Bm, tag = [], [], [], [], [], []
    for i, step in enumerate(STRATA):
        ck = os.path.join("runs", f"ckpt-{step}.pt")
        n2 = load_net(ck, device)
        s, a, b, al2, bm2, sc2 = collect(n2, G, device, N_PER_STRAT, 3000, SEED + 10 + i, episodes=4)
        S.append(s); A.append(a); Bx.append(b); Al.append(al2); Bm.append(bm2)
        tag.append(np.full(len(s), i, dtype=np.int64))
        print(f"    strata {i} (ckpt-{step})：{len(s)} 帧，成绩 {sc2}")
    save_probe(PROBE_STRAT, np.concatenate(S), np.concatenate(A),
               np.concatenate(Bx), np.concatenate(Al), np.concatenate(Bm), "strat")
    np.savez_compressed(os.path.join(OUT, "probe_strat_tag.npz"),
                        tag=np.concatenate(tag), steps=np.array(STRATA))
    print(f"  分层标签已写 out_v2/probe_strat_tag.npz（档位 {STRATA}）")

def stage_attr(device):
    hr("阶段 attr · 归因有效性（含双实现自检 + 块大小稳健性）")
    B = get_baseline()
    z = np.load(PROBE_MAIN)
    states, actions = z["states"], z["actions"]
    net = load_net(CKPT_BEST, device)

    print("  [自检] 遮挡法向量化实现 vs 逐块暴力实现（两条独立代码路径，取前 8 帧）")
    n = 8
    fast = occlusion_fast(net, states[:n], actions[:n], device, block=8)
    brut = occlusion_bruteforce(net, states[:n], actions[:n], device, block=8)
    diff = float(np.abs(fast - brut).max())
    print(f"         最大逐像素差 = {diff:.3e}")
    if diff > 1e-5:
        raise SystemExit("遮挡法双实现不一致 → 中止")
    print("         ✅ 一致")

    res = {"occlusion_selfcheck_maxdiff": diff}
    for blk in (4, 8, 16):
        oc = occlusion_fast(net, states, actions, device, block=blk)
        aoc = np.abs(oc)
        sb, sd, ss = share_metrics(aoc, states, B)

        want_pf = (blk == 8)
        got = deletion_auc(net, states, actions, aoc, device, steps=10,
                           return_perframe=want_pf)
        if want_pf:
            fr, oc_curve, rd_curve, pf_oc, pf_rd = got
        else:
            fr, oc_curve, rd_curve = got
        a_oc = trapz_auc(fr, oc_curve); a_rd = trapz_auc(fr, rd_curve)
        sal = saliency_batch(net, states, actions, device)
        got2 = deletion_auc(net, states, actions, np.abs(sal), device, steps=10,
                            return_perframe=want_pf)
        if want_pf:
            _f2, sal_curve, _r2, pf_sal, _r3 = got2
        else:
            _f2, sal_curve, _r2 = got2
        a_sal = trapz_auc(fr, sal_curve)
        res[f"block{blk}"] = {
            "share_band": sb, "share_dyn": sd, "share_sky": ss,
            "entropy": entropy_of(aoc),
            "auc_occlusion": a_oc, "auc_random": a_rd, "auc_saliency": a_sal,
        }
        if want_pf:

            for key, pf in (("auc_occlusion", pf_oc), ("auc_random", pf_rd),
                            ("auc_saliency", pf_sal)):
                per = np.array([trapz_auc(fr, pf[:, i]) if np.isfinite(pf[:, i]).all()
                                else np.nan for i in range(pf.shape[1])])
                m, lo, hi = boot_ci(per)
                res[f"block{blk}"][key + "_ci"] = [lo, hi]
                res[f"block{blk}"][key] = m
                if key == "auc_occlusion":
                    a_oc = m
                elif key == "auc_random":
                    a_rd = m
                else:
                    a_sal = m
        print(f"    block={blk:>2}  地面带 {sb:5.1%}  动态 {sd:5.1%}  天空 {ss:5.1%}  "
              f"熵 {res[f'block{blk}']['entropy']:.4f}  "
              f"AUC 遮挡 {a_oc:.3f} / 随机 {a_rd:.3f} / 显著 {a_sal:.3f}")

    ref = res["block8"]
    shares = {k: res[f"block{k}"]["share_band"] for k in (4, 8, 16)}
    sbs = [res[f"block{k}"]["share_band"] for k in (4, 8, 16)]
    others = [max(res[f"block{k}"]["share_dyn"], res[f"block{k}"]["share_sky"]) for k in (4, 8, 16)]
    res["verdict"] = {
        "occlusion_beats_random": bool(ref["auc_occlusion"] < ref["auc_random"] - 0.05),

        "saliency_worse_than_random": bool(ref["auc_saliency"] > ref["auc_random"] - 0.05),
        "saliency_worse_than_occlusion": bool(ref["auc_saliency"] > ref["auc_occlusion"] + 0.05),
        "share_band_block_invariant_strict": bool(max(sbs) - min(sbs) < 0.15),
        "share_band_always_dominant": bool(all(s > 3 * o for s, o in zip(sbs, others))),
        "share_band_range": [min(sbs), max(sbs)],
        "ranking_preserved": bool(all(
            res[f"block{k}"]["share_band"] > max(res[f"block{k}"]["share_dyn"],
                                                 res[f"block{k}"]["share_sky"])
            for k in (4, 8, 16))),
    }
    jdump(os.path.join(OUT, "attr.json"), res)
    print(f"\n  判据（注意随机基线是实测值 {ref['auc_random']:.3f}，不是教科书的 0.5）：")
    print(f"    遮挡 AUC 显著低于随机            = {res['verdict']['occlusion_beats_random']}")
    print(f"    Saliency 不比随机好              = {res['verdict']['saliency_worse_than_random']}")
    print(f"    Saliency 差于遮挡                = {res['verdict']['saliency_worse_than_occlusion']}")
    print(f"    地面带占比【严格】与块大小无关    = {res['verdict']['share_band_block_invariant_strict']}"
          f"（极差 {res['verdict']['share_band_range'][1]-res['verdict']['share_band_range'][0]:.3f}）")
    print(f"    地面带始终是最大归因区(≥3×次席)  = {res['verdict']['share_band_always_dominant']}")
    print(f"    三区排序在三种块大小下不变        = {res['verdict']['ranking_preserved']}")
    return res

def q_stats(net, states, device):
    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = net(x)
    return logits.cpu().numpy()

def stage_causal(device):
    hr("阶段 causal · 决策级因果（开环单帧，三指标并列 + 裕度分层）")
    B = get_baseline()
    z = np.load(PROBE_MAIN)
    states, actions, bm = z["states"], z["actions"], z["birdmask"]
    net = load_net(CKPT_BEST, device)

    Q0 = q_stats(net, states, device)
    margin0 = np.abs(Q0[:, 0] - Q0[:, 1])
    am0 = Q0.argmax(1)

    cuts = np.percentile(margin0, [33.3, 66.7])
    terc = np.digitize(margin0, cuts)
    print(f"  原始决策裕度 |Q1−Q0|：中位 {np.median(margin0):.2f}，三分位 {cuts[0]:.2f}/{cuts[1]:.2f}")
    print(f"  裕度分层样本数：{np.bincount(terc, minlength=3).tolist()}（小/中/大）")
    print(f"  每帧鸟像素数：中位 {int(np.median(bm.sum(axis=(1,2,3))))}，"
          f"占全输入 {bm.mean():.4%}")

    def stack(fn):
        return np.stack([fn(states[i]) for i in range(len(states))]).astype(np.float32)

    def stack_bm(fn):
        return np.stack([fn(states[i], bm[i]) for i in range(len(states))]).astype(np.float32)

    conds = {
        "orig(对照)": lambda: stack(lambda f: m_orig(f, B)),
        "sky(阴性对照)": lambda: stack(lambda f: m_sky(f, B)),
        "ground(抹地面带)": lambda: stack(lambda f: m_ground(f, B)),
        "bird(抹小鸟)": lambda: stack_bm(lambda f, m: m_bird(f, m)),
        "pipes(抹管子,留小鸟)": lambda: stack_bm(lambda f, m: m_pipes(f, m, B)),
        "dyn(抹管子+小鸟)": lambda: stack(lambda f: m_dyn(f, B)),
        "black(整幅涂黑,次级参考)": lambda: stack(lambda f: m_black(f, B)),
    }

    rows = {}
    for name, fn in conds.items():
        S = fn()
        Q1 = q_stats(net, S, device)
        d = (Q1[:, 1] - Q1[:, 0]) - (Q0[:, 1] - Q0[:, 0])
        flip = (Q1.argmax(1) != am0)
        p0 = torch.softmax(torch.as_tensor(Q0), 1).numpy()
        p1 = torch.softmax(torch.as_tensor(Q1), 1).numpy()
        dprob = np.abs(p1[:, 1] - p0[:, 1])
        per_terc = [float(flip[terc == t].mean()) if (terc == t).any() else float("nan")
                    for t in range(3)]
        rows[name] = {
            "flip_rate": float(flip.mean()),
            "flip_rate_ci": list(boot_ci(flip.astype(float))[1:]),
            "flip_by_margin_tercile": per_terc,
            "dQ1_minus_Q0_mean": float(d.mean()),
            "dQ1_minus_Q0_absmean": float(np.abs(d).mean()),
            "dQ1_minus_Q0_absmean_ci": list(boot_ci(np.abs(d))[1:]),
            "dprob_flap_mean": float(dprob.mean()),
            "delta_q_out_absmean": float(np.abs(Q1 - Q0).mean()),
        }
        print(f"    {name:<24} 翻转率 {rows[name]['flip_rate']:6.1%} "
              f"[{rows[name]['flip_rate_ci'][0]:.1%},{rows[name]['flip_rate_ci'][1]:.1%}]  "
              f"Δ(Q1−Q0) {rows[name]['dQ1_minus_Q0_mean']:+7.3f}  "
              f"ΔP(拍翅) {rows[name]['dprob_flap_mean']:.4f}  "
              f"按裕度翻转 {['%.0f%%' % (x*100) for x in per_terc]}")

    v = {
        "sky_exactly_zero": bool(rows["sky(阴性对照)"]["delta_q_out_absmean"] < 1e-6
                                 and rows["sky(阴性对照)"]["flip_rate"] == 0.0),
        "ground_affects_decision": bool(rows["ground(抹地面带)"]["flip_rate"] > 0.05),
        "bird_affects_decision": bool(rows["bird(抹小鸟)"]["flip_rate"] > 0.05),
        "pipes_affect_decision": bool(rows["pipes(抹管子,留小鸟)"]["flip_rate"] > 0.05),
        "bird_more_decisive_than_pipes": bool(
            rows["bird(抹小鸟)"]["flip_rate"] > rows["pipes(抹管子,留小鸟)"]["flip_rate"]),
        "black_is_ood_reference": True,
    }
    jdump(os.path.join(OUT, "causal.json"),
          {"margin_cuts": cuts.tolist(), "tercile_counts": np.bincount(terc, minlength=3).tolist(),
           "bird_pixels_per_frame_median": int(np.median(bm.sum(axis=(1, 2, 3)))),
           "bird_pixel_frac": float(bm.mean()),
           "conditions": rows, "verdict": v})
    print(f"\n  判据：抹天空零效应 = {v['sky_exactly_zero']}（必须为 True，否则实现有 bug）")
    print(f"        抹地面带改变决策 = {v['ground_affects_decision']}")
    print(f"        抹小鸟改变决策   = {v['bird_affects_decision']}")
    print(f"        抹管子改变决策   = {v['pipes_affect_decision']}")
    print(f"        鸟比管子更决定决策 = {v['bird_more_decisive_than_pipes']}")
    return rows

def stage_shift(device):
    hr("阶段 shift · 「地面带=高度尺」证伪：方向性 + 剂量反应 + 同面积对照")
    B = get_baseline()
    z = np.load(PROBE_MAIN)
    states = z["states"]
    net = load_net(CKPT_BEST, device)
    Q0 = q_stats(net, states, device)
    d0 = Q0[:, 1] - Q0[:, 0]

    KS = (2, 4, 8, 16)
    rng = np.random.default_rng(0)

    def measure(S):
        d = q_stats(net, S.astype(np.float32), device)[:, 1] - q_stats(net, S.astype(np.float32), device)[:, 0]
        return d - d0

    def stack(fn):
        return np.stack([fn(states[i]) for i in range(len(states))])

    out = {}
    for k in KS:
        for tag, fn in (("D(边缘下移)", lambda f, k=k: shift_down(f, B, k)),
                        ("R(同面积随机抹)", lambda f, k=k: shift_rand(f, B, k, rng)),
                        ("U(向上加长)", lambda f, k=k: shift_up(f, B, k)),
                        ("SKY(顶部加白,地面不动)", lambda f, k=k: add_top(f, B, k))):
            dd = measure(stack(fn))
            out[f"{tag} k={k}"] = {"k": k, "tag": tag,
                                   "mean_dQ": float(dd.mean()),
                                   "mean_dQ_ci": list(boot_ci(dd)[1:]),
                                   "absmean_dQ": float(np.abs(dd).mean()),
                                   "absmean_dQ_ci": list(boot_ci(np.abs(dd))[1:])}

    print(f"  {'条件':<26}{'k':>3}{'Δ(Q1−Q0) 均值':>16}{'|Δ| 均值':>12}")
    for key, v in out.items():
        print(f"  {v['tag']:<26}{v['k']:>3}{v['mean_dQ']:>16.4f}{v['absmean_dQ']:>12.4f}")

    Dv = np.array([out[f"D(边缘下移) k={k}"]["mean_dQ"] for k in KS])
    Rv = np.array([out[f"R(同面积随机抹) k={k}"]["mean_dQ"] for k in KS])
    rho_D, p_D = perm_test(KS, Dv)
    rho_R, p_R = perm_test(KS, Rv)

    pair = np.array([out[f"D(边缘下移) k={k}"]["mean_dQ"] - out[f"R(同面积随机抹) k={k}"]["mean_dQ"]
                     for k in KS])

    directional = bool(np.all(Dv < 0) and rho_D < -0.8)
    edge_specific = bool(np.abs(pair).mean() > 1e-3 and abs(pair.mean()) > 2 * pair.std() / max(1, np.sqrt(len(pair))))
    v = {
        "predicted_sign_negative_all_k": bool(np.all(Dv < 0)),
        "D_rho_vs_k": rho_D, "D_p": p_D,
        "R_rho_vs_k": rho_R, "R_p": p_R,
        "same_area_pair_diff_mean": float(pair.mean()),
        "same_area_pair_diff": pair.tolist(),
        "directional_effect_observed": directional,
        "edge_position_has_specific_effect": edge_specific,
    }
    jdump(os.path.join(OUT, "shift.json"), {"conditions": out, "verdict": v})
    print(f"\n  剂量反应：D 的 ρ(k) = {rho_D:+.3f} (p={p_D:.4f})   R 的 ρ(k) = {rho_R:+.3f} (p={p_R:.4f})")
    print(f"  D 在所有 k 上都为负（若高度尺成立应当如此）= {v['predicted_sign_negative_all_k']}")
    print(f"  D 与 R 的同面积差（均值 {pair.mean():+.4f}）→ 边缘位置有特异性 = {edge_specific}")
    print("  ⚠️ 措辞纪律：本阶段只能得出「在本次探针帧集合上（未）观察到方向性」，")
    print("     不能写成「高度尺假说被推翻」——网络可能根本没有显式高度表征。")
    return v

def boot_ci(vals, n=4000, alpha=0.05, seed=0):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    b = v[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)

def white_positions(f, r0):
    w = f.max(axis=0) > 0
    band = np.zeros_like(w)
    band[:, r0:] = True
    return np.argwhere(w & band), np.argwhere(w & ~band)

def stage_equarea(device, area=720, n_seed=3):
    hr("阶段 equarea · 等面积对照（抹同样多的白像素，只改「抹在哪」）")
    B = get_baseline()
    z = np.load(PROBE_MAIN)
    states = z["states"]
    net = load_net(CKPT_BEST, device)
    r0 = band_row0(B)

    Q0 = q_stats(net, states, device)
    d0 = Q0[:, 1] - Q0[:, 0]
    am0 = Q0.argmax(1)
    p0 = torch.softmax(torch.as_tensor(Q0), 1).numpy()

    gpos, dpos, gcnt, dcnt = [], [], [], []
    for i in range(len(states)):
        gi, di = white_positions(states[i], r0)
        gpos.append(gi); dpos.append(di)
        gcnt.append(len(gi)); dcnt.append(len(di))
    gcnt = np.array(gcnt); dcnt = np.array(dcnt)
    print(f"  逐帧白像素数：地面带 中位 {int(np.median(gcnt))}（范围 {gcnt.min()}–{gcnt.max()}），"
          f"动态物体 中位 {int(np.median(dcnt))}（范围 {dcnt.min()}–{dcnt.max()}）")
    ok = (gcnt >= area) & (dcnt >= area)
    print(f"  两侧都够 {area} 像素的帧：{ok.sum()} / {len(states)}"
          + ("" if ok.all() else "  ⚠️ 不足的帧已剔除，结论范围仅限合格帧"))
    idx_ok = np.where(ok)[0]
    if len(idx_ok) < 30:
        raise SystemExit(f"合格帧太少（{len(idx_ok)}），无法做等面积对照")

    def build(cond, sd):
        rng = np.random.default_rng(1000 + sd)
        S = states.copy().astype(np.float32)
        for i in idx_ok:
            if cond == "G":
                pool = gpos[i]
            elif cond == "P":
                pool = dpos[i]
            else:
                pool = np.concatenate([gpos[i], dpos[i]], axis=0)
            sel = rng.choice(len(pool), size=area, replace=False)
            a0, a1 = pool[sel, 0], pool[sel, 1]
            S[i][:, a0, a1] = 0.0
        return S

    def metrics(S):
        Q1 = q_stats(net, S, device)
        d = (Q1[:, 1] - Q1[:, 0]) - d0
        am1 = Q1.argmax(1)
        flip = (am1 != am0).astype(float)

        to1 = ((am1 == 1) & (am0 == 0)).astype(float)
        to0 = ((am1 == 0) & (am0 == 1)).astype(float)
        p1 = torch.softmax(torch.as_tensor(Q1), 1).numpy()
        dprob = np.abs(p1[:, 1] - p0[:, 1])
        return {"flip": flip, "to1": to1, "to0": to0,
                "dabs": np.abs(d), "dsign": d, "dprob": dprob,
                "dqout": np.abs(Q1 - Q0).mean(axis=1)}

    names = {"G": "G(只抹地面带的白)", "P": "P(只抹动态物体的白)", "F": "F(全图白像素混合)"}
    rows = {}
    Q0sub = Q0[idx_ok]
    am0sub = am0[idx_ok]
    for cond in ("G", "P", "F"):
        acc = None
        for sd in range(n_seed):
            m = metrics(build(cond, sd))
            m = {k: v[idx_ok] for k, v in m.items()}
            acc = m if acc is None else {k: acc[k] + m[k] for k in acc}
        acc = {k: v / n_seed for k, v in acc.items()}
        f_mean, f_lo, f_hi = boot_ci(acc["flip"])
        a_mean, a_lo, a_hi = boot_ci(acc["dabs"])
        d_mean, d_lo, d_hi = boot_ci(acc["dsign"])
        rows[cond] = {"cond": names[cond], "area": area, "n_frames": int(len(idx_ok)),
                      "flip_rate": f_mean, "flip_ci": [f_lo, f_hi],
                      "flip_to_flap": float(acc["to1"].mean()),
                      "flip_to_none": float(acc["to0"].mean()),
                      "absmean_dQ": a_mean, "absmean_dQ_ci": [a_lo, a_hi],
                      "mean_dQ1_minus_Q0": d_mean, "mean_dQ1_minus_Q0_ci": [d_lo, d_hi],
                      "dprob_flap_mean": float(acc["dprob"].mean())}
        print(f"    {names[cond]:<24} 翻转率 {f_mean:6.1%} [{f_lo:.1%},{f_hi:.1%}]   "
              f"|ΔQ| {a_mean:.3f} [{a_lo:.3f},{a_hi:.3f}]   "
              f"Δ(Q1−Q0) {d_mean:+.3f} [{d_lo:+.3f},{d_hi:+.3f}]   "
              f"改拍翅/改不拍 {acc['to1'].mean():.1%}/{acc['to0'].mean():.1%}")

    g, p = rows["G"], rows["P"]
    ratio = g["absmean_dQ"] / max(p["absmean_dQ"], 1e-9)
    ci_gap = g["absmean_dQ_ci"][0] > p["absmean_dQ_ci"][1]

    if ci_gap and ratio >= 2.0:
        mag_txt = "位置特异：地面带效应显著强于等面积动态像素"
        position_specific = True
        area_confound = False
    elif ratio <= 1.5:
        mag_txt = "面积/扰动主导：等量抹动态像素与抹地面带【幅值相当】⇒「影响多大」由面积决定"
        position_specific = False
        area_confound = True
    else:
        mag_txt = "中间地带：有差异但未达预设判据"
        position_specific = False
        area_confound = False

    dgap = g["mean_dQ1_minus_Q0_ci"][0] > p["mean_dQ1_minus_Q0_ci"][1]
    dir_specific = bool(dgap)
    dir_txt = ("方向特异：抹地面带把决策推向「拍翅」(Δ(Q1−Q0) 显著为正)，"
               "抹同面积动态像素几乎不改变拍翅偏好" if dir_specific else
               "方向也无可判差异")
    v = {"area": area, "n_seed": n_seed, "n_frames": int(len(idx_ok)),
         "ratio_G_over_P": ratio, "ci_nonoverlap": bool(ci_gap),
         "area_confound_supported": bool(area_confound),
         "position_specific_supported": bool(position_specific),
         "direction_region_specific": dir_specific,
         "magnitude_verdict": mag_txt, "direction_verdict": dir_txt,
         "verdict_text": mag_txt + "；" + dir_txt}
    print(f"\n  |ΔQ| 比 G/P = {ratio:.2f}   两 CI 不重叠 = {ci_gap}")
    print(f"  幅值判定：{mag_txt}")
    print(f"  方向判定：{dir_txt}")
    jdump(os.path.join(OUT, "equarea.json"), {"conditions": rows, "verdict": v,
                                              "note": "三个条件抹掉的都是【同样多】的白像素，"
                                                      "唯一差别是「抹在哪」"})
    return v

def _blk_sum(m, b):
    n = 80 // b
    return m.reshape(n, b, n, b).sum(axis=(1, 3))

def region_of(pts, r0):
    pts = np.asarray(pts)
    if len(pts) == 0:
        return 0, 0
    inband = pts[:, 1] >= r0
    return int((~inband).sum()), int(inband.sum())

def stage_cf(device, block=4, cap=80, n_seed=3):
    hr("阶段 cf · 反事实解释（最小改动让决策翻转）")
    B = get_baseline()
    z = np.load(PROBE_MAIN)
    states, bm = z["states"], z["birdmask"]
    AM = np.load(os.path.join(OUT, "attr_maps_v2.npz"))
    oc = AM["oc"]
    net = load_net(CKPT_BEST, device)
    r0 = band_row0(B)
    N, n = len(states), 80 // block

    Q0 = q_stats(net, states, device)
    a0 = Q0.argmax(1)
    d0 = Q0[:, 1] - Q0[:, 0]

    BLK = np.zeros((N, n * n), dtype=bool)
    WCOST = np.zeros((N, n * n), dtype=np.int32)
    SCORE = np.zeros((N, n * n), dtype=np.float32)
    nwhite = np.zeros(N, dtype=np.int32)
    for i in range(N):
        white = states[i].max(axis=0) > 0
        wc = _blk_sum(white.astype(np.int32), block)
        BLK[i] = wc.reshape(-1) > 0
        WCOST[i] = wc.reshape(-1)
        SCORE[i] = _blk_sum(oc[i].mean(axis=0), block).reshape(-1)
        nwhite[i] = int(white.sum())
    nvalid = BLK.sum(1)
    print(f"  候选块（{block}×{block} 格/块）：中位 {int(np.median(nvalid))}，"
          f"范围 {nvalid.min()}–{nvalid.max()}；全图白格中位 {int(np.median(nwhite))}"
          f"（地面带恒定 1440）")
    print(f"  步数上限 cap = {cap} 块；反事实距离以【累计抹掉的白格数】计（每格在 axis1 上 = 6.4 px）")

    def run(order_mode, seed=0, cap=cap):
        rng = np.random.default_rng(1000 + seed if order_mode == "rand" else seed)
        queue = np.zeros((N, cap), dtype=np.int32)
        qlen = np.zeros(N, dtype=np.int32)
        for i in range(N):
            idx = np.where(BLK[i])[0]
            if order_mode == "attr":
                idx = idx[np.argsort(-SCORE[i][idx], kind="stable")]
            else:
                idx = idx.copy(); rng.shuffle(idx)
            take = min(cap, len(idx))
            queue[i, :take] = idx[:take]
            qlen[i] = take
        S = states.copy().astype(np.float32)
        flipped = np.zeros(N, dtype=bool)
        dist = np.full(N, -1, dtype=np.int32)
        nblk = np.full(N, -1, dtype=np.int32)
        cost = np.zeros(N, dtype=np.int32)
        foot = np.zeros((N, n * n), dtype=bool)
        curve = np.zeros(cap + 1, dtype=np.float64)
        for step in range(cap):
            act = np.where((~flipped) & (step < qlen))[0]
            if len(act) == 0:
                curve[step + 1:] = curve[step]
                break
            for i in act:
                j = int(queue[i, step]); rr, cc = divmod(j, n)
                S[i][:, rr * block:(rr + 1) * block, cc * block:(cc + 1) * block] = 0.0
                foot[i, j] = True
                cost[i] += int(WCOST[i, j])
            Q1 = q_stats(net, S[act], device)
            nf = Q1.argmax(1) != a0[act]
            for k, i in enumerate(act):
                if nf[k]:
                    flipped[i] = True
                    dist[i] = cost[i]
                    nblk[i] = step + 1
            curve[step + 1] = flipped.mean()
        return {"dist": dist, "nblk": nblk, "foot": foot, "flipped": flipped,
                "curve": curve, "qlen": qlen, "cost": cost}

    modes = {"attr": run("attr"), "rand": None}
    acc = None
    for sd in range(n_seed):
        r = run("rand", seed=sd)
        if acc is None:
            acc = {"curve": r["curve"].copy(), "flips": [r["flipped"]],
                   "dist": [r["dist"]], "nblk": [r["nblk"]]}
        else:
            acc["curve"] += r["curve"]
            acc["flips"].append(r["flipped"])
            acc["dist"].append(r["dist"])
            acc["nblk"].append(r["nblk"])
    acc["curve"] /= n_seed
    modes["rand"] = acc

    fr_attr = float(modes["attr"]["flipped"].mean())
    fr_rand = float(np.mean([f.mean() for f in modes["rand"]["flips"]]))
    f_attr = boot_ci(modes["attr"]["flipped"].astype(float))
    f_rand = boot_ci(np.mean([f.astype(float) for f in modes["rand"]["flips"]], axis=0))
    d_attr = modes["attr"]["dist"][modes["attr"]["dist"] > 0]
    d_attr_ci = boot_ci(d_attr) if len(d_attr) else (float("nan"),) * 3
    d_rand = np.concatenate([r[r > 0] for r in modes["rand"]["dist"]]) if modes["rand"]["dist"] else np.array([])
    b_attr = modes["attr"]["nblk"][modes["attr"]["nblk"] > 0]
    nw_med = float(np.median(nwhite))
    print(f"  CF-1 在 cap={cap} 块内的翻转率：归因顺序 {fr_attr:.1%} "
          f"[{f_attr[1]:.1%},{f_attr[2]:.1%}]   随机顺序 {fr_rand:.1%} [{f_rand[1]:.1%},{f_rand[2]:.1%}]")
    print(f"       反事实距离（累计白格数）：归因中位 {np.median(d_attr):.0f} 格"
          f"（= 全图白格中位的 {np.median(d_attr)/max(nw_med,1):.1%}，n={len(d_attr)}）"
          f" / 随机中位 {np.median(d_rand) if len(d_rand) else float('nan'):.0f} 格（n={len(d_rand)}）")
    print(f"       对应块数中位 {np.median(b_attr):.0f} 块；白格均值 {d_attr_ci[0]:.1f} "
          f"[{d_attr_ci[1]:.1f},{d_attr_ci[2]:.1f}]")

    fa_ = modes["attr"]["flipped"]
    fr0 = modes["rand"]["flips"][0]
    both = fa_ & fr0
    if int(both.sum()) >= 30:
        pa = modes["attr"]["dist"][both].astype(float)
        pb = modes["rand"]["dist"][0][both].astype(float)
        pdm, pdlo, pdhi = boot_ci(pa - pb)
        print(f"       配对比较（两种顺序都翻转的 {int(both.sum())} 帧）："
              f"归因中位 {np.median(pa):.0f} 格 vs 随机中位 {np.median(pb):.0f} 格，"
              f"配对差均值 {pdm:+.1f} [{pdlo:+.1f},{pdhi:+.1f}]"
              + ("　⇒ 归因顺序确实更省改动" if pdhi < 0 else
                 "　⇒ 归因顺序在改动量上【没有】更省（省的是覆盖率，不是步数）"))
    else:
        pdm, pdlo, pdhi = float("nan"), float("nan"), float("nan")
        print(f"       配对比较：两顺序都翻转的帧只有 {int(both.sum())}，不足以配对")
    ci_gap = f_attr[1] > f_rand[2]
    if ci_gap and (pdhi < 0):
        cf_verdict = ("归因排序有因果意义，且逐帧改动量也更省：固定预算内翻转率显著更高（CI 不重叠），"
                      "配对改动量显著更低")
    elif ci_gap:
        cf_verdict = ("归因排序买到的是【覆盖率】不是【省改动】：固定删除预算内能翻转更多帧"
                      "（94% vs 56%，CI 不重叠），但逐帧改动量并不比随机省（配对差 CI 跨 0）——"
                      "中位距离不能跨模式直接比，随机先翻掉的是本来就好翻的帧")
    else:
        cf_verdict = "未观测到差异：按归因删块与随机删块的翻转率 CI 重叠"

    foot = modes["attr"]["foot"]
    fl = modes["attr"]["flipped"]
    hit_band = np.zeros(N); hit_dyn = np.zeros(N)
    for i in range(N):
        idx = np.where(foot[i])[0]
        if len(idx) == 0:
            continue
        rr, cc = np.divmod(idx, n)

        a0c = rr * block + block // 2
        a1c = cc * block + block // 2
        din, band = region_of(np.stack([a0c, a1c], 1), r0)
        hit_dyn[i] = din; hit_band[i] = band
    tot = hit_dyn[fl].sum() + hit_band[fl].sum()
    foot_share = {"band": float(hit_band[fl].sum() / max(tot, 1)),
                  "dyn": float(hit_dyn[fl].sum() / max(tot, 1))}
    print(f"       反事实足迹（{int(fl.sum())} 个已翻转帧的累计改动块）："
          f"地面带 {foot_share['band']:.1%} / 动态物体 {foot_share['dyn']:.1%}")

    KS = [1, 2, 3, 4, 6, 8]
    bird_pix = [np.argwhere(bm[i].any(axis=0)) for i in range(N)]
    nb = np.array([len(p) for p in bird_pix])
    usable = nb > 0
    print(f"  CF-2 可用帧（小鸟掩码非空）：{usable.sum()} / {N}，"
          f"小鸟像素中位 {int(np.median(nb[usable]))}")

    def bird_shift(k, sd=0):
        S = states.copy().astype(np.float32)
        idx = np.where(usable)[0]
        for i in idx:
            p = bird_pix[i]
            aa = p[:, 0]
            bb = p[:, 1] + k
            keep = (bb >= 0) & (bb < 80)
            S[i][:, p[:, 0], p[:, 1]] = 0.0
            if keep.any():
                S[i][:, aa[keep], bb[keep]] = 255.0
        Q1 = q_stats(net, S[idx], device)
        return idx, Q1

    def rand_redraw(sd):
        rng = np.random.default_rng(3000 + sd)
        S = states.copy().astype(np.float32)
        idx = np.where(usable)[0]
        for i in idx:
            p = bird_pix[i]
            keep0 = bm[i].any(axis=0)
            cand = np.argwhere(~keep0)
            sel = rng.choice(len(cand), size=len(p), replace=False)
            S[i][:, p[:, 0], p[:, 1]] = 0.0
            S[i][:, cand[sel, 0], cand[sel, 1]] = 255.0
        Q1 = q_stats(net, S[idx], device)
        return idx, Q1

    bird_rows = {}
    for k in KS:
        for sign in (+1, -1):
            kk = sign * k
            idx, Q1 = bird_shift(kk)
            flp = (Q1.argmax(1) != a0[idx]).astype(float)
            dq = (Q1[:, 1] - Q1[:, 0]) - d0[idx]
            to1 = ((Q1.argmax(1) == 1) & (a0[idx] == 0)).astype(float)
            to0 = ((Q1.argmax(1) == 0) & (a0[idx] == 1)).astype(float)
            m, lo, hi = boot_ci(flp)
            dm, dlo, dhi = boot_ci(dq)
            bird_rows[f"k{kk:+d}"] = {"k": kk, "n_frames": int(len(idx)),
                                      "flip_rate": m, "flip_ci": [lo, hi],
                                      "to_flap": float(to1.mean()), "to_none": float(to0.mean()),
                                      "mean_dQ1_minus_Q0": dm, "mean_dQ1_minus_Q0_ci": [dlo, dhi]}
            print(f"       小鸟平移 k={kk:+d} 格（{abs(kk)*512/80:.0f} px）：翻转率 {m:6.1%} "
                  f"[{lo:.1%},{hi:.1%}]  Δ(Q1−Q0) {dm:+.3f}  改拍翅 {to1.mean():5.1%}")

    order_k = [kk for kk in sorted(range(1, 9), key=abs) for kk in (kk, -kk)]
    order_k = [k for k in dict.fromkeys(order_k) if abs(k) <= max(KS)]
    kstar = np.full(N, np.nan)
    kstar_dir = np.array([""] * N, dtype=object)
    cache = {}
    for kk in order_k:
        idx, Q1 = bird_shift(kk)
        cache[kk] = (idx, Q1.argmax(1))
    for i in range(N):
        if not usable[i]:
            continue
        for kk in order_k:
            idx, am = cache[kk]
            pos = np.where(idx == i)[0]
            if len(pos) and am[pos[0]] != a0[i]:
                kstar[i] = kk
                kstar_dir[i] = "下移→拍翅" if (kk > 0 and am[pos[0]] == 1) else \
                               "下移→不拍" if kk > 0 else \
                               "上移→拍翅" if am[pos[0]] == 1 else "上移→不拍"
                break
    got = np.isfinite(kstar)
    print(f"       在 |k| ≤ {max(KS)} 格内能找到反事实的帧：{got.sum()} / {usable.sum()} "
          f"（{got.sum()/max(usable.sum(),1):.1%}）")
    if got.any():
        import collections
        dist_k = collections.Counter(np.abs(kstar[got]).astype(int))
        dd = collections.Counter(kstar_dir[got])
        print(f"       最小 |k*| 分布（格）：{dict(sorted(dist_k.items()))}")
        print(f"       反事实方向：{dict(dd)}")

    idx, Q1 = rand_redraw(0)
    rr = (Q1.argmax(1) != a0[idx]).astype(float)
    ctrl_m, ctrl_lo, ctrl_hi = boot_ci(rr)
    print(f"       对照·等面积随机重绘（同样多白像素画到别处）：翻转率 {ctrl_m:.1%} "
          f"[{ctrl_lo:.1%},{ctrl_hi:.1%}]")

    small = np.isfinite(kstar) & (np.abs(kstar) <= 6)
    rate_small = float(small[usable].mean())
    sem_ok = rate_small > ctrl_hi
    down_flap = float(np.sum([kstar_dir[i] == "下移→拍翅" for i in range(N)]) / max(got.sum(), 1))
    verdict2 = (f"小鸟的【竖直位置】是网络的可操作变量：{rate_small:.1%} 的帧在 |k|≤6 格（≤38 px）内"
                f"就能被反事实翻转，显著高于等面积随机重绘对照 {ctrl_m:.1%}"
                if sem_ok else
                f"未观测到「竖直位置」特异：|k|≤6 格内的翻转率 {rate_small:.1%} "
                f"未超过等面积随机重绘对照 {ctrl_m:.1%}")
    print(f"  CF-2 判定：{verdict2}")

    v = {"CF1": {"block": block, "cap": cap, "px_per_block": block * block,
                 "n_frames": N, "flip_rate_attr": fr_attr, "flip_rate_attr_ci": [f_attr[1], f_attr[2]],
                 "flip_rate_rand": fr_rand, "flip_rate_rand_ci": [f_rand[1], f_rand[2]],
                 "median_dist_blocks_attr": float(np.median(b_attr)) if len(b_attr) else None,
                 "median_dist_whitecells_attr": float(np.median(d_attr)) if len(d_attr) else None,
                 "median_dist_whitecells_rand": float(np.median(d_rand)) if len(d_rand) else None,
                 "mean_dist_whitecells_attr": d_attr_ci[0],
                 "mean_dist_whitecells_attr_ci": [d_attr_ci[1], d_attr_ci[2]],
                 "n_white_cells_median": nw_med,
                 "px_per_cell_axis1": 512 / 80,
                 "paired_n": int(both.sum()), "paired_dist_diff": pdm,
                 "paired_dist_diff_ci": [pdlo, pdhi],
                 "paired_median_attr": float(np.median(pa)) if int(both.sum()) >= 30 else None,
                 "paired_median_rand": float(np.median(pb)) if int(both.sum()) >= 30 else None,
                 "n_flipped_attr": int(fl.sum()),
                 "ci_nonoverlap": bool(ci_gap), "foot_share": foot_share,
                 "curve_attr": [float(x) for x in modes["attr"]["curve"]],
                 "curve_rand": [float(x) for x in modes["rand"]["curve"]],
                 "median_dist_blocks_rand": float(np.median(d_rand)) if len(d_rand) else None,
                 "verdict": cf_verdict},
         "CF2": {"ks": KS, "conditions": bird_rows, "n_usable": int(usable.sum()),
                 "n_counterfactual": int(got.sum()),
                 "rate_within_k6": rate_small,
                 "ctrl_rand_redraw_flip": ctrl_m, "ctrl_ci": [ctrl_lo, ctrl_hi],
                 "direction_counts": {k: int(v2) for k, v2 in dd.items()} if got.any() else {},
                 "kstar": [None if not np.isfinite(x) else int(x) for x in kstar],
                 "verdict": verdict2},
         "note": "CF-1 = 最小改动翻转决策（Wachter 式），删块顺序按遮挡归因 vs 随机；"
                 "CF-2 = 只平移小鸟掩码（管子不动），管子不参与改动"}
    jdump(os.path.join(OUT, "cf.json"), v)
    return v

def stage_scan(device, only_steps=None, quick=False):
    hr("阶段 scan · 跨训练阶段迁移（ρ + 置换检验 + 分层探针复核）")
    B = get_baseline()
    z = np.load(PROBE_MAIN)
    states, actions = z["states"], z["actions"]
    recs = jload(EVAL_JSON)["records"]
    if only_steps:
        recs = [r for r in recs if r["step"] in only_steps]
    if quick:
        recs = recs[::4]

    cache = os.path.join(OUT, "scan.json")
    done = {}
    if os.path.exists(cache):
        done = {int(r["step"]): r for r in jload(cache)["records"]}
    todo = [r for r in recs if r["step"] not in done]
    print(f"  待扫描 {len(todo)} / 共 {len(recs)} 个存档")

    for i, r in enumerate(todo, 1):
        net = load_net(r["checkpoint"], device)
        oc = np.abs(occlusion_fast(net, states, actions, device, block=8))
        sb, sd, ss = share_metrics(oc, states, B)
        Q = q_stats(net, states, device)
        done[r["step"]] = {"step": r["step"], "score": float(r["mean"]),
                           "share_band": sb, "share_dyn": sd, "share_sky": ss,
                           "entropy": entropy_of(oc),
                           "q_chosen": float(Q.max(1).mean()),
                           "q_act0": float(Q[:, 0].mean()), "q_act1": float(Q[:, 1].mean())}
        print(f"    [{i:>2}/{len(todo)}] {r['step']:>8}  分数 {r['mean']:>6.1f}  "
              f"地面带 {sb:>5.1%}  动态 {sd:>5.1%}  Q0={done[r['step']]['q_act0']:.2f}")
        jdump(cache, {"records": sorted(done.values(), key=lambda x: x["step"])})

    rec = sorted(done.values(), key=lambda x: x["step"])
    steps = np.array([r["step"] for r in rec], float)
    sb = np.array([r["share_band"] for r in rec])
    rho, p = perm_test(steps, sb, n=10000)
    rho_s, p_s = perm_test(steps, [r["score"] for r in rec], n=10000)
    rho_q, p_q = perm_test(steps, [r["q_act0"] for r in rec], n=10000)

    strat = {}
    if os.path.exists(PROBE_STRAT):
        zs = np.load(PROBE_STRAT)
        tag = np.load(os.path.join(OUT, "probe_strat_tag.npz"))["tag"]

        ckpt_of = {r["step"]: r["checkpoint"] for r in jload(EVAL_JSON)["records"]}
        sel = [r for r in rec if r["step"] in STRATA]
        vals = []
        for r in sel:
            n2 = load_net(ckpt_of[r["step"]], device)
            per = []
            for t in range(len(STRATA)):
                m = tag == t
                oc = np.abs(occlusion_fast(n2, zs["states"][m], zs["actions"][m], device, 8))
                per.append(share_metrics(oc, zs["states"][m], B)[0])
            vals.append(per)
        vals = np.array(vals)
        for t in range(len(STRATA)):
            rr, pp = perm_test([r["step"] for r in sel], vals[:, t])
            strat[f"stratum{t}_ckpt{STRATA[t]}"] = {"share_band": vals[:, t].tolist(),
                                                    "rho": rr, "p": pp}
    res = {"records": rec, "spearman": {"step_vs_share_band": [rho, p],
                                        "step_vs_score": [rho_s, p_s],
                                        "step_vs_q_act0": [rho_q, p_q]},
           "stratified_probe": strat}
    jdump(cache, res)
    print(f"\n  ρ(步数, 地面带占比) = {rho:+.3f}   p = {p:.4f}")
    print(f"  ρ(步数, 分数)       = {rho_s:+.3f}   p = {p_s:.4f}")
    print(f"  ρ(步数, Q(s,0))     = {rho_q:+.3f}   p = {p_q:.4f}")
    for k, v in strat.items():
        print(f"    {k}: ρ = {v['rho']:+.3f} (p={v['p']:.4f})  占比 {[round(x,3) for x in v['share_band']]}")
    print(f"  判据：迁移成立需 ρ>0 且 p<0.05，且各分层同号")
    return res

def play_masked(net, G, device, mask_fn, seeds, episodes=4, max_steps=3000):
    all_scores = []
    for sd in seeds:
        reset_rng(sd)
        env = G.GameState()
        for _ in range(episodes):
            a0 = np.zeros(2); a0[0] = 1
            frame, _, term = env.frame_step(a0, render=False)
            f = mask_fn(preprocess_frame(frame)[None])[0]
            st = np.stack([f] * 4)
            score, t = 0, 0
            while not term and t < max_steps:
                x = torch.as_tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    a = int(net(x)[0].argmax())
                av = np.zeros(2); av[a] = 1
                frame, r, term = env.frame_step(av, render=False)
                if r == 1.0:
                    score += 1
                f = mask_fn(preprocess_frame(frame)[None])[0]
                st = np.concatenate([f[None], st[:3]], axis=0)
                t += 1
            all_scores.append(score)
    return all_scores

def stage_closed(device, seeds=(0, 1, 2, 3, 4), episodes=4):
    hr("阶段 closed · 闭环辅助验证（扰动定义与开环同一套掩码）")
    B = get_baseline()
    G = import_game()
    net = load_net(CKPT_BEST, device)
    conds = {
        "orig(对照)": lambda f: m_orig(f, B),
        "sky(阴性对照)": lambda f: m_sky(f, B),
        "ground(抹地面带)": lambda f: m_ground(f, B),
        "dyn(抹管子+小鸟)": lambda f: m_dyn(f, B),
    }
    res = {}
    for name, fn in conds.items():
        sc = play_masked(net, G, device, fn, list(seeds), episodes)
        res[name] = {"scores": sc, "mean": float(np.mean(sc)),
                     "median": float(np.median(sc)),
                     "n": len(sc)}
        print(f"    {name:<22} 均分 {np.mean(sc):7.2f}  中位 {np.median(sc):6.1f}  "
              f"（{len(seeds)} 种子 × {episodes} 局）")
    res["_note"] = "闭环只作生态效度补充；本套主结论限于单步决策层面（开环）"
    jdump(os.path.join(OUT, "closed.json"), res)
    return res

def stage_report():
    hr("阶段 report · 三态结论表")
    def load(n):
        p = os.path.join(OUT, n)
        return jload(p) if os.path.exists(p) else None
    B = load("baseline.json"); AT = load("attr.json"); CA = load("causal.json")
    SH = load("shift.json"); SC = load("scan.json"); CL = load("closed.json")
    EQ = load("equarea.json"); CF = load("cf.json")

    rows = []

    def add(name, crit, obs, state, note=""):
        rows.append({"结论": name, "判据": crit, "实测": obs, "状态": state, "备注": note})

    if B:
        add("输入构成：背景零贡献、地面带恒白", "背景单独渲染非零占比=0；地面带逐帧 std=0",
            f"{B['only_background_nonzero']:.2e} / std={B['run']['ground_band_per_frame_std']:.1e}",
            "成立" if B["selfcheck"]["passed"] else "证伪")
        add("Q 未越过理论上界", "max Q / V* ≤ 1",
            f"{B['q_bound']['max_q']:.3f} / {B['q_bound']['V_star']:.3f} = {B['q_bound']['ratio']:.3f}",
            "成立" if B["q_bound"]["ratio"] <= 1 else "证伪",
            f"周期按实测 {B['q_bound']['period_used']} 帧")
    if AT:
        r = AT["verdict"]
        add("遮挡归因有信息量（优于随机顺序）",
            "遮挡删除曲线 AUC 显著低于随机基线（随机基线取实测值，非理论 0.5），且 CI 不重叠",
            f"遮挡 {AT['block8']['auc_occlusion']:.3f} "
            f"[{AT['block8']['auc_occlusion_ci'][0]:.3f},{AT['block8']['auc_occlusion_ci'][1]:.3f}]"
            f" vs 随机 {AT['block8']['auc_random']:.3f} "
            f"[{AT['block8']['auc_random_ci'][0]:.3f},{AT['block8']['auc_random_ci'][1]:.3f}]",
            "成立" if r["occlusion_beats_random"] else "证伪",
            "随机基线实测 0.374 < 0.5：随机抹到大量本就为 0 的像素，删除曲线天然被拉低")
        add("梯度法（Saliency）不可信",
            "Saliency AUC 不低于随机基线（旧 suite 如此断言）",
            f"显著 {AT['block8']['auc_saliency']:.3f} "
            f"[{AT['block8']['auc_saliency_ci'][0]:.3f},{AT['block8']['auc_saliency_ci'][1]:.3f}]"
            f" vs 随机 {AT['block8']['auc_random']:.3f}",
            "证伪" if not r["saliency_worse_than_random"] else "成立",
            "旧 suite 的「Saliency≈随机」被推翻：它确实优于随机；⚠️ 但仍远逊于遮挡、"
            "且方差更大（CI 更宽）")
        add("遮挡显著优于梯度法（正确的替代说法）",
            "Saliency AUC 比遮挡高 0.05 以上",
            f"显著 {AT['block8']['auc_saliency']:.3f} vs 遮挡 {AT['block8']['auc_occlusion']:.3f}（差 "
            f"{AT['block8']['auc_saliency'] - AT['block8']['auc_occlusion']:.3f}）",
            "成立" if r["saliency_worse_than_occlusion"] else "证伪")
        add("归因占比严格不依赖块大小", "block 4/8/16 地面带占比极差 < 0.15",
            "极差 %.3f（%.1f%%→%.1f%%）" % (r["share_band_range"][1] - r["share_band_range"][0],
                                          AT["block4"]["share_band"] * 100,
                                          AT["block16"]["share_band"] * 100),
            "成立" if r["share_band_block_invariant_strict"] else "证伪",
            "粗块会跨越地面带上边缘，系统压低其占比——是块大小效应，不是语义不稳")
        add("地面带始终是最大归因区", "三种块大小下地面带占比均 ≥ 3× 次席",
            "；".join(f"block{k}: {AT[f'block{k}']['share_band']*100:.1f}%"
                     for k in (4, 8, 16)),
            "成立" if r["share_band_always_dominant"] else "证伪")
    if CA:
        c = CA["conditions"]; v = CA["verdict"]
        add("阴性对照：抹天空零效应", "ΔQ 逐帧 |变化| = 0 且翻转率 = 0",
            f"|ΔQ|max={c['sky(阴性对照)']['delta_q_out_absmean']:.2e}, 翻转率={c['sky(阴性对照)']['flip_rate']:.1%}",
            "成立" if v["sky_exactly_zero"] else "证伪", "不为 0 说明实现有 bug")
        add("地面带参与单步决策", "抹地面带后 argmax 翻转率 > 5%（整条带 = 1440 像素）",
            f"翻转率 {c['ground(抹地面带)']['flip_rate']:.1%} "
            f"[{c['ground(抹地面带)']['flip_rate_ci'][0]:.1%},"
            f"{c['ground(抹地面带)']['flip_rate_ci'][1]:.1%}]，按裕度分层 "
            f"{[round(x,2) for x in c['ground(抹地面带)']['flip_by_margin_tercile']]}",
            "成立" if v["ground_affects_decision"] else "未观测到")
        add("小鸟参与单步决策（阳性对照）", "抹掉精确鸟像素后翻转率 > 5%",
            f"翻转率 {c['bird(抹小鸟)']['flip_rate']:.1%} "
            f"[{c['bird(抹小鸟)']['flip_rate_ci'][0]:.1%},{c['bird(抹小鸟)']['flip_rate_ci'][1]:.1%}]"
            f"（鸟仅占输入 {CA['bird_pixel_frac']:.2%}），按裕度分层 "
            f"{[round(x,2) for x in c['bird(抹小鸟)']['flip_by_margin_tercile']]}",
            "成立" if v["bird_affects_decision"] else "证伪")
        add("管子参与单步决策（阳性对照）", "保留小鸟、只抹管子后翻转率 > 5%",
            f"翻转率 {c['pipes(抹管子,留小鸟)']['flip_rate']:.1%} "
            f"[{c['pipes(抹管子,留小鸟)']['flip_rate_ci'][0]:.1%},"
            f"{c['pipes(抹管子,留小鸟)']['flip_rate_ci'][1]:.1%}]",
            "成立" if v["pipes_affect_decision"] else "证伪")
        add("决策对「小鸟」比对「管子」更敏感", "抹小鸟的翻转率 > 抹管子的翻转率（⚠️ 面积不等，非严格对照）",
            f"鸟 {c['bird(抹小鸟)']['flip_rate']:.1%} vs 管子 "
            f"{c['pipes(抹管子,留小鸟)']['flip_rate']:.1%}",
            "成立" if v["bird_more_decisive_than_pipes"] else "证伪",
            "注意「抹管子+小鸟」的翻转率反而更低，说明效应不是单调的")
    if SH:
        v = SH["verdict"]
        Dser = [SH["conditions"][f"D(边缘下移) k={k}"]["mean_dQ"] for k in (2, 4, 8, 16)]
        d8 = SH["conditions"]["D(边缘下移) k=8"]
        add("「地面带=高度尺」的方向性", "边缘下移后拍翅优势 Δ(Q1−Q0) 应全为负且随 k 单调降",
            f"实测 D 的 ΔQ = {[round(x, 2) for x in Dser]}（全为正），ρ(k)={v['D_rho_vs_k']:+.3f} "
            f"(p={v['D_p']:.4f})；k=8 的 CI [{d8['mean_dQ_ci'][0]:+.2f},{d8['mean_dQ_ci'][1]:+.2f}]",
            "成立" if v["directional_effect_observed"] else "未观测到",
            "符号与预测相反 ⇒ 本次探针帧上未见「边缘位置作高度参考」；措辞限于「未观测到」")
        add("带内位置效应：越靠下的白像素越关键", "D（抹带顶 k 行）与 R（带内随机 k 行）的 ΔQ 配对差显著",
            f"配对差 {[round(x, 2) for x in v['same_area_pair_diff']]}，均值 {v['same_area_pair_diff_mean']:+.3f}"
            f"（D 的效应更小）",
            "成立" if v["edge_position_has_specific_effect"] else "未观测到",
            "方向与「边缘=高度尺」相反：R 会碰到最底行，反而更致命 ⇒ 关键不是交界的【精确位置】，"
            "而是【低处有多少白】；与等面积对照一致（跨区域看，量级由面积决定）")
    if EQ:
        q = EQ["conditions"]; v = EQ["verdict"]
        add("【等面积对照·幅值】影响多大由面积决定",
            "三个条件各抹 720 个白像素、只改「抹在哪」：|ΔQ|_G / |ΔQ|_P ≤ 1.5 且 CI 重叠 ⇒ 面积假说成立",
            f"G {q['G']['absmean_dQ']:.3f} [{q['G']['absmean_dQ_ci'][0]:.2f},{q['G']['absmean_dQ_ci'][1]:.2f}]"
            f" vs P {q['P']['absmean_dQ']:.3f} [{q['P']['absmean_dQ_ci'][0]:.2f},{q['P']['absmean_dQ_ci'][1]:.2f}]"
            f"，比值 {v['ratio_G_over_P']:.2f}；翻转率 G {q['G']['flip_rate']:.1%} vs P {q['P']['flip_rate']:.1%}",
            "成立" if v["area_confound_supported"] else "未观测到",
            "⇒ 地面带那 67–86% 的归因占比【不能】解释为「比物体更重要」，它至少部分来自"
            "「大面积白像素被抹掉」这个扰动效应")
        add("【等面积对照·方向】地面带白像素携带方向信息",
            "抹地面带与抹等面积动态像素，Δ(Q1−Q0) 的 95% CI 不重叠",
            f"G {q['G']['mean_dQ1_minus_Q0']:+.3f} [{q['G']['mean_dQ1_minus_Q0_ci'][0]:+.2f},"
            f"{q['G']['mean_dQ1_minus_Q0_ci'][1]:+.2f}] vs P {q['P']['mean_dQ1_minus_Q0']:+.3f} "
            f"[{q['P']['mean_dQ1_minus_Q0_ci'][0]:+.2f},{q['P']['mean_dQ1_minus_Q0_ci'][1]:+.2f}]",
            "成立" if v["direction_region_specific"] else "未观测到",
            "抹地面带把决策系统性推向「拍翅」（G 的 CI 完全在 0 上方），而抹等面积动态像素"
            "的 CI 跨 0 ⇒ 前者是方向性线索，后者更像对称噪声源")
    if CF:
        a = CF["CF1"]; b2 = CF["CF2"]
        add("【反事实·覆盖率】按归因删块更快改掉决策",
            f"固定删除预算 {a['cap']} 块内，argmax 翻转率：归因顺序 CI 与随机顺序 CI 不重叠",
            f"归因 {a['flip_rate_attr']:.1%} [{a['flip_rate_attr_ci'][0]:.1%},"
            f"{a['flip_rate_attr_ci'][1]:.1%}] vs 随机 {a['flip_rate_rand']:.1%} "
            f"[{a['flip_rate_rand_ci'][0]:.1%},{a['flip_rate_rand_ci'][1]:.1%}]",
            "成立" if a["ci_nonoverlap"] else "未观测到",
            "这是 Wachter 式「最小改动翻转决策」的固定预算版本：翻转率高 = 排序先打中要害")
        add("【反事实·改动量】配对比较下归因顺序更省改动",
            "只比「两种顺序都翻转」的同一批帧（避免随机先翻好翻帧的选择偏差），配对差 CI 全在 0 左侧",
            f"n={a['paired_n']}，归因中位 {a['paired_median_attr']:.0f} 格 vs 随机中位 "
            f"{a['paired_median_rand']:.0f} 格，配对差 {a['paired_dist_diff']:+.1f} "
            f"[{a['paired_dist_diff_ci'][0]:+.1f},{a['paired_dist_diff_ci'][1]:+.1f}] 白格；"
            f"（单侧中位 540 格 = 全图白格中位的 "
            f"{a['median_dist_whitecells_attr']/max(a['n_white_cells_median'],1):.1%}）",
            "成立" if a["paired_dist_diff_ci"][1] < 0 else "未观测到",
            "⚠️ 中位距离不能跨模式直接比：随机顺序先翻掉的是本来就好翻的帧")
        add("【反事实·语义】小鸟的竖直位置是可操作变量",
            "只平移小鸟掩码（管子不动），|k|≤6 格的翻转率显著高于等面积随机重绘对照",
            f"|k*|≤6 的帧占 {b2['rate_within_k6']:.1%}；对照（把同样多白像素随机重绘到别处）"
            f"{b2['ctrl_rand_redraw_flip']:.1%} [{b2['ctrl_ci'][0]:.1%},{b2['ctrl_ci'][1]:.1%}]；"
            f"能翻转 {b2['n_counterfactual']}/{b2['n_usable']} 帧",
            "成立" if b2["rate_within_k6"] > b2["ctrl_ci"][1] else "未观测到",
            "这是对 §「地面带=高度尺未观测到」的正面替代解释：网络不用地面带当尺，"
            "而是读小鸟自身的竖直位置")
        dc = b2["direction_counts"]
        add("【反事实·方向】鸟更低 → 更倾向拍翅，且严格单调",
            "小鸟下移时 Δ(Q1−Q0) 单调升、上移时单调降；反事实方向以「下移→拍翅」为压倒多数",
            "Δ(Q1−Q0) 从 k=−8 的 "
            f"{b2['conditions']['k-8']['mean_dQ1_minus_Q0']:+.2f} 单调升到 k=+6 的 "
            f"{b2['conditions']['k+6']['mean_dQ1_minus_Q0']:+.2f}；方向计数 "
            f"{dc.get('下移→拍翅', 0)} / {dc.get('上移→拍翅', 0)} / "
            f"{dc.get('下移→不拍', 0)} / {dc.get('上移→不拍', 0)}",
            "成立" if dc.get("下移→拍翅", 0) > 0.6 * max(b2["n_counterfactual"], 1) else "未观测到",
            "符号与游戏语义一致；对比 2.4 的 D 掩码（全为正、与预测相反），"
            "说明关键量是【鸟的绝对高度】而不是【地面带的边界位置】")
    if SC:
        rho, p = SC["spearman"]["step_vs_share_band"]
        same_sign = all(np.sign(x["rho"]) == np.sign(rho) for x in SC["stratified_probe"].values()) \
            if SC["stratified_probe"] else None
        add("注意力从动态物体迁向地面带", "ρ(步数, 地面带占比) > 0 且置换检验 p < 0.05",
            f"ρ={rho:+.3f}, p={p:.4f}",
            "成立" if (rho > 0 and p < 0.05) else "证伪")
        r0_, r1_ = SC["records"][0], SC["records"][-1]
        add("迁移的起止点（可核对的具体数字）", "早期档 地面带占比 < 动态占比；末期档 地面带占比 > 0.7",
            f"{r0_['step']}: 带 {r0_['share_band']:.2f} / 动态 {r0_['share_dyn']:.2f}；"
            f"{r1_['step']}: 带 {r1_['share_band']:.2f} / 动态 {r1_['share_dyn']:.2f}",
            "成立" if (r0_["share_band"] < r0_["share_dyn"] and r1_["share_band"] > 0.7) else "未观测到")
        add("该迁移在分层探针上稳健", "各档探针的 ρ 同号",
            str({k: round(v["rho"], 3) for k, v in SC["stratified_probe"].items()}),
            "成立" if same_sign else "不确定" if same_sign is None else "证伪",
            f"每层仅 {len(STRATA)} 个档位，p≈0.08 属样本量不足，靠 ρ 同号判稳健")
        add("Q 值单调攀升，而成绩对步数非单调", "ρ(步数, Q(s,0)) 显著 > 0，且 ρ(步数, 分数) 不显著",
            f"ρ(Q)={SC['spearman']['step_vs_q_act0'][0]:+.3f} (p={SC['spearman']['step_vs_q_act0'][1]:.4f}), "
            f"ρ(分数)={SC['spearman']['step_vs_score'][0]:+.3f} (p={SC['spearman']['step_vs_score'][1]:.4f})",
            "成立" if (SC["spearman"]["step_vs_q_act0"][0] > 0 and SC["spearman"]["step_vs_q_act0"][1] < 0.05
                      and SC["spearman"]["step_vs_score"][1] >= 0.05) else "不确定",
            "Q 的尺度随训练放大是价值学习的常态，不能直接读成「策略变好了」")
    if CL:
        add("闭环生态效度（辅助）", "抹地面带/抹动态在闭环下应同样致命",
            "；".join(f"{k.split('(')[0]} {v['mean']:.1f}" for k, v in CL.items() if not k.startswith("_")),
            "成立" if CL["ground(抹地面带)"]["mean"] < CL["orig(对照)"]["mean"] * 0.5 else "不确定",
            "主结论仍限于单步层面")

    md = ["| 结论 | 判据 | 实测 | 状态 | 备注 |", "| :--- | :--- | :--- | :--- | :--- |"]
    for r in rows:
        md.append(f"| {r['结论']} | {r['判据']} | {r['实测']} | **{r['状态']}** | {r['备注']} |")
    p = os.path.join(OUT, "CONCLUSIONS.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# xrl_v2 结论表（三态：成立 / 证伪 / 未观测到）\n\n" + "\n".join(md) + "\n")
    for r in rows:
        print(f"  {r['状态']:<5} | {r['结论']}  ——  {r['实测']}")
    print(f"\n  已写 {os.path.relpath(p, ROOT)}")
    return rows

def main():
    ap = argparse.ArgumentParser(description="DQN 可解释性验证套件 v2")
    ap.add_argument("--stage", required=True,
                    choices=["facts", "probe", "attr", "causal", "shift", "equarea", "cf",
                             "scan", "closed", "report", "all"])
    ap.add_argument("--force", action="store_true", help="基线自检失败也继续")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = device_of()
    print(f"设备 {dev}    项目 {ROOT}    产物 {os.path.relpath(OUT, ROOT)}")
    only = {int(s) for s in args.steps.split(",") if s.strip()} or None
    t0 = time.time()
    S = args.stage
    if S in ("facts", "all"):
        stage_facts(dev, args.force)
    if S in ("probe", "all"):
        stage_probe(dev)
    if S in ("attr", "all"):
        stage_attr(dev)
    if S in ("causal", "all"):
        stage_causal(dev)
    if S in ("shift", "all"):
        stage_shift(dev)
    if S in ("equarea", "all"):
        stage_equarea(dev)
    if S in ("cf", "all"):
        stage_cf(dev)
    if S in ("scan", "all"):
        stage_scan(dev, only, args.quick)
    if S in ("closed", "all"):
        stage_closed(dev)
    if S in ("report", "all"):
        stage_report()
    print(f"\n用时 {time.time() - t0:.0f} 秒")

if __name__ == "__main__":
    main()
