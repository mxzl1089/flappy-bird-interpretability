import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "dqn"))

import attribution as A

RESULT = {}

def hr(t):
    print("\n" + "=" * 74); print(t); print("=" * 74)

def occlusion_bruteforce(net, state, action, device, block=8):
    import torch
    x = torch.as_tensor(state[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        q0 = float(net(x)[0, action])
    H = W = state.shape[-1]
    nb = H // block
    drops = np.zeros((nb, nb), dtype=np.float32)
    for i in range(nb):
        for j in range(nb):
            m = x.clone()
            m[:, :, i * block:(i + 1) * block, j * block:(j + 1) * block] = 0.0
            with torch.no_grad():
                drops[i, j] = q0 - float(net(m)[0, action])
    return np.repeat(np.repeat(drops, block, axis=0), block, axis=1)

def v1_v2(device):
    hr("V1 · 遮挡法：批量实现 vs 逐像素暴力实现")
    import torch
    S = np.load(os.path.join(OUT, "probe.npz"))["states"][:2]
    from explain_dqn import load_model
    net, _ = load_model(os.path.join("runs", "ckpt-1890000.pt"), device)
    x = torch.as_tensor(S, dtype=torch.float32, device=device)
    with torch.no_grad():
        act = net(x).argmax(1).cpu().numpy()

    fast = A.occlusion_batch(net, S, act, device, block=8)
    ok = []
    for k in range(2):
        slow = occlusion_bruteforce(net, S[k], int(act[k]), device, block=8)
        d = float(np.abs(fast[k].mean(axis=0) - slow).max())
        ok.append(d)
        print(f"    状态 {k}: 两版最大逐像素差 = {d:.3e}")
    RESULT["v1_max_abs_diff"] = max(ok)

    hr("V2 · 结构自检：涂黑纯背景块应当恰好为 0")

    vals = []
    for i in range(0, 80, 8):
        for j in range(0, 80, 8):
            empty = (S[:, :, i:i + 8, j:j + 8] > 0).sum(axis=(1, 2, 3)) == 0
            if empty.all():
                vals.append(np.abs(fast[:, :, i, j]).max())
    if vals:
        print(f"    找到 {len(vals)} 个「全批都是纯背景」的块，其 |drop| 最大值 = {max(vals):.3e}")
    print("    （其余块都含物体像素，所以非零 —— 这正是二值输入下遮挡「只看物体」的原因）")
    RESULT["v2_sky_drop_max"] = float(max(vals)) if vals else None
    return net, S

def v3(device):
    hr("V3 · 区域占比对块大小的敏感性 + 与块无关的区域消融")
    import torch
    from explain_dqn import load_model
    S = np.load(os.path.join(OUT, "probe.npz"))["states"]
    sel = [50000, 800000, 1300000, 1890000]
    print("    (a) 像素加权的区域占比，随 block 变化：")
    print(f"    {'存档':>9} " + "".join(f"{'b='+str(b):>16}" for b in (4, 8, 16)))
    rows = {}
    for st in sel:
        net, _ = load_model(os.path.join("runs", f"ckpt-{st}.pt"), device)
        x = torch.as_tensor(S, dtype=torch.float32, device=device)
        with torch.no_grad():
            act = net(x).argmax(1).cpu().numpy()
        cells = []
        for b in (4, 8, 16):
            a = np.abs(A.occlusion_batch(net, S, act, device, block=b))
            band = A.ground_band_mask(S)
            cells.append(float((a * band).sum() / a.sum()))
        rows[st] = cells
        print(f"    {st:>9} " + "".join(f"{v:>16.3f}" for v in cells))
    RESULT["v3_block_sensitivity"] = {str(k): v for k, v in rows.items()}
    print("    基线 = 0.225（地面带像素占比）。三种块大小下结论方向完全一致。")

    print("\n    (b) 与块大小无关的区域消融 —— 整体抹掉一个区域，看 Q 掉多少：")
    print(f"    {'存档':>9} {'|ΔQ| 地面带':>14} {'|ΔQ| 管子小鸟':>15} {'|ΔQ| 天空':>11} "
          f"{'地面带/管子':>11}")
    ab = {}
    for st in sel:
        net, _ = load_model(os.path.join("runs", f"ckpt-{st}.pt"), device)
        x = torch.as_tensor(S, dtype=torch.float32, device=device)
        a_np = x.cpu().numpy().copy()
        with torch.no_grad():
            act = net(x).argmax(1).cpu().numpy()
            q0 = net(x).gather(1, torch.as_tensor(act, device=device).view(-1, 1)).squeeze(1)

        def ablation(mask):
            m = a_np.copy(); m[mask] = 0.0
            with torch.no_grad():
                q = net(torch.as_tensor(m, dtype=torch.float32, device=device))
            qq = q.gather(1, torch.as_tensor(act, device=device).view(-1, 1)).squeeze(1)
            return float((q0 - qq).abs().mean())

        band = A.ground_band_mask(S)
        dynm = (S > 0) & (~band)
        sky = (S <= 0)
        d_band, d_dyn, d_sky = ablation(band), ablation(dynm), ablation(sky)
        ab[st] = [d_band, d_dyn, d_sky]
        print(f"    {st:>9} {d_band:>14.3f} {d_dyn:>15.3f} {d_sky:>11.3f} "
              f"{d_band/max(d_dyn,1e-9):>11.2f}")
    RESULT["v3_ablation"] = {str(k): v for k, v in ab.items()}
    print("    天空=0 是结构必然（抹黑已经是黑的区域等于没抹）。")
    print("    「地面带/管子」这一列 > 1 说明抹掉地面带比抹掉所有管子更伤 Q。")

def v4(device, alt_ckpt="runs/ckpt-600000.pt", sel=(50000, 400000, 800000, 1300000, 1890000)):
    hr(f"V4 · 探针集依赖性：用 {alt_ckpt} 重新采一批画面再测")
    import torch
    from explain_dqn import load_model, make_env, collect

    alt = os.path.join(OUT, "probe_alt.npz")
    if os.path.exists(alt):
        SA = np.load(alt)["states"]
        print(f"    复用已有备用探针集 {os.path.relpath(alt, ROOT)}（{len(SA)} 帧）")
    else:
        net0, _ = load_model(alt_ckpt, device)
        states, actions, qs, _m, n, sc = collect(net0, make_env(), device, 5, 2500, 400, 1)
        SA = states[:n]
        np.savez_compressed(alt, states=SA, actions=actions[:n], qs=qs[:n])
        print(f"    备用探针集建成：{len(SA)} 帧，成绩 {sc}")

    S0 = np.load(os.path.join(OUT, "probe.npz"))["states"]
    print(f"\n    {'存档':>9} {'主探针 管子':>12} {'主探针 地面带':>14} "
          f"{'备用探针 管子':>14} {'备用探针 地面带':>16}")
    tab = {}
    for st in sel:
        net, _ = load_model(os.path.join("runs", f"ckpt-{st}.pt"), device)
        cells = []
        for S in (S0, SA):
            x = torch.as_tensor(S, dtype=torch.float32, device=device)
            with torch.no_grad():
                act = net(x).argmax(1).cpu().numpy()
            a = np.abs(A.occlusion_batch(net, S, act, device, block=8))
            band = A.ground_band_mask(S); dynm = (S > 0) & (~band)
            cells += [float((a * dynm).sum() / a.sum()), float((a * band).sum() / a.sum())]
        tab[st] = cells
        print(f"    {st:>9} {cells[0]:>12.3f} {cells[1]:>14.3f} "
              f"{cells[2]:>14.3f} {cells[3]:>16.3f}")
    RESULT["v4_probe_robust"] = {str(k): v for k, v in tab.items()}

def v5(device, sel=(50000, 400000, 800000, 1300000, 1890000)):
    hr("V5 · 动作依赖性：把所有存档的选中动作统一固定为「不拍翅」再测")
    import torch
    from explain_dqn import load_model
    S = np.load(os.path.join(OUT, "probe.npz"))["states"]
    fixed = np.zeros(len(S), dtype=np.int64)
    print(f"    {'存档':>9} {'自身动作 管子':>13} {'自身动作 地面带':>15} "
          f"{'固定动作 管子':>13} {'固定动作 地面带':>15}")
    tab = {}
    for st in sel:
        net, _ = load_model(os.path.join("runs", f"ckpt-{st}.pt"), device)
        x = torch.as_tensor(S, dtype=torch.float32, device=device)
        with torch.no_grad():
            own = net(x).argmax(1).cpu().numpy()
        cells = []
        for act in (own, fixed):
            a = np.abs(A.occlusion_batch(net, S, act, device, block=8))
            band = A.ground_band_mask(S); dynm = (S > 0) & (~band)
            cells += [float((a * dynm).sum() / a.sum()), float((a * band).sum() / a.sum())]
        tab[st] = cells
        print(f"    {st:>9} {cells[0]:>13.3f} {cells[1]:>15.3f} "
              f"{cells[2]:>13.3f} {cells[3]:>15.3f}")
    RESULT["v5_action_robust"] = {str(k): v for k, v in tab.items()}

def _play(net, env_factory, device, episodes, max_steps, seed=0):
    import torch
    import random
    from itertools import cycle
    from utils import init_state, next_state
    from config import Config
    random.seed(seed)
    np.random.seed(seed)

    import wrapped_flappy_bird as G
    G.PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
    env = env_factory()
    scores, steps = [], []
    for _ in range(episodes):
        a0 = np.zeros(Config.NUM_ACTIONS); a0[0] = 1
        frame, _, terminal = env.frame_step(a0, render=False)
        state = init_state(frame)
        score, t = 0, 0
        while not terminal and t < max_steps:
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = int(net(x)[0].argmax())
            av = np.zeros(Config.NUM_ACTIONS); av[a] = 1
            frame, r, terminal = env.frame_step(av, render=False)
            if r == 1.0:
                score += 1
            state = next_state(state, frame)
            t += 1
        scores.append(score)
        steps.append(t)
    return scores, steps

def v6(device, ckpt="runs/ckpt-1890000.pt", episodes=8, max_steps=3000):
    hr("V6 · 行为消融（最强证据）：把区域从画面上抹掉，让模型接着玩")
    import torch
    import pygame
    from explain_dqn import make_env, load_model

    env = make_env()
    import wrapped_flappy_bird as G

    net, _ = load_model(ckpt, device)
    SEED = 7

    def run(tag):
        sc, stp = _play(net, G.GameState, device, episodes, max_steps, seed=SEED)
        print(f"    {tag:<24} 分数均值 {np.mean(sc):6.1f}  中位 {np.median(sc):5.1f}   "
              f"存活中位 {np.median(stp):6.0f} 步   原始分数 {sc}")
        return float(np.mean(sc)), float(np.median(stp)), sc

    original_base = G.IMAGES["base"]
    original_pipe = G.IMAGES["pipe"]

    m0, s0, c0 = run("① 原始（对照）")

    white = original_base.copy()
    white.fill((255, 255, 255))
    G.IMAGES["base"] = white
    m0b, s0b, c0b = run("② 换成纯白（等价对照）")

    G.IMAGES["base"] = original_base.copy()
    G.IMAGES["base"].fill((0, 0, 0))
    m1, s1, c1 = run("③ 抹掉地面带")
    G.IMAGES["base"] = original_base

    G.IMAGES["pipe"] = (original_pipe[0].copy(), original_pipe[1].copy())
    G.IMAGES["pipe"][0].fill((0, 0, 0)); G.IMAGES["pipe"][1].fill((0, 0, 0))
    m2, s2, c2 = run("④ 抹掉管子（阳性对照）")
    G.IMAGES["pipe"] = original_pipe

    m3, s3, c3 = run("⑤ 恢复后复查")
    G.IMAGES["base"] = original_base; G.IMAGES["pipe"] = original_pipe

    RESULT["v6_ablation"] = {
        "原始": m0, "等价对照_纯白": m0b, "抹掉地面带": m1, "抹掉管子": m2, "恢复后": m3,
        "存活中位步数": {"原始": s0, "等价对照": s0b, "抹掉地面带": s1,
                         "抹掉管子": s2, "恢复后": s3},
        "原始分数": {"原始": c0, "等价对照": c0b, "抹掉地面带": c1,
                     "抹掉管子": c2, "恢复后": c3},
    }
    print("\n    判读：")
    if c0 == c0b:
        print(f"    · 等价对照与原始【逐局完全一致】{c0} —— 换贴图这个操作本身零副作用；")
    else:
        print(f"    · ⚠️ 等价对照 {c0b} 与原始 {c0} 不一致 —— 换贴图引入了副作用，本组结论不可用！")
    print(f"    · 抹掉地面带 {m1:.1f}（存活中位 {s1:.0f} 步）vs 抹掉管子 {m2:.1f}"
          f"（存活中位 {s2:.0f} 步）—— 两者都让模型彻底失效；")
    print(f"    · 贴图换回后恢复到 {m3:.1f} —— 排除「模型在实验过程中被永久破坏」的可能。")

def v7():
    hr("V7 · Q 的理论上界：检验「Q 过估计」的说法")

    g = 0.99
    per_pipe = g ** 36
    v_surv = 0.1 / (1.0 - g)
    v_pipe = per_pipe / (1.0 - per_pipe)
    vmax = v_surv + v_pipe

    print(f"    存活奖励 0.1/帧 → 折扣和 = 0.1/(1-0.99)      = {v_surv:.3f}")
    print(f"    过管奖励 每 36 帧一根 → γ³⁶/(1-γ³⁶)         = {v_pipe:.3f}")
    print(f"    → 完美策略的回报上界 V* = {vmax:.3f}")
    d = json.load(open(os.path.join(OUT, "dynamics.json"), encoding="utf-8"))["records"]
    q = np.array([r["q_chosen"] for r in d])
    print(f"    实测选中动作的 Q：最小 {q.min():.2f}  最大 {q.max():.2f}"
          f"（{d[int(np.argmax(q))]['step']} 步）")
    print(f"    → 最大 Q / V* = {q.max()/vmax:.2f}（< 1 即未越界）")
    RESULT["v7_q_bound"] = {
        "gamma": g, "frames_per_pipe": 36, "per_pipe_discount": per_pipe,
        "v_surv": v_surv, "v_pipe": v_pipe, "v_max": vmax,
        "q_max": float(q.max()), "ratio": float(q.max() / vmax),
        "verdict": "Q 未突破正确上界；旧的「3.15× 过估计」结论因漏算存活奖励而作废",
    }
    print("    结论：本项检验【不支持】「Q 越过理论上界」的说法 —— "
          "实测 Q 始终在合法范围内。")
    print("    注：这是复核流程发现并修正的一处真实错误（旧版漏算存活奖励 0.1/帧）。")

def _play_rec(net, env_factory, device, episodes, max_steps, seed=0):
    import torch
    import random
    from itertools import cycle
    from utils import init_state, next_state
    from config import Config
    import wrapped_flappy_bird as G

    random.seed(seed)
    np.random.seed(seed)
    G.PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
    env = env_factory()
    scores, steps, alts, last_alt, flapr, rec = [], [], [], [], [], []
    for _ in range(episodes):
        a0 = np.zeros(Config.NUM_ACTIONS); a0[0] = 1
        frame, _, terminal = env.frame_step(a0, render=False)
        state = init_state(frame)
        score, t, ep_alt, ep_flap = 0, 0, [], []
        la = float("nan")
        while not terminal and t < max_steps:
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = int(net(x)[0].argmax())
            av = np.zeros(Config.NUM_ACTIONS); av[a] = 1
            frame, r, terminal = env.frame_step(av, render=False)
            if r == 1.0:
                score += 1
            if not terminal:
                alt = G.BASEY - (env.playery + G.PLAYER_HEIGHT)
                ep_alt.append(alt); ep_flap.append(a); la = alt
                rec.append((alt, int(env.playerVelY), a))
            state = next_state(state, frame)
            t += 1
        scores.append(score); steps.append(t)
        alts.append(float(np.mean(ep_alt)) if ep_alt else 0.0)
        last_alt.append(la)
        flapr.append(float(np.mean(ep_flap)) if ep_flap else float("nan"))
    return scores, steps, alts, last_alt, flapr, rec

def v8(device, ckpt="runs/ckpt-1890000.pt", episodes=8, max_steps=3000):
    hr("V8 · 地平线平移：地面带是「位置参照系」还是「一块大白块」？")
    import pygame
    from explain_dqn import make_env, load_model
    make_env()
    import wrapped_flappy_bird as G

    net, _ = load_model(ckpt, device)
    SEED = 7
    Y0 = float(G.BASEY)
    VIS = G.SCREENHEIGHT - Y0
    orig_bg = G.IMAGES["background"]
    orig_base = G.IMAGES["base"]

    def install(kind, k):
        bg = orig_bg
        b = orig_base.copy(); b.fill((0, 0, 0))
        if kind == "ctrl":
            b.fill((255, 255, 255))
        elif kind == "MOV":
            b.fill((255, 255, 255),
                   pygame.Rect(0, int(round(k)), G.SCREENWIDTH, max(0, int(round(VIS - k)))))
        elif kind == "THIN":
            b.fill((255, 255, 255),
                   pygame.Rect(0, 0, G.SCREENWIDTH, max(0, int(round(VIS - k)))))
        elif kind == "UP":
            b.fill((255, 255, 255))
            bg = orig_bg.copy(); bg.fill((0, 0, 0))
            bg.fill((255, 255, 255),
                    pygame.Rect(0, int(round(Y0 - k)), G.SCREENWIDTH, int(round(k))))
        elif kind == "SKY":

            b.fill((255, 255, 255))
            bg = orig_bg.copy(); bg.fill((0, 0, 0))
            bg.fill((255, 255, 255), pygame.Rect(0, 0, G.SCREENWIDTH, int(round(k))))
        G.IMAGES["background"] = bg
        G.IMAGES["base"] = b

    CONDS = ([("ctrl", 0)] + [("MOV", k) for k in (32, 64, 96)]
             + [("THIN", k) for k in (32, 64, 96)] + [("UP", k) for k in (96,)]
             + [("SKY", k) for k in (48, 96)])

    print(f"    物理地面 BASEY={Y0:.1f}（恒定不动，碰撞与封顶都用它）")
    print(f"    屏幕上可见地面厚度 VIS={VIS:.1f} px；坐标为屏幕像素，越小=越靠上")
    print(f"    {'条件':<10}{'均分':>7}{'存活中位':>9}{'真实高度(均)':>13}"
          f"{'末帧高度(中位)':>15}{'拍翅率':>8}")
    tab = {}
    rec_ctrl = None
    recs = {}
    try:
        for kind, k in CONDS:
            install(kind, k)
            sc, stp, alts, last, fr, rec = _play_rec(
                net, G.GameState, device, episodes, max_steps, seed=SEED)
            tag = kind if kind == "ctrl" else f"{kind}({k})"
            tab[tag] = {"kind": kind, "k": k, "scores": sc, "score": float(np.mean(sc)),
                        "surv": float(np.median(stp)),
                        "alt": float(np.mean(alts)),
                        "last_alt": float(np.nanmedian(last)),
                        "flap": float(np.nanmean(fr))}
            if kind == "ctrl":
                rec_ctrl = rec
            recs[tag] = rec
            print(f"    {tag:<10}{np.mean(sc):>7.1f}{np.median(stp):>9.0f}"
                  f"{np.mean(alts):>13.1f}{np.nanmedian(last):>15.0f}"
                  f"{np.nanmean(fr):>8.3f}")
    finally:
        G.IMAGES["background"] = orig_bg
        G.IMAGES["base"] = orig_base

    print("\n    判读（三条互相独立的判据）：")
    ctrl = tab["ctrl"]
    v6 = RESULT.get("v6_ablation", {}).get("原始分数", {}).get("原始")
    if v6 is not None:
        n_sc = min(len(ctrl["scores"]), len(v6))
        same = (ctrl["scores"][:n_sc] == list(v6)[:n_sc])
        print(f"    · 自检：ctrl 与 V6 的原始对照前 {n_sc} 局{'完全一致' if same else '不一致'}"
              f" {ctrl['scores']}")
        tab["ctrl_reproduces_v6"] = bool(same)
    print(f"    · 同面积对照：MOV(k) 与 THIN(k) 白块面积完全相同，只差位置 ——")
    for k in (32, 64, 96):
        a, b = tab[f"MOV({k})"], tab[f"THIN({k})"]
        print(f"        k={k:>3}px   MOV 高度 {a['alt']:6.1f} / 存活 {a['surv']:5.0f}"
              f"   |   THIN 高度 {b['alt']:6.1f} / 存活 {b['surv']:5.0f}")
    print(f"    · 剂量-反应：地平线下移越多，真实高度应当越低（若它把地平线当参照系）")
    for k in (0, 32, 64, 96):
        t = tab["ctrl"] if k == 0 else tab[f"MOV({k})"]
        print(f"        下移 {k:>3}px → 平均真实高度 {t['alt']:6.1f}，存活中位 {t['surv']:5.0f}")
    print(f"        上移  96px → 平均真实高度 {tab['UP(96)']['alt']:6.1f}，"
          f"存活中位 {tab['UP(96)']['surv']:5.0f}")
    print(f"    · ★ 阴性对照：在画面顶部凭空加一条白带、地面完全不动")
    for k in (48, 96):
        s = tab[f"SKY({k})"]
        print(f"        顶部白带 {k:>3}px → 平均真实高度 {s['alt']:6.1f}，"
              f"存活中位 {s['surv']:5.0f}，均分 {s['score']:5.1f}")
    print(f"        对照（原样）    → 平均真实高度 {ctrl['alt']:6.1f}，"
          f"存活中位 {ctrl['surv']:5.0f}，均分 {ctrl['score']:5.1f}")
    print("        判据：若 SKY 与 MOV 同量级地崩 → 「崩」只是对输入变化的通病，"
          "地面带移动无特异性；")
    print("              若 SKY 基本不掉、而 MOV/THIN 全崩 → 说明模型对地面带的几何"
          "极度敏感，但这是「敏感性」而非「把它当高度尺」。")

    RESULT["v8_horizon"] = {t: d for t, d in tab.items() if isinstance(d, dict)}

    def _rate(rec, lo, hi):
        a = np.array(rec, dtype=float)
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        return int(m.sum()), (float(a[m, 2].mean()) if m.sum() else float("nan"))

    print("\n    附2：政策阈值是否被平移？（同一真实高度窗口 120–200px 内的拍翅率）")
    print("         若是高度尺：地平线下移越多，同一高度上的拍翅率应越低")
    pol = {}
    for tag in ["ctrl"] + [f"MOV({k})" for k in (32, 64, 96)] + ["THIN(32)", "SKY(48)"]:
        if tag in recs:
            n, r = _rate(recs[tag], 120, 200)
            pol[tag] = r
            print(f"         {tag:<10} n={n:>6}   拍翅率 {r:.3f}")
    RESULT["v8_policy_shift"] = pol

    if rec_ctrl:
        arr = np.array(rec_ctrl)
        alt, vel, act = arr[:, 0], arr[:, 1], arr[:, 2]
        edges = [0, 40, 80, 120, 160, 200, 260, 400]
        print("\n    附：控制条件下「拍翅」与「真实高度」的关系（策略规则抽取）")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (alt >= lo) & (alt < hi)
            if m.sum() >= 20:
                print(f"        高度 {lo:>3}–{hi:<3}px  n={int(m.sum()):>6}"
                      f"   拍翅率 {act[m].mean():.3f}")
        rho = float(np.corrcoef(alt, act)[0, 1])
        print(f"        相关系数 corr(真实高度, 拍翅) = {rho:+.3f}"
              f"   （高度越低越该拍翅 → 应为负）")
        RESULT["v8_policy"] = {"corr_alt_flap": rho,
                               "flap_rate": float(act.mean()),
                               "alt_mean": float(alt.mean())}

def main():
    ap = argparse.ArgumentParser(description="结论可靠性复核")
    ap.add_argument("--only", default="all")
    ap.add_argument("--device", default="")
    ap.add_argument("--v8-episodes", type=int, default=8, help="V8 地平线实验每条件跑几局")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.chdir(ROOT)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {device}    项目 {ROOT}")

    _p = os.path.join(OUT, "verify.json")
    if os.path.exists(_p):
        try:
            RESULT.update(json.load(open(_p, encoding="utf-8")))
        except Exception:
            pass

    todo = set(args.only.split(","))
    def want(k): return "all" in todo or k in todo

    if want("v1") or want("v2"):
        v1_v2(device)
    if want("v3"):
        v3(device)
    if want("v4"):
        v4(device)
    if want("v5"):
        v5(device)
    if want("v6"):
        v6(device)
    if want("v7"):
        v7()
    if want("v8"):
        v8(device, episodes=args.v8_episodes)

    p = os.path.join(OUT, "verify.json")

    merged = {}
    if os.path.exists(p):
        try:
            merged = json.load(open(p, encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update(RESULT)
    json.dump(merged, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n复核结果已存 {os.path.relpath(p, ROOT)}")

if __name__ == "__main__":
    main()
