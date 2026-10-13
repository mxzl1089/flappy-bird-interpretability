import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(ROOT, "dqn"))
sys.path.insert(0, HERE)

from config import Config
from utils import init_state, next_state
import explain_dqn as E
import attribution as A

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REGION_COLORS = ["#9AA0A6", "#D85A30", "#185FA5", "#1D9E75", "#C9C6BF"]
DEATH_BACK = 30

def sample(net, env, device, episodes, max_steps, seed, stride, log_every=5):
    random.seed(seed)
    np.random.seed(seed)

    rec = {k: [] for k in ("state", "lab", "q", "act", "bird", "is_death", "ep")}
    seen = set()
    g = 0
    t0 = time.time()
    ep_stats = []

    def push(ep, t, state, lab, q, a, bird, is_death):
        if (ep, t) in seen:
            return
        seen.add((ep, t))
        rec["state"].append(state.astype(np.uint8))
        rec["lab"].append(lab.astype(np.uint8))
        rec["q"].append(q.astype(np.float32))
        rec["act"].append(int(a))
        rec["bird"].append(np.array(bird, np.int32))
        rec["is_death"].append(bool(is_death))
        rec["ep"].append(int(ep))

    for ep in range(episodes):
        a0 = np.zeros(Config.NUM_ACTIONS)
        a0[0] = 1
        frame, _, term = env.frame_step(a0, render=False)
        state = init_state(frame)

        buf = []
        t = 0
        score = 0
        while not term and t < max_steps:
            lab = E.region_label(env)
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = net(x)[0].cpu().numpy()
            a = int(q.argmax())
            bird = (env.playerIndex, env.playery, env.playerx)

            buf.append((t, state, lab, q, a, bird))
            if len(buf) > DEATH_BACK + 1:
                buf.pop(0)
            if g % stride == 0:
                push(ep, t, state, lab, q, a, bird, False)
            g += 1

            av = np.zeros(Config.NUM_ACTIONS)
            av[a] = 1
            frame, r, term = env.frame_step(av, render=False)
            if r == 1.0:
                score += 1
            state = next_state(state, frame)
            t += 1

        if term:
            for (tt, ss, ll, qq, aa, bb) in buf:
                push(ep, tt, ss, ll, qq, aa, bb, True)

        n_d = len(buf) if term else 0
        ep_stats.append(dict(ep=ep, steps=t, score=score, died=bool(term)))
        if (ep + 1) % log_every == 0 or ep == episodes - 1:
            print(f"    [采样] {ep + 1}/{episodes} 局  累计 {g} 帧  "
                  f"抽样 {len(rec['state'])}  已用 {time.time() - t0:.0f}s")

    out = {k: np.array(v) for k, v in rec.items()}
    out["ep_stats"] = ep_stats
    out["total_frames"] = g
    print(f"    [采样完成] 总帧 {g}，抽样 {len(out['state'])} 帧"
          f"（其中死亡窗口 {int(out['is_death'].sum())}），用时 {time.time() - t0:.0f}s")
    return out

def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def _fwd(net, S, device, chunk=512):
    out = []
    for i in range(0, len(S), chunk):
        x = torch.as_tensor(S[i:i + chunk].astype(np.float32), device=device)
        with torch.no_grad():
            out.append(net(x).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2), np.float32)

def _fwd_masked(net, S, LAB, r, device, chunk=512):
    out = []
    for i in range(0, len(S), chunk):
        x = torch.as_tensor(S[i:i + chunk].astype(np.float32), device=device)
        m = torch.as_tensor(LAB[i:i + chunk] == r, device=device)
        with torch.no_grad():
            out.append(net(x * (~m).unsqueeze(1).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2), np.float32)

def ablation(net, S, LAB, Q, ACT, device):
    N = len(S)
    q0 = _fwd(net, S, device)
    a_sel = ACT.astype(np.int64)
    q_sel = q0[np.arange(len(q0)), a_sel]

    cnt = np.stack([(LAB == r).sum(axis=(1, 2)) for r in range(5)], axis=1)
    res = {}
    for r, name in enumerate(E.REGION_NAMES):
        present = cnt[:, r] > 0
        q1 = _fwd_masked(net, S, LAB, r, device)
        q1_sel = q1[np.arange(N), a_sel]
        impact = np.abs(q1_sel - q_sel) / np.abs(q_sel) * 100.0
        signed = (q_sel - q1_sel) / np.abs(q_sel) * 100.0
        flip = (q1.argmax(1) != a_sel).astype(float) * 100.0
        dq10 = (q1[:, 1] - q1[:, 0]) - (q0[:, 1] - q0[:, 0])
        p = present
        area = cnt[p, r].mean() / (Config.FRAME_SIZE ** 2)
        raw = float(impact[p].mean())
        res[name] = dict(
            n_frames=int(p.sum()),
            area_frac=float(area),
            impact_pct=raw, impact_ci=_boot_ci(impact[p]),
            unit_impact_pct=float(raw / (area * 100.0)) if area > 0 else float("nan"),
            unit_impact_ci=[float(v / (area * 100.0)) if area > 0 else float("nan")
                            for v in _boot_ci(impact[p])],
            drop_signed_pct=float(signed[p].mean()),
            flip_pct=float(flip[p].mean()), flip_ci=_boot_ci(flip[p]),
            dq_margin=float(dq10[p].mean()), dq_margin_ci=_boot_ci(dq10[p]),
        )
    return res, q0, q_sel

def equal_area(net, S, LAB, device, budgets=(180, 360, 720, 1080), n_seed=3):
    yy = np.arange(Config.FRAME_SIZE)[None, :]
    bandrow = yy >= A.GROUND_BAND
    white = S.astype(np.uint8).max(axis=1) > 0
    is_g = white & bandrow
    is_p = white & (~bandrow)

    with torch.no_grad():
        q0 = _fwd(net, S, device)
    d0 = q0[:, 1] - q0[:, 0]

    gcnt = is_g.reshape(len(S), -1).sum(1)
    pcnt = is_p.reshape(len(S), -1).sum(1)

    out = {}
    for B in budgets:
        idx = np.where((gcnt >= B) & (pcnt >= B))[0]
        if len(idx) < 30:
            out[str(B)] = dict(budget=int(B), n=int(len(idx)), ratio=float("nan"))
            continue
        gpos = [np.argwhere(is_g[i]) for i in idx]
        ppos = [np.argwhere(is_p[i]) for i in idx]
        acc = None
        for sd in range(n_seed):
            rng = np.random.default_rng(1000 + sd)
            SG, SP = S[idx].copy(), S[idx].copy()
            for k in range(len(idx)):
                g = gpos[k][rng.permutation(len(gpos[k]))[:B]]
                p = ppos[k][rng.permutation(len(ppos[k]))[:B]]
                SG[k][:, g[:, 0], g[:, 1]] = 0.0
                SP[k][:, p[:, 0], p[:, 1]] = 0.0
            qg, qp = _fwd(net, SG, device), _fwd(net, SP, device)
            mg = (qg[:, 1] - qg[:, 0]) - d0[idx]
            mp = (qp[:, 1] - qp[:, 0]) - d0[idx]
            cur = [np.abs(mg), mg, np.abs(qp[:, 1] - qp[:, 0] - d0[idx]), mp]
            acc = cur if acc is None else [acc[j] + cur[j] for j in range(4)]
        ag, mg, ap, mp = [v / n_seed for v in acc]
        out[str(B)] = dict(
            budget=int(B), n=int(len(idx)),
            ground_absmean_dmargin=float(ag.mean()), ground_ci=_boot_ci(ag),
            dyn_absmean_dmargin=float(ap.mean()), dyn_ci=_boot_ci(ap),
            ratio=float(ag.mean() / ap.mean()) if ap.mean() > 0 else float("nan"),
            ground_margin_shift=float(mg.mean()), ground_margin_ci=_boot_ci(mg),
            dyn_margin_shift=float(mp.mean()), dyn_margin_ci=_boot_ci(mp),
        )
    return out

def sky_control(net, S, ACT, device, max_frames=200):
    X = torch.as_tensor(S[:max_frames].astype(np.float32), device=device)
    a_sel = ACT[:max_frames].astype(np.int64)
    with torch.no_grad():
        q0 = net(X).cpu().numpy()
    B = 8
    n = Config.FRAME_SIZE // B
    blocks = []
    for i in range(min(len(S), max_frames)):
        for bi in range(n):
            for bj in range(n):
                sl = (slice(bi * B, (bi + 1) * B), slice(bj * B, (bj + 1) * B))
                if S[i][:, sl[0], sl[1]].max() == 0:
                    blocks.append((i, bi, bj))
    if not blocks:
        return dict(n_blocks=0, max_abs_dev=float("nan"))
    Xi = torch.as_tensor(np.stack([S[i] for i, _, _ in blocks]).astype(np.float32),
                         device=device)
    with torch.no_grad():
        qa = net(Xi).cpu().numpy()
    ai = np.array([a_sel[i] for i, _, _ in blocks])
    qb = q0[[i for i, _, _ in blocks], ai]
    return dict(n_blocks=len(blocks),
                max_abs_dev=float(np.max(np.abs(qa[np.arange(len(ai)), ai] - qb))))

def cf_bird(net, S, LAB, ACT, BIRD, device, ks=(1, 2, 3, 4, 6, 8), n_ctrl=1, seed=0):
    rng = np.random.default_rng(seed)
    N = len(S)
    lo = {k: [] for k in list(ks) + [-k for k in ks]}
    dz = {k: [] for k in list(ks) + [-k for k in ks]}
    kstar = []
    flip_any = np.zeros(N, bool)
    dirs = {"下移→拍翅": 0, "上移→拍翅": 0, "下移→不拍": 0, "上移→不拍": 0}
    ctrl_flip = []

    for i in range(N):
        pi, py, px = BIRD[i]
        mask = E._bird_cells(pi, py, px)
        oi = np.argwhere(mask)
        if len(oi) == 0:
            continue
        s_i = S[i].astype(np.float32)
        x0 = torch.as_tensor(s_i, device=device).unsqueeze(0)
        with torch.no_grad():
            q0 = net(x0).cpu().numpy()[0]
        a0 = int(ACT[i])
        base = s_i.copy()
        base[:, mask] = 0.0
        found = None

        for k in [kk for kk in sorted(set(list(ks) + [-kk for kk in ks]))]:
            si = oi + np.array([0, k])
            ok = ((si[:, 0] >= 0) & (si[:, 0] < Config.FRAME_SIZE) &
                  (si[:, 1] >= 0) & (si[:, 1] < Config.FRAME_SIZE))
            si = si[ok]
            src = oi[ok]
            if len(si) == 0:
                continue
            for c in range(Config.STACK_LEN):
                base[c, si[:, 0], si[:, 1]] = s_i[c, src[:, 0], src[:, 1]]
            with torch.no_grad():
                q1 = net(torch.as_tensor(base, device=device).unsqueeze(0)).cpu().numpy()[0]
            fl = 1.0 if int(q1.argmax()) != a0 else 0.0
            lo[k].append(fl)
            dz[k].append(float((q1[1] - q1[0]) - (q0[1] - q0[0])))
            if fl > 0 and found is None:
                found = k
            if fl > 0:
                flip_any[i] = True
                if int(q1.argmax()) == 1:
                    dirs["下移→拍翅" if k > 0 else "上移→拍翅"] += 1
                else:
                    dirs["下移→不拍" if k > 0 else "上移→不拍"] += 1
            base[:, si[:, 0], si[:, 1]] = 0.0

        kstar.append(found)

        if len(oi):
            row = []
            for _ in range(n_ctrl):
                si = np.stack([rng.integers(0, Config.FRAME_SIZE, size=len(oi)),
                               rng.integers(0, Config.FRAME_SIZE, size=len(oi))], 1)
                c = s_i.copy()
                c[:, mask] = 0.0
                for ch in range(Config.STACK_LEN):
                    c[ch, si[:, 0], si[:, 1]] = 255.0
                with torch.no_grad():
                    qc = net(torch.as_tensor(c, device=device).unsqueeze(0)).cpu().numpy()[0]
                row.append(1.0 if int(qc.argmax()) != a0 else 0.0)
            ctrl_flip.append(float(np.mean(row)))

    n_eval = len(kstar)
    kf = [abs(v) for v in kstar if v is not None]
    n_any = len(kf)
    n_within6 = sum(1 for v in kf if v <= 6)
    out = dict(
        n_frames=N, n_evaluated=n_eval,
        n_flipped_any=n_any,
        flip_rate_any=float(n_any / n_eval) if n_eval else float("nan"),
        n_flipped_within_k6=n_within6,
        rate_within_k6=float(n_within6 / n_eval) if n_eval else float("nan"),
        ctrl_rand_redraw_flip=float(np.mean(ctrl_flip)) if ctrl_flip else float("nan"),
        ctrl_ci=_boot_ci(np.array(ctrl_flip)) if ctrl_flip else [float("nan")] * 2,
        kstar_abs=kf,
        direction_counts=dirs,
        flip_by_k={str(k): dict(flip=float(np.mean(v)) if v else float("nan"),
                                n=len(v),
                                flip_ci=_boot_ci(np.array(v)) if v else [float("nan")] * 2,
                                dq_margin=float(np.mean(dz[k])) if dz[k] else float("nan"))
                   for k, v in lo.items()},
    )
    return out

def cf_blocks(net, S, ACT, device, cap=80, block=4, n_ctrl=3, seed=0):
    rng = np.random.default_rng(seed)
    N = len(S)
    ka, kr = [], []
    for i in range(N):
        s_i, a0 = S[i], int(ACT[i])
        attr = A.occlusion(net, s_i, a0, device, block=block)
        sc = np.abs(attr).reshape(Config.STACK_LEN,
                                 Config.FRAME_SIZE // block, block,
                                 Config.FRAME_SIZE // block, block).sum(axis=(0, 2, 4))
        order = np.dstack(np.unravel_index(np.argsort(-sc, axis=None), sc.shape))[0]
        ka.append(_need(net, s_i, a0, order, device, cap, block=block))
        rr = []
        for _ in range(n_ctrl):
            perm = order[rng.permutation(len(order))]
            rr.append(_need(net, s_i, a0, perm, device, cap, block=block))
        finite = [v for v in rr if np.isfinite(v)]
        kr.append(float(np.mean(finite)) if finite else float("nan"))
    ka, kr = np.array(ka, float), np.array(kr, float)
    ok = np.isfinite(ka) & np.isfinite(kr)
    d = (ka - kr)[ok]
    paired_idx = np.where(ok)[0]
    return dict(n=N, n_flip_attr=int(np.isfinite(ka).sum()),
                n_flip_rand=int(np.isfinite(kr).sum()),
                flip_rate_attr=float(np.isfinite(ka).mean()),
                flip_rate_rand=float(np.isfinite(kr).mean()),
                median_attr=float(np.nanmedian(ka)) if ok.any() else float("nan"),
                median_rand=float(np.nanmedian(kr)) if ok.any() else float("nan"),
                paired_n=int(ok.sum()),
                paired_median_attr=float(np.median(ka[ok])) if ok.any() else float("nan"),
                paired_median_rand=float(np.median(kr[ok])) if ok.any() else float("nan"),
                paired_diff_mean=float(d.mean()) if ok.any() else float("nan"),
                paired_diff_ci=_boot_ci(d) if ok.any() else [float("nan")] * 2,
                paired_idx=[int(v) for v in paired_idx])

def _need(net, s_i, a0, order, device, cap, block):
    x = torch.as_tensor(s_i.astype(np.float32), device=device).unsqueeze(0)
    used = 0
    cur = x.clone()
    for (bi, bj) in order[:cap]:
        cur[:, :, bi * block:(bi + 1) * block, bj * block:(bj + 1) * block] = 0.0
        used += 1
        with torch.no_grad():
            if int(net(cur)[0].argmax()) != a0:
                return float(used)
    return float("nan")

def fig_ablation(res, ea, path):
    names = list(E.REGION_NAMES)
    raw = np.array([res[n]["impact_pct"] for n in names])
    rc = np.array([res[n]["impact_ci"] for n in names]).T
    unit = np.array([res[n]["unit_impact_pct"] for n in names])
    uc = np.array([res[n]["unit_impact_ci"] for n in names]).T
    flip = np.array([res[n]["flip_pct"] for n in names])
    fc = np.array([res[n]["flip_ci"] for n in names]).T
    y = np.arange(len(names))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), layout="constrained",
                             gridspec_kw=dict(wspace=0.08))
    panels = (
        (raw, rc, "原始影响（未归一）", "抹除后 |ΔQ(s,a*)| 相对变化 (%)", "{:.1f}"),
        (unit, uc, "单位面积影响（已按面积归一）", "%Q 幅值 ／ %画面", "{:.2f}"),
        (flip, fc, "决策翻转率", "argmax 翻转率 (%)", "{:.1f}"),
    )
    for k, (a, panel) in enumerate(zip(axes, panels)):
        v, ci, ttl, xlab, fmt = panel
        a.barh(y, v, color=REGION_COLORS, height=0.62)
        a.errorbar(v, y, xerr=[np.maximum(v - ci[0], 0), np.maximum(ci[1] - v, 0)],
                   fmt="none", ecolor="#5F5E5A", elinewidth=1, capsize=2.5)
        for i, (yy, vv) in enumerate(zip(y, v)):
            a.annotate(fmt.format(vv), (max(vv, ci[1][i]), yy), textcoords="offset points",
                       xytext=(4, 0), va="center", fontsize=9)
        a.set_yticks(y)
        a.set_yticklabels(names if k == 0 else [], fontsize=10.5)
        a.set_xlim(0, v.max() * 1.26)
        a.set_xlabel(xlab, fontsize=9.5)
        a.set_title(ttl, fontsize=11)
        a.grid(axis="x", alpha=0.22)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axes[0].annotate("面积 22.5%，独大", (raw[0], y[0]), textcoords="offset points",
                     xytext=(-6, 14), ha="right", fontsize=8.6, color="#5F5E5A")
    axes[1].annotate("按面积归一后\n小鸟反超地面带", (unit[1], y[1]), textcoords="offset points",
                     xytext=(6, 16), ha="left", fontsize=8.6, color="#5F5E5A")

    keys = sorted(ea.keys(), key=lambda s: int(s))
    e = ea["720"] if "720" in ea else ea[keys[len(keys) // 2]]
    rr = sorted(ea[k]["ratio"] for k in keys)
    span = "%.2f~%.2f" % (rr[0], rr[-1])
    fig.supxlabel(
        "等面积对照（抹同样多白像素，只改抹在哪）：|Δ(Q1−Q0)| 地面带 %.2f vs 动态 %.2f"
        "（720px 比值 %.2f，各预算 %s）→ 幅值同量级、由面积定\n"
        "但方向由位置定：带符号 Δ(Q1−Q0) 地面带 %+.2f（把决策推向拍翅）"
        "vs 动态 %+.2f（近中性）"
        % (e["ground_absmean_dmargin"], e["dyn_absmean_dmargin"], e["ratio"], span,
           e["ground_margin_shift"], e["dyn_margin_shift"]),
        fontsize=8.6, color="#5F5E5A")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("    ✔ 已保存", os.path.relpath(path))

def _geo_panel(ax, sample_path=None, kmax=8):
    ax.set_title("① 几何：横向固定，只上下平移", fontsize=10.5, pad=6)
    n = Config.FRAME_SIZE
    L, py, px = None, 245, 57
    if sample_path and os.path.exists(sample_path):
        try:
            z = np.load(sample_path, allow_pickle=True)
            lab, bird = z["lab"], z["bird"]
            m = np.array([(lab[i] == 1).sum() >= 5 for i in range(len(lab))])
            idx = np.nonzero(m)[0]
            if len(idx):
                t = idx[np.argsort(bird[idx, 1])[len(idx) // 2]]
                L, py, px = lab[t], int(bird[t, 1]), int(bird[t, 2])
        except Exception:
            L = None
    if L is None:
        L = np.full((n, n), 4, np.uint8)
        L[:, 62:] = 0
        L[0:40, 30:44] = 2
        L[56:80, 30:44] = 3
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(REGION_COLORS)
    ax.imshow(L.T, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 5.5), cmap.N),
              interpolation="nearest", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#B4B2A9")

    bb = np.argwhere(L == 1)
    if len(bb):
        j0, j1 = bb[:, 1].min(), bb[:, 1].max()
        i0, i1 = bb[:, 0].min(), bb[:, 0].max()
        xc = (i0 + i1) / 2.0
        ax.add_patch(plt.Rectangle((i0 - 0.5, j0 - 0.5), i1 - i0 + 1, j1 - j0 + 1,
                                   fill=False, ec="k", lw=1.5))

        ax.annotate("", xy=(xc, j0 - 0.5 - kmax), xytext=(xc, j1 + 0.5 + kmax),
                    arrowprops=dict(arrowstyle="<->", color="#0B0B0B", lw=1.2))
        ax.text(xc + 1.5, j1 + 0.5 + kmax, "±k 格\n（±8 格内）", fontsize=8.4,
                color="#0B0B0B", va="center", ha="left")
        ax.text(xc + 1.5, j0 - 0.5, "k = 0 基线\n= 小鸟原位置", fontsize=8.4,
                color="#0B0B0B", va="center", ha="left")
    ax.text(0.5, 1.5, "↑ 天空（Y=0）", fontsize=8.4, color="#5F5E5A", va="top")
    ax.text(0.5, n - 1.5, "地面带（画面固定区域）", fontsize=8.4, color="#5F5E5A", va="bottom")
    ax.set_xlabel("小鸟横向固定在屏幕左 1/5（x≈57，全程不动）；\n"
                  "本实验只沿竖直方向把它整体平移 k 格",
                  fontsize=8.4, color="#5F5E5A", labelpad=4)

def fig_cf(cf, cb, path, sample_path=None):
    ks = sorted([int(k) for k in cf["flip_by_k"] if int(k) != 0], key=lambda v: v)
    fl = np.array([cf["flip_by_k"][str(k)]["flip"] * 100 for k in ks])
    lo = np.array([cf["flip_by_k"][str(k)]["flip_ci"][0] * 100 for k in ks])
    hi = np.array([cf["flip_by_k"][str(k)]["flip_ci"][1] * 100 for k in ks])
    dz = np.array([cf["flip_by_k"][str(k)]["dq_margin"] for k in ks])

    fig, (gx, ax, bx) = plt.subplots(
        1, 3, figsize=(14.6, 4.5), layout="constrained",
        gridspec_kw=dict(width_ratios=[0.92, 1.30, 0.92], wspace=0.16))

    _geo_panel(gx, sample_path)

    ctrl = cf["ctrl_rand_redraw_flip"] * 100
    yz = -np.array(ks)
    ax.errorbar(fl, yz, xerr=[np.maximum(fl - lo, 0), np.maximum(hi - fl, 0)], fmt="-o",
                ms=4.5, lw=1.9, color="#D85A30", capsize=3, label="决策翻转率")
    ax.axvline(ctrl, color="#888780", ls="--", lw=1.1, label="随机重绘对照 %.1f%%" % ctrl)
    ax.axhline(0, color="#5F5E5A", ls=":", lw=1.3)
    ax.set_ylim(-9.8, 9.8)
    ax.set_yticks(sorted(set([-8, -6, -4, -2, 0, 2, 4, 6, 8])))
    ax.set_xlim(0, max(19, fl.max() * 1.35))
    ax.set_xlabel("决策翻转率 (%)", color="#D85A30", fontsize=9.5)
    ax.set_ylabel("小鸟竖直平移（格）\n↑ 更靠天空（+）　　↓ 更靠地面（−）", fontsize=9.5)
    ax.tick_params(axis="x", colors="#D85A30")
    ax.text(max(19, fl.max() * 1.35) * 0.985, 0.6, "0 = 基线（小鸟原位置）",
            fontsize=8.2, color="#5F5E5A", ha="right", va="bottom")
    b2 = ax.twiny()
    b2.plot(dz, yz, "--s", ms=4, lw=1.5, color="#185FA5", label="Δ(Q1 − Q0)")
    b2.set_xlim(-2.2, 2.2)
    b2.axvline(0, color="#B4B2A9", lw=0.8)
    b2.set_xlabel("Δ(Q1 − Q0)　→ 正 = 更倾向拍翅", color="#185FA5", fontsize=9.5)
    b2.tick_params(axis="x", colors="#185FA5")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.6, frameon=False, loc="lower right")
    ax.set_title("② 平移响应：|k| ≤ 6 格内可翻转 %.1f%%（随机重绘等量白像素 %.1f%%）\n"
                 "只平移小鸟、管子不动，n=%d 帧"
                 % (cf["rate_within_k6"] * 100, cf["ctrl_rand_redraw_flip"] * 100,
                    cf["n_evaluated"]), fontsize=10.2)
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)

    vals = [cb["flip_rate_attr"] * 100, cb["flip_rate_rand"] * 100]
    bx.bar([0, 1], vals, color=["#D85A30", "#9AA0A6"], width=0.5)
    for x0, v in zip([0, 1], vals):
        bx.text(x0, v, "%.1f%%" % v, ha="center", va="bottom", fontsize=11)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["按归因序抹", "随机序抹"], fontsize=10.5)
    bx.set_xlim(-0.6, 1.6)
    bx.set_ylim(0, max(vals) * 1.30)
    bx.set_ylabel("固定预算内决策翻转率 (%)", fontsize=9.5)
    bx.set_xlabel("配对 n=%d：中位所需块数 归因 %.0f vs 随机 %.0f（配对差 %+.1f 块）"
                  % (cb["paired_n"], cb["paired_median_attr"], cb["paired_median_rand"],
                     cb["paired_diff_mean"]), fontsize=9)
    bx.set_title("反事实 ②：删块顺序（%d 帧）" % cb["n"], fontsize=10.5)
    bx.grid(axis="y", alpha=0.22)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("    ✔ 已保存", os.path.relpath(path))

def fig_migration(path, src=None):
    src = src or os.path.join(OUT, "dynamics.json")
    d = json.load(open(src, encoding="utf-8"))
    rec = d["records"]
    base = d.get("baseline", {})
    a_band = float(base.get("ground_band", 0.225))
    a_dyn = float(base.get("dynamic_obj", 0.225))
    step = np.array([r["step"] for r in rec], float) / 1e4
    band = np.array([r["share_band"] for r in rec], float) / a_band
    dyn = np.array([r["share_dyn"] for r in rec], float) / a_dyn
    rb = float(np.corrcoef(step, band)[0, 1])
    rd = float(np.corrcoef(step, dyn)[0, 1])

    cross = float("nan")
    sgn = np.sign(band - dyn)
    k = np.where(np.diff(sgn) != 0)[0]
    if len(k):
        i = int(k[0])
        t = (dyn[i] - band[i]) / ((band[i + 1] - band[i]) - (dyn[i + 1] - dyn[i]))
        cross = float(step[i] + t * (step[i + 1] - step[i]))

    fig, ax = plt.subplots(figsize=(6.8, 4.0), layout="constrained")
    ax.plot(step, band, "-o", ms=4, lw=1.8, color="#9AA0A6",
            label="地面带　ρ = %+.2f" % rb)
    ax.plot(step, dyn, "-s", ms=4, lw=1.8, color="#D85A30",
            label="动态物体（管子+小鸟）　ρ = %+.2f" % rd)
    ax.axhline(1.0, color="#888780", lw=1.1, ls="--")
    ax.annotate("%.0f万步：%.2f" % (step[-1], band[-1]), (step[-1], band[-1]),
                textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=9,
                color="#5F5E5A")
    ax.annotate("%.0f万步：%.2f" % (step[-1], dyn[-1]), (step[-1], dyn[-1]),
                textcoords="offset points", xytext=(-8, -14), ha="right", fontsize=9,
                color="#5F5E5A")
    if np.isfinite(cross):
        ax.annotate("约 %.0f 万步两线交叉" % cross, (cross, 1.6),
                    textcoords="offset points", xytext=(6, 10), fontsize=8.6,
                    color="#5F5E5A")
    ax.set_yscale("log")
    ax.set_xlabel("训练步数（万步）", fontsize=10)
    ax.set_ylabel("归因密度（单位面积归因）", fontsize=10)
    ax.set_ylim(0.15, 6.0)
    ax.legend(fontsize=9, frameon=False, loc="center right")
    ax.grid(alpha=0.20, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("    ✔ 已保存", os.path.relpath(path),
          f"（端点 地面带 {band[0]:.2f}→{band[-1]:.2f}，"
          f"动态 {dyn[0]:.2f}→{dyn[-1]:.2f}，交叉 ≈ {cross:.0f} 万步）")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("runs", "ckpt-1890000.pt"))
    ap.add_argument("--episodes", type=int, default=45)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=16, help="抹除测试的抽帧步长")
    ap.add_argument("--cf-frames", type=int, default=300, help="反事实用的帧数")
    ap.add_argument("--ea-frames", type=int, default=1200, help="等面积对照用的帧数")
    ap.add_argument("--quick", action="store_true", help="小样本自检")
    ap.add_argument("--tag", default="")
    ap.add_argument("--figs-only", action="store_true", help="不重算，只按 cf_regions.json 重绘三张图")
    ap.add_argument("--no-cache", action="store_true", help="忽略采样缓存，强制重采样")
    args = ap.parse_args()

    if args.quick:
        args.episodes, args.stride, args.cf_frames, args.ea_frames = 3, 4, 40, 120
        args.tag = args.tag or "_quick"

    if args.figs_only:
        d = json.load(open(os.path.join(OUT, "cf_regions%s.json" % args.tag), encoding="utf-8"))
        fig_ablation(d["ablation"], d["equal_area"],
                     os.path.join(OUT, "figF_region_ablation%s.png" % args.tag))
        fig_cf(d["cf_bird"], d["cf_blocks"],
               os.path.join(OUT, "figG_counterfactual%s.png" % args.tag),
               sample_path=os.path.join(OUT, "cf_sample%s.npz" % args.tag))
        fig_migration(os.path.join(OUT, "figH_attention_migration%s.png" % args.tag))
        return

    device = torch.device("cpu")
    t0 = time.time()
    net, path = E.load_model(args.ckpt, device)
    print(f"[载入] {os.path.relpath(path)}")
    env = E.make_env()

    sp = os.path.join(OUT, "cf_sample%s.npz" % args.tag)
    rec = None
    if os.path.exists(sp) and not args.no_cache:
        z = np.load(sp, allow_pickle=True)
        if (int(z["episodes"]) == args.episodes and int(z["seed"]) == args.seed
                and int(z["stride"]) == args.stride and int(z["max_steps"]) == args.max_steps):
            rec = {k: z[k] for k in ("state", "lab", "q", "act", "bird", "is_death", "ep")}
            rec["ep_stats"] = list(z["ep_stats"])
            rec["total_frames"] = int(z["total_frames"])
            print(f"[1/5] 复用采样缓存 {os.path.relpath(sp)}（{len(rec['state'])} 帧）")
    if rec is None:
        print("\n[1/5] 采样（与相位层同源 seed=%d）…" % args.seed)
        rec = sample(net, env, device, args.episodes, args.max_steps, args.seed, args.stride)
        np.savez_compressed(sp, **rec, episodes=args.episodes, seed=args.seed,
                            stride=args.stride, max_steps=args.max_steps)
        print("    ✔ 采样已缓存", os.path.relpath(sp))
    S, LAB, Q, ACT, BIRD = (rec["state"], rec["lab"], rec["q"], rec["act"], rec["bird"])
    print(f"    死亡窗口帧 {int(rec['is_death'].sum())}；死亡局数 "
          f"{sum(1 for e in rec['ep_stats'] if e['died'])}")

    print("\n[2/5] 五区抹除…")
    abl, q0, q_sel = ablation(net, S, LAB, Q, ACT, device)
    for n in E.REGION_NAMES:
        d = abl[n]
        print(f"    {n:<5} n={d['n_frames']:<5} 面积{d['area_frac']*100:5.2f}%  "
              f"原始幅值 {d['impact_pct']:6.2f}%  单位面积 {d['unit_impact_pct']:6.2f}  "
              f"翻转 {d['flip_pct']:5.1f}%  Δmargin {d['dq_margin']:+.3f}")

    print("\n[3/5] 等面积对照 + 天空阴性对照…")
    idx = np.arange(len(S))
    step = max(1, len(S) // args.ea_frames)
    ei = idx[::step]
    ea = equal_area(net, S[ei], LAB[ei], device)
    sky = sky_control(net, S, ACT, device)
    for k in sorted(ea.keys(), key=lambda s: int(s)):
        d = ea[k]
        print(f"    等面积 {d['budget']:>4}px (n={d['n']:<5})：抹地面带 |Δmargin|="
              f"{d['ground_absmean_dmargin']:.3f} vs 抹动态区 {d['dyn_absmean_dmargin']:.3f}"
              f"（比值 {d['ratio']:.3f}）；Δmargin {d['ground_margin_shift']:+.3f} vs "
              f"{d['dyn_margin_shift']:+.3f}")
    print(f"    天空阴性对照：{sky['n_blocks']} 块，最大偏差 {sky['max_abs_dev']:.3e}")

    print("\n[4/5] 反事实…")
    ci = idx[::max(1, len(S) // args.cf_frames)]
    cf = cf_bird(net, S[ci], LAB[ci], ACT[ci], BIRD[ci], device)
    cb = cf_blocks(net, S[ci], ACT[ci], device)
    print(f"    CF-2 小鸟平移：可平移 {cf['n_evaluated']} 帧，|k|≤6 内可翻转 "
          f"{cf['rate_within_k6']:.1%}（任意 |k|≤8 为 {cf['flip_rate_any']:.1%}），"
          f"随机重绘对照 {cf['ctrl_rand_redraw_flip']:.1%}")
    print(f"    CF-1 最小删块：归因序翻转 {cb['flip_rate_attr']:.1%} vs 随机 "
          f"{cb['flip_rate_rand']:.1%}；配对 n={cb['paired_n']} 差 "
          f"{cb['paired_diff_mean']:.1f}")

    print("\n[5/5] 出图 / 落盘…")
    report = dict(
        ckpt=path, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
        stride=args.stride, cf_frames=len(ci), ea_frames=len(ei),
        total_frames=int(rec["total_frames"]), n_sampled=int(len(S)),
        n_death_window=int(rec["is_death"].sum()),
        n_death_episodes=int(sum(1 for e in rec["ep_stats"] if e["died"])),
        region_names=list(E.REGION_NAMES),
        ablation=abl, equal_area=ea, sky_control=sky,
        cf_bird=cf, cf_blocks=cb,
        seconds=round(time.time() - t0, 1),
    )
    jp = os.path.join(OUT, "cf_regions%s.json" % args.tag)
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    print("    ✔ 已保存", os.path.relpath(jp))

    fig_ablation(abl, ea, os.path.join(OUT, "figF_region_ablation%s.png" % args.tag))
    fig_cf(cf, cb, os.path.join(OUT, "figG_counterfactual%s.png" % args.tag),
           sample_path=sp)
    fig_migration(os.path.join(OUT, "figH_attention_migration%s.png" % args.tag))
    print(f"\n全部完成，用时 {time.time() - t0:.0f}s。")

if __name__ == "__main__":
    main()
