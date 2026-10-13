import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "dqn"))

from explain_dqn import make_env, load_model, collect
import attribution as A

PROBE = os.path.join(OUT, "probe.npz")
JSON = os.path.join(OUT, "dynamics.json")
MAPS = os.path.join(OUT, "attr_maps.npz")
EVAL = os.path.join(ROOT, "results", "dqn.json")

def build_probe(ckpt, device, episodes, max_steps, size, seed):
    net, path = load_model(ckpt, device)
    env = make_env()
    states, actions, qs, _moments, n_generic, scores = collect(
        net, env, device, episodes, max_steps, size, seed)

    st = states[:n_generic]
    np.savez_compressed(PROBE, states=st, actions=actions[:n_generic],
                        qs=qs[:n_generic], scores=np.array(scores))
    obj = st > 0
    dyn = obj.copy()
    dyn[:, :, :, A.GROUND_BAND:] = False
    print(f"    探针集：{len(st)} 个状态（来自 {episodes} 局，成绩 {scores}）")
    print(f"    基线① 全物体像素占比 = {obj.mean():.3f}"
          f"   基线② 动态物体(管子+小鸟)占比 = {dyn.mean():.3f}")
    print(f"    ↑ 「接地率」必须和这两个数比：只有显著高于 ② 才算真的在盯会动的东西")
    print(f"    已保存 {os.path.relpath(PROBE, ROOT)}")
    return st

def scan_one(net, states, device, sg_n=8, block=8):
    S = states
    x = torch.as_tensor(S, dtype=torch.float32, device=device)
    with torch.no_grad():
        Q = net(x).cpu().numpy()
    act = Q.argmax(axis=1)

    oc = A.occlusion_batch(net, S, act, device, block=block)
    sal = A.saliency_batch(net, S, act, device)
    sg = A.smoothgrad_batch(net, S, act, device, n=sg_n)

    aoc = np.abs(oc)
    ent, gini = A.concentration(aoc)
    m_all, _p_all, m_dyn, _p_dyn = A.grounding(aoc, S)
    cr, cc = A.centroid(aoc)

    band = A.ground_band_mask(S)
    share_band = float((aoc * band).sum() / (aoc.sum() + 1e-12))
    share_dyn = float(m_dyn)
    share_sky = float(1.0 - share_band - share_dyn)

    a_sal = np.abs(sal)
    _m1, _p1, gd_sal, gp_sal = A.grounding(a_sal, S)
    ent_sal = float(A.concentration(a_sal)[0].mean())

    row = np.arange(len(Q))
    m = {

        "q_chosen": float(Q[row, act].mean()),
        "q_max": float(Q.max(axis=1).mean()),
        "q_margin": float(np.abs(Q[:, 0] - Q[:, 1]).mean()),
        "flap_rate": float((act == 1).mean()),

        "frame_sal": A.frame_importance(a_sal).tolist(),
        "frame_sg": A.frame_importance(np.abs(sg)).tolist(),

        "entropy": float(ent.mean()),
        "gini": float(gini.mean()),
        "centroid_r": float(cr.mean()),
        "centroid_c": float(cc.mean()),
        "attr_abs": float(aoc.mean()),

        "share_band": share_band,
        "share_dyn": share_dyn,
        "share_sky": share_sky,

        "entropy_sal": ent_sal,
        "share_dyn_sal": float(gd_sal),
        "prec_dyn_sal": float(gp_sal),
    }
    maps = {"oc": aoc.mean(axis=0).astype(np.float32),
            "oc_signed": oc.mean(axis=0).astype(np.float32),
            "sal": a_sal.mean(axis=0).astype(np.float32)}
    return m, maps

def load_records():
    with open(EVAL, encoding="utf-8") as f:
        recs = json.load(f)["records"]
    return [{"step": r["step"], "ckpt": r["checkpoint"], "score": r["mean"]} for r in recs]

def scan(device, only_steps, sg_n, seed, block=8):
    if not os.path.exists(PROBE):
        raise FileNotFoundError("还没建探针集，先跑 --mode probe")
    probe = np.load(PROBE)
    states = probe["states"]

    records = load_records()
    if only_steps:
        records = [r for r in records if r["step"] in only_steps]

    done, maps = {}, {}
    if os.path.exists(JSON):
        with open(JSON, encoding="utf-8") as f:
            done = {r["step"]: r for r in json.load(f)["records"]}
    if os.path.exists(MAPS):
        z = np.load(MAPS)
        maps = {}
        for k in z.files:
            s, name = k.split("_", 1)
            maps.setdefault(int(s), {})[name] = z[k]

    todo = [r for r in records if r["step"] not in done]
    print(f"    待扫描 {len(todo)} 个存档（已完成 {len(done)} 个）")

    for i, r in enumerate(todo, 1):
        t0 = time.time()
        net, _ = load_model(r["ckpt"], device)
        m, mp = scan_one(net, states, device, sg_n, block)
        m.update(step=r["step"], score=float(r["score"]),
                 seconds=round(time.time() - t0, 1))
        done[r["step"]] = m
        maps[r["step"]] = mp
        print(f"    [{i:>2}/{len(todo)}] {r['step']:>8} 步  分数 {r['score']:>6.1f}  "
              f"Q={m['q_chosen']:>6.2f}  熵={m['entropy']:.4f}  "
              f"地面带={m['share_band']:>5.1%}  管子={m['share_dyn']:>5.1%}  "
              f"用时 {m['seconds']}s")

        _save(done, maps, states)

    print(f"    扫描完成，结果在 {os.path.relpath(JSON, ROOT)}")
    return done, maps

def _save(done, maps, states):
    ordered = sorted(done.values(), key=lambda r: r["step"])
    obj = states > 0
    with open(JSON, "w", encoding="utf-8") as f:
        json.dump({"n_probe": int(len(states)),

                   "baseline": {"ground_band": 0.225,
                                "dynamic_obj": 0.225,
                                "sky": 0.550,
                                "all_object": float(obj.mean())},
                   "ground_band_from": A.GROUND_BAND,
                   "method_primary": "Occlusion(block=8, 因果)",
                   "records": ordered},
                  f, ensure_ascii=False, indent=2)
    flat = {}
    for step, mp in maps.items():
        for k, v in mp.items():
            flat[f"{step}_{k}"] = v
    np.savez_compressed(MAPS, **flat)

def main():
    ap = argparse.ArgumentParser(description="跨训练阶段的归因扫描")
    ap.add_argument("--mode", choices=["probe", "scan", "all"], default="all")
    ap.add_argument("--ckpt", default=os.path.join("runs", "ckpt-1890000.pt"),
                    help="建探针集用的存档（默认最佳存档）")
    ap.add_argument("--episodes", type=int, default=5, help="建探针集时跑几局")
    ap.add_argument("--max-steps", type=int, default=2500, help="每局最多多少步")
    ap.add_argument("--probe-size", type=int, default=400, help="探针集大小")
    ap.add_argument("--sg-samples", type=int, default=8, help="SmoothGrad 噪声副本数")
    ap.add_argument("--block", type=int, default=8, help="遮挡法的块边长（80/8=10，共 100 块）")
    ap.add_argument("--steps", default="", help="只扫这些步数（逗号分隔），空=全部 40 个")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.chdir(ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    only = {int(s) for s in args.steps.split(",") if s.strip()} or None
    t0 = time.time()

    print("=" * 72)
    print("跨训练阶段归因扫描")
    print("=" * 72)
    if args.mode in ("probe", "all"):
        print("\n[1/2] 建固定探针集（用最佳存档跑几局，冻结画面）")
        build_probe(args.ckpt, device, args.episodes, args.max_steps,
                    args.probe_size, args.seed)
    if args.mode in ("scan", "all"):
        print("\n[2/2] 扫描各存档")
        scan(device, only, args.sg_samples, args.seed, args.block)
    print(f"\n总用时 {time.time() - t0:.0f} 秒。")

if __name__ == "__main__":
    main()
