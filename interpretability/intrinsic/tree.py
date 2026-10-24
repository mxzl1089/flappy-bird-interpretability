"""Offline linear-leaf surrogate; extracted from the original implementation."""
import numpy as np
LAM = 100.0
DEPTH = 10
MIN_CHILD = 8

def ridge(F, t, lam, iters=80, tol=1e-8):
    n, d = F.shape
    mu, tm = F.mean(0), t.mean()
    Fc, tc = F - mu, t - tm
    bb = Fc.T @ tc
    pre = 1.0 / np.maximum((Fc * Fc).sum(0) + lam, 1e-12)
    w = np.zeros(d)
    r = bb.copy(); z = pre * r; p = z.copy(); rz = r @ z
    nb = float(bb @ bb) + 1e-30
    k = 0
    for k in range(iters):
        Ap = Fc.T @ (Fc @ p) + lam * p
        den = float(p @ Ap)
        if den <= 0:
            break
        a = rz / den
        w += a * p; r -= a * Ap
        if float(r @ r) < tol * nb:
            break
        z = pre * r; rz2 = float(r @ z)
        p = z + (rz2 / rz) * p; rz = rz2
    return w, tm - mu @ w, k + 1, float(np.sqrt((r @ r) / nb))

def best_split(F, t, min_child):
    n = len(t)
    if n < 2 * min_child:
        return None, None
    n1 = F.sum(0).astype(np.float64)
    s1 = (F.T @ t).astype(np.float64)
    q1 = (F.T @ (t * t)).astype(np.float64)
    n0 = n - n1
    s0 = float(t.sum()) - s1
    q0 = float((t * t).sum()) - q1
    ok = (n1 >= min_child) & (n0 >= min_child)
    if not ok.any():
        return None, None
    sse = np.where(ok, (q1 - s1 * s1 / np.maximum(n1, 1)) + (q0 - s0 * s0 / np.maximum(n0, 1)),
                   np.inf)
    j = int(np.argmin(sse))
    if not np.isfinite(sse[j]):
        return None, None
    return j, F[:, j] > 0.5

def build_path(F, t, idx, nodes, path, depth, max_depth, min_child):
    ni = len(nodes)
    nodes.append({"idx": idx, "j": None, "l": None, "r": None, "path": tuple(path)})
    if depth >= max_depth or len(idx) < 2 * min_child:
        return ni
    j, m = best_split(F[idx], t[idx], min_child)
    if j is None:
        return ni
    nodes[ni]["j"] = j
    nodes[ni]["l"] = build_path(F, t, idx[~m], nodes, path + [j], depth + 1, max_depth, min_child)
    nodes[ni]["r"] = build_path(F, t, idx[m], nodes, path + [j], depth + 1, max_depth, min_child)
    return ni

def fit(F, t, mode, feat_global=None):
    nodes = []
    root = build_path(F, t, np.arange(len(t)), nodes, [], 0, DEPTH, MIN_CHILD)
    for nd in nodes:
        if nd["j"] is not None:
            continue
        i = nd["idx"]
        if mode == "full":
            feat = None
        elif mode == "topk":
            feat = feat_global
        else:
            feat = np.asarray(sorted(set(nd["path"])), dtype=np.int64)
        nd["feat"] = feat
        if feat is not None and len(feat) == 0:
            nd["feat"], nd["w"], nd["b"] = None, None, float(t[i].mean())
            continue
        Fi = F[i] if feat is None else F[i][:, feat]
        nd["w"], nd["b"], _, _ = ridge(Fi, t[i], LAM)
    return {"nodes": nodes, "root": root,
            "leaves": sum(1 for n in nodes if n["j"] is None),
            "feat_used": sorted({int(f) for n in nodes if n["j"] is None and n["feat"] is not None
                                 for f in n["feat"]})}

def predict(F, tree):
    nodes, out = tree["nodes"], np.zeros(len(F))
    stack = [(tree["root"], np.arange(len(F)))]
    while stack:
        ni, idx = stack.pop()
        nd = nodes[ni]
        if nd["j"] is None:
            if nd["w"] is None:
                out[idx] = nd["b"]
            else:
                Fi = F[idx] if nd["feat"] is None else F[idx][:, nd["feat"]]
                out[idx] = Fi @ nd["w"] + nd["b"]
        else:
            m = F[idx, nd["j"]] > 0.5
            stack.append((nd["l"], idx[~m]))
            stack.append((nd["r"], idx[m]))
    return out

def screening(F, t):
    n = len(t)
    n1 = F.sum(0).astype(np.float64)
    s1 = (F.T @ t).astype(np.float64)
    ok = (n1 >= MIN_CHILD) & ((n - n1) >= MIN_CHILD)
    gain = np.where(ok, s1 * s1 / np.maximum(n1, 1) + (t.sum() - s1) ** 2 / np.maximum(n - n1, 1),
                    -np.inf)
    return np.argsort(-gain)
