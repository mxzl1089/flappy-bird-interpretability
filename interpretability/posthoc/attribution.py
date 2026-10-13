import numpy as np
import torch
import torch.nn.functional as F

def _t(state, device):
    return torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

def _q(net, x, a):
    return net(x)[:, a]

def _in_grad(net, x, a):
    x = x.detach().requires_grad_(True)
    net.zero_grad(set_to_none=True)
    _q(net, x, a).sum().backward()
    return x.grad

def _unfold(x, b):
    return F.unfold(x, kernel_size=b, stride=b)

def _fold_blocks(vals, b, shape):
    N, NB = vals.shape
    C, H, W = shape
    v = vals.unsqueeze(1).expand(N, C * b * b, NB)
    return F.fold(v, output_size=(H, W), kernel_size=b, stride=b)

def _mask_forward(net, xb, z, b, shape, a):
    M = z.shape[0]
    v = xb.expand(M, -1, -1) * z[:, None, :]
    xm = F.fold(v, output_size=shape[1:], kernel_size=b, stride=b)
    with torch.no_grad():
        return _q(net, xm, a)

def saliency(net, state, action, device):
    x = _t(state, device)
    return _in_grad(net, x, action)[0].abs().cpu().numpy()

def smoothgrad(net, state, action, device, n=16, flip=0.005, seed=0, chunk=16):
    x0 = _t(state, device)
    g = torch.Generator(device=device).manual_seed(seed)
    acc = torch.zeros_like(x0)
    done = 0
    while done < n:
        k = min(chunk, n - done)
        probs = torch.rand((k,) + tuple(x0.shape[1:]), generator=g, device=device)
        mask = (probs < flip).float()
        xn = x0 * (1.0 - mask) + (255.0 - x0) * mask
        acc += _in_grad(net, xn, action).abs().sum(0, keepdim=True)
        done += k
    return (acc / n)[0].cpu().numpy()

def integrated_gradients(net, state, action, device, steps=32, baseline=0.0):
    x0 = _t(state, device)
    base = torch.full_like(x0, float(baseline))
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1, 1)
    xs = base + alphas * (x0 - base)
    grads = _in_grad(net, xs, action)
    return ((x0 - base)[0] * grads.mean(0)).cpu().numpy()

def occlusion(net, state, action, device, block=8):
    x = _t(state, device)
    shape = tuple(x.shape[1:])
    xb = _unfold(x, block)
    NB = xb.shape[-1]
    z = 1.0 - torch.eye(NB, device=device)
    with torch.no_grad():
        q0 = _q(net, x, action).item()
    drop = q0 - _mask_forward(net, xb, z, block, shape, action)
    return _fold_blocks(drop.unsqueeze(0), block, shape)[0].cpu().numpy()

def _kernel_weight(d, k):
    k = k.double()
    logC = (torch.lgamma(torch.tensor(d + 1.0, device=k.device))
            - torch.lgamma(k + 1) - torch.lgamma(d - k + 1))
    logw = np.log(d - 1) - torch.log(k.clamp(min=1)) - torch.log((d - k).clamp(min=1)) - logC
    return torch.where((k > 0) & (k < d), logw.exp(), torch.zeros_like(logw)).float()

def kernelshap(net, state, action, device, block=8, samples=2048, seed=0,
               chunk=256, verbose=True):
    x = _t(state, device)
    shape = tuple(x.shape[1:])
    xb = _unfold(x, block)
    NB = xb.shape[-1]

    rng = torch.Generator().manual_seed(seed)
    k = torch.randint(1, NB, (samples,), generator=rng)
    z = torch.zeros(samples, NB)
    for i, ki in enumerate(k.tolist()):
        z[i, torch.randperm(NB, generator=rng)[:ki]] = 1.0

    z = torch.cat([torch.zeros(1, NB), torch.ones(1, NB), z]).to(device)

    y = torch.cat([_mask_forward(net, xb, z[s:s + chunk], block, shape, action)
                   for s in range(0, z.shape[0], chunk)])
    phi0 = y[0].item()

    w = _kernel_weight(NB, z.sum(1))
    sw = w.sqrt()
    phi = torch.linalg.lstsq(sw[:, None] * z,
                             (sw * (y - phi0)).unsqueeze(1)).solution.squeeze(1)

    if verbose:
        gap = abs(phi.sum().item() + phi0 - y[1].item()) / max(abs(y[1].item()), 1e-6)
        print(f"    [SHAP 自检] 完备性误差 {gap * 100:.2f}%  "
              f"(Q全集={y[1].item():.2f}, Q空集={phi0:.2f})")
    return _fold_blocks(phi.view(1, NB), block, shape)[0].cpu().numpy()

METHODS = {
    "Saliency": saliency,
    "SmoothGrad": smoothgrad,
    "IG": integrated_gradients,
    "Occlusion": occlusion,
    "KernelSHAP": kernelshap,
}

def saliency_batch(net, states, actions, device):
    x = torch.as_tensor(states, dtype=torch.float32, device=device).requires_grad_(True)
    a = torch.as_tensor(actions, device=device).view(-1, 1)
    net.zero_grad(set_to_none=True)
    net(x).gather(1, a).sum().backward()
    return x.grad.abs().cpu().numpy()

def occlusion_batch(net, states, actions, device, block=8):
    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    N = x.shape[0]
    shape = tuple(x.shape[1:])
    xb = _unfold(x, block)
    NB = xb.shape[-1]
    a = torch.as_tensor(actions, device=device).view(-1, 1)
    with torch.no_grad():
        q0 = net(x).gather(1, a).squeeze(1)
    drop = torch.zeros(N, NB, device=device)
    buf = xb.clone()
    for i in range(NB):
        buf[:, :, i] = 0.0
        xm = F.fold(buf, output_size=shape[1:], kernel_size=block, stride=block)
        with torch.no_grad():
            drop[:, i] = q0 - net(xm).gather(1, a).squeeze(1)
        buf[:, :, i] = xb[:, :, i]
    return _fold_blocks(drop, block, shape).cpu().numpy()

def smoothgrad_batch(net, states, actions, device, n=8, flip=0.005, seed=0):
    x0 = torch.as_tensor(states, dtype=torch.float32, device=device)
    a = torch.as_tensor(actions, device=device).view(-1, 1)
    g = torch.Generator(device=device).manual_seed(seed)
    acc = torch.zeros_like(x0)
    for _ in range(n):
        mask = (torch.rand(x0.shape, generator=g, device=device) < flip).float()
        xn = (x0 * (1.0 - mask) + (255.0 - x0) * mask).requires_grad_(True)
        net.zero_grad(set_to_none=True)
        net(xn).gather(1, a).sum().backward()
        acc += xn.grad.abs()
    return (acc / n).cpu().numpy()

def concentration(attr):
    p = np.abs(attr).reshape(len(attr), -1)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(p.shape[1])
    sp = np.sort(p, axis=1)
    n = sp.shape[1]
    idx = np.arange(1, n + 1)
    gini = (2 * (sp * idx).sum(axis=1) - (n + 1) * sp.sum(axis=1)) / (n * sp.sum(axis=1) + 1e-12)
    return ent, gini

GROUND_BAND = 62

def ground_band_mask(states, band=GROUND_BAND):
    m = np.zeros_like(states, dtype=bool)
    m[:, :, :, band:] = True
    return m

def sky_mask(states, band=GROUND_BAND):
    return states <= 0

def grounding(attr, states, band=GROUND_BAND):
    a = np.abs(attr)
    obj = states > 0
    dyn = obj.copy()
    dyn[:, :, :, band:] = False

    def _stat(mask):
        mass = (a * mask).sum() / (a.sum() + 1e-12)
        thr = np.percentile(a.reshape(len(a), -1), 99, axis=1)
        top = a > thr[:, None, None, None]
        prec = (top & mask).sum() / (top.sum() + 1e-12)
        return float(mass), float(prec)

    m_all, p_all = _stat(obj)
    m_dyn, p_dyn = _stat(dyn)
    return m_all, p_all, m_dyn, p_dyn

def centroid(attr):
    m = np.abs(attr).sum(axis=1)
    tot = m.sum(axis=(1, 2)) + 1e-12
    r = (m * np.arange(m.shape[1])[None, :, None]).sum(axis=(1, 2)) / tot
    c = (m * np.arange(m.shape[2])[None, None, :]).sum(axis=(1, 2)) / tot
    return r, c

def frame_importance(attr):
    v = np.abs(attr).sum(axis=(0, 2, 3))
    return v / (v.sum() + 1e-12) * 100.0

def _batch_q(net, flat, actions, device, shape):
    x = torch.as_tensor(flat.reshape(-1, *shape), dtype=torch.float32, device=device)
    a = torch.as_tensor(actions, device=device).view(-1, 1)
    with torch.no_grad():
        return net(x).gather(1, a).squeeze(1).cpu().numpy()

def deletion_curve(net, states, actions, attrs, device, steps=10, n_random=3, seed=0,
                   mode="percent"):
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
        m = flat.copy()
        m[rows, order[:, :k]] = 0.0
        cur[i, 0] = _batch_q(net, m, actions, device, shape)
        for s in range(n_random):
            rng = np.random.default_rng(seed + s)
            mr = flat.copy()
            mr[rows, rng.permutation(NPIX)[:k]] = 0.0
            cur[i, 1] += _batch_q(net, mr, actions, device, shape) / n_random

    out = np.zeros((len(fracs), 2))
    for j in range(2):
        if mode == "percent":

            base = cur[0, j]
            out[:, j] = (cur[:, j] / np.where(np.abs(base) < 1e-6, 1.0, base)
                         * 100).mean(axis=1)
        else:
            full = cur[-1, j]
            span = cur[0, j] - full
            good = np.abs(span) > 1e-3
            rel = np.full_like(cur[:, j], np.nan)
            if good.any():
                rel[:, good] = ((cur[:, j][:, good] - full[good]) / span[good])
            out[:, j] = np.nanmean(rel, axis=1)
    return fracs, out[:, 0], out[:, 1]
