"""Fit offline linear-leaf Q surrogates and inspect feature/block sensitivity."""
import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
import tree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'dqn'))
sys.path.insert(0, str(ROOT/'env'))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
from model import QNet
from utils import init_state, next_state

def node_influence(fitted, features, targets, selected, global_indices, dimension=25600):
    """Node indices MUST refer to the same rows supplied to tree.fit.

    A private copy is used so explanatory node regressions do not overwrite the
    predictive leaf regressions. The caller passes the actual fit subset.
    """
    if len(features) != len(targets):
        raise ValueError('Feature/target rows differ')
    root_ids = fitted['nodes'][fitted['root']]['idx']
    if not np.array_equal(root_ids, np.arange(len(targets))):
        raise ValueError('Pass exactly the rows used to fit this tree')
    selected = np.asarray(selected, dtype=int)
    pos = {int(f): k for k, f in enumerate(selected)}
    influence = np.zeros(dimension, dtype=float)
    for node in fitted['nodes']:
        if node['j'] is None: continue
        ids, feature = node['idx'], int(node['j'])
        mask = features[ids, feature] > .5
        left, right = ids[~mask], ids[mask]
        if not len(left) or not len(right): continue
        gain = targets[ids].var() - (len(left)*targets[left].var()+len(right)*targets[right].var())/len(ids)
        weights, _, _, _ = tree.ridge(features[ids][:, selected], targets[ids], 100.)
        denom = float(weights @ weights)
        correction = float(weights[pos[feature]]**2/denom) if feature in pos and denom > 0 else 0.
        influence[global_indices[feature]] += (1+correction)*max(float(gain), 0.)
    return influence.reshape(4, 80, 80)

def path_features(fitted, sample):
    node_id, result = fitted['root'], []
    while fitted['nodes'][node_id]['j'] is not None:
        node = fitted['nodes'][node_id]; f = int(node['j'])
        result.append(f); node_id = node['r'] if sample[f] > .5 else node['l']
    return result

def qvalues(net, states, device):
    with torch.inference_mode():
        return net(torch.as_tensor(states, dtype=torch.float32, device=device)).cpu().numpy()

def collect(net, device, episodes, first_seed, max_steps):
    import wrapped_flappy_bird as game
    frames, values, episode_ids = [], [], []
    for ep in range(episodes):
        random.seed(first_seed+ep); np.random.seed(first_seed+ep)
        game.PLAYER_INDEX_GEN = itertools.cycle([0, 1, 2, 1])
        env = game.GameState()
        frame, _, done = env.frame_step([1, 0], render=False)
        state = init_state(frame)
        for step in range(max_steps):
            if done: break
            q = qvalues(net, state[None], device)[0]
            if step % 4 == 0:
                frames.append(state.astype(np.uint8)); values.append(q); episode_ids.append(first_seed+ep)
            action = np.eye(2)[int(q.argmax())]
            frame, _, done = env.frame_step(action, render=False)
            state = next_state(state, frame)
    return np.asarray(frames), np.asarray(values), np.asarray(episode_ids)

def block_maps(net, inputs, influence, device):
    inputs = inputs.astype(np.float32)
    baseline = qvalues(net, inputs, device)[:, 1]
    sensitivity, importance = np.zeros((4, 10, 10)), np.zeros((4, 10, 10))
    for channel in range(4):
        for x in range(10):
            for y in range(10):
                xs, ys = slice(8*x, 8*x+8), slice(8*y, 8*y+8)
                masked = inputs.copy(); masked[:, channel, xs, ys] = 0
                sensitivity[channel,x,y] = np.mean(np.abs(qvalues(net, masked, device)[:,1]-baseline))
                importance[channel,x,y] = influence[channel,xs,ys].sum()
    return importance, sensitivity

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', default='runs/ckpt-1890000.pt')
    p.add_argument('--data', help='Optional previously collected NPZ; requires Xtr,Qtr,Xte,Qte,eid_tr,eid_te')
    p.add_argument('--out', default='interpretability/intrinsic/out')
    p.add_argument('--max-steps', type=int, default=100000)
    p.add_argument('--occlusion-frames', type=int, default=120)
    args=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=QNet(2).to(device)
    net.load_state_dict(torch.load(args.ckpt,map_location=device,weights_only=False)['policy']);net.eval()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.data:
        z=np.load(args.data);Xtr,Qtr,Xte,Qte,eid_tr,eid_te=[z[k] for k in ['Xtr','Qtr','Xte','Qte','eid_tr','eid_te']]
    else:
        Xtr,Qtr,eid_tr=collect(net,device,10,1000,args.max_steps)
        Xte,Qte,eid_te=collect(net,device,4,5000,args.max_steps)
        np.savez_compressed(out/'data.npz',Xtr=Xtr,Qtr=Qtr,Xte=Xte,Qte=Qte,eid_tr=eid_tr,eid_te=eid_te)
    assert not set(eid_tr)&set(eid_te), 'Train/test episodes overlap'
    binary=(Xtr.reshape(len(Xtr),-1)>.5).astype(np.float32)
    order=np.random.default_rng(0).permutation(len(binary)); fit_ids=order[len(order)//5:]
    freq=binary[fit_ids].mean(0); keep=(freq>=.01)&(freq<=.99); global_indices=np.flatnonzero(keep)
    # All node-local row indices are defined on these arrays, not the full pool.
    fit_x,fit_q=binary[fit_ids][:,keep],Qtr[fit_ids]
    test_x=(Xte.reshape(len(Xte),-1)>.5).astype(np.float32)[:,keep]
    fitted=[tree.fit(fit_x,fit_q[:,a],'full') for a in (0,1)]
    prediction=np.column_stack([tree.predict(test_x,t) for t in fitted])
    influence=node_influence(fitted[1],fit_x,fit_q[:,1],tree.screening(fit_x,fit_q[:,1])[:1024],global_indices)
    threshold=float(influence.sum()/len(global_indices))
    paths=[path_features(fitted[1],test_x[i]) for i in range(min(2,len(test_x)))]
    ids=np.random.default_rng(0).choice(len(Xte),min(args.occlusion_frames,len(Xte)),replace=False)
    importance,sensitivity=block_maps(net,Xte[ids],influence,device)
    np.savez_compressed(out/'explanations.npz',influence=influence,block_importance=importance,block_sensitivity=sensitivity,test_frame_indices=ids)
    summary={'indexing':'all node statistics use the exact fit subset','train_frames':len(Xtr),'fit_frames':len(fit_x),'test_frames':len(Xte),'features':int(keep.sum()),'leaves':[t['leaves'] for t in fitted],'test_action_agreement':float((prediction.argmax(1)==Qte.argmax(1)).mean()),'test_Q_MAE':np.mean(np.abs(prediction-Qte),axis=0).tolist(),'highlight_threshold':threshold,'paths_global_feature_indices':[[int(global_indices[j]) for j in path] for path in paths]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for c in range(4):
        for row,values,label in [(0,importance,'Surrogate influence'),(1,sensitivity,'Teacher sensitivity')]:
            im=axs[row,c].imshow(values[c].T,origin='upper');axs[row,c].set_title(f'{label}: frame {c}');fig.colorbar(im,ax=axs[row,c],shrink=.65)
    fig.savefig(out/'block_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
