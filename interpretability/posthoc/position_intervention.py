"""Latest-frame rerendering intervention; a portable implementation of the report protocol."""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'dqn'));sys.path.insert(0,str(ROOT/'env'))
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
from model import QNet
from utils import init_state, next_state, preprocess_frame

def render_shift(game,env,dx=0,dy=0):
    """Redraw without advancing physics, RNG, animation, or the environment."""
    import pygame
    screen=pygame.Surface((game.SCREENWIDTH,game.SCREENHEIGHT))
    screen.blit(game.IMAGES['background'],(0,0))
    for upper,lower in zip(env.upperPipes,env.lowerPipes):
        screen.blit(game.IMAGES['pipe'][0],(upper['x'],upper['y']))
        screen.blit(game.IMAGES['pipe'][1],(lower['x'],lower['y']))
    screen.blit(game.IMAGES['base'],(env.basex,game.BASEY))
    screen.blit(game.IMAGES['player'][env.playerIndex],(env.playerx+dx,env.playery+dy))
    return pygame.surfarray.array3d(screen)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',required=True);p.add_argument('--episodes',type=int,default=45)
    p.add_argument('--max-steps',type=int,default=2500);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--out',default='interpretability/posthoc/out/position.jsonl')
    args=p.parse_args()
    import wrapped_flappy_bird as game
    device='cuda' if torch.cuda.is_available() else 'cpu';net=QNet(2).to(device)
    net.load_state_dict(torch.load(args.ckpt,map_location=device,weights_only=False)['policy']);net.eval()
    def qs(s):
        with torch.inference_mode():return net(torch.as_tensor(s,dtype=torch.float32,device=device)).cpu().numpy()
    shifts=[(0,d) for d in (-72,-48,-24,24,48,72)]+[(d,0) for d in (-24,-6,6,24)]
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    counts={'frames':0,'eligible_frames':0,'zero_shift_checks':0,'coordinate':'screen y positive downward','implementation':'portable rerender, latest channel only'}
    with out.open('w',encoding='utf-8') as f:
        for ep in range(args.episodes):
            random.seed(args.seed+ep);np.random.seed(args.seed+ep);game.PLAYER_INDEX_GEN=itertools.cycle([0,1,2,1])
            env=game.GameState();frame,_,done=env.frame_step([1,0],render=False);state=init_state(frame)
            for step in range(args.max_steps):
                if done:break
                q=qs(state[None])[0];margin=float(q[1]-q[0]);counts['frames']+=1
                if abs(margin)<args.threshold:
                    assert np.array_equal(render_shift(game,env),frame),'Zero displacement does not preserve pixels'
                    counts['zero_shift_checks']+=1;counts['eligible_frames']+=1
                    width=game.IMAGES['player'][env.playerIndex].get_width()
                    in_pipe=any(env.playerx+width>u['x'] and env.playerx<u['x']+game.PIPE_WIDTH for u in env.upperPipes)
                    altered=[]
                    for dx,dy in shifts:
                        s=state.copy();s[0]=preprocess_frame(render_shift(game,env,dx,dy));altered.append(s)
                    newq=qs(np.asarray(altered))
                    for (dx,dy),vq in zip(shifts,newq):
                        row={'episode':ep,'frame':step,'in_pipe':bool(in_pipe),'dx':dx,'dy':dy,'margin':margin,'delta_margin':float(vq[1]-vq[0]-margin),'flipped':bool(vq.argmax()!=q.argmax())}
                        f.write(json.dumps(row)+'\n')
                frame,_,done=env.frame_step(np.eye(2)[int(q.argmax())],render=False);state=next_state(state,frame)
    out.with_suffix('.meta.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(json.dumps(counts))

if __name__=='__main__':main()
