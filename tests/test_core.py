import importlib.util
import os
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'interpretability/intrinsic'))
import tree
import surrogate
sys.path.insert(0,str(ROOT/'interpretability/posthoc'))
import position_intervention as position

class NumericalChecks(unittest.TestCase):
    def test_ridge_matches_closed_form(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(40,5));y=rng.normal(size=40)
        w,b,_,_=tree.ridge(x,y,3.,tol=1e-14)
        centered=x-x.mean(0);expected=np.linalg.solve(centered.T@centered+3*np.eye(5),centered.T@(y-y.mean()))
        np.testing.assert_allclose(w,expected,atol=1e-6)
        self.assertAlmostEqual(b,float(y.mean()-x.mean(0)@w),places=7)

    def test_node_statistics_use_fit_subset(self):
        # A shuffled strict subset: indexing the original pool would be wrong.
        full=np.array([[0],[1],[0],[1],[0],[1]],dtype=float)
        target=np.array([100,-100,1,5,1,5],dtype=float)
        ids=np.array([5,2,3,4]);x,y=full[ids],target[ids]
        fitted={'root':0,'nodes':[{'idx':np.arange(4),'j':0}]}
        g=surrogate.node_influence(fitted,x,y,[0],[0])
        # Var(1,5)=4, child variance 0, one nonzero weight gives factor 2.
        self.assertAlmostEqual(float(g.sum()),8.,places=6)
        with self.assertRaisesRegex(ValueError,'exactly'):
            surrogate.node_influence(fitted,full,target,[0],[0])

    def test_zero_variance_is_valid(self):
        x=np.array([[0],[0],[1],[1]],dtype=float)
        fitted={'root':0,'nodes':[{'idx':np.arange(4),'j':0}]}
        self.assertEqual(float(surrogate.node_influence(fitted,x,np.ones(4),[0],[0]).sum()),0.)

    def test_tree_prediction_and_path(self):
        x=np.repeat([[0.,0.],[1.,1.]],20,axis=0);y=x[:,0]*4+1
        fitted=tree.fit(x,y,'full')
        np.testing.assert_allclose(tree.predict(x,fitted),y,atol=1e-6)
        self.assertTrue(surrogate.path_features(fitted,x[0]))

    def test_rerender_identity_and_history(self):
        import wrapped_flappy_bird as game
        from utils import init_state,preprocess_frame
        env=game.GameState();frame,_,_=env.frame_step([1,0],render=False)
        before=(env.playerx,env.playery,env.score)
        np.testing.assert_array_equal(position.render_shift(game,env),frame)
        self.assertEqual(before,(env.playerx,env.playery,env.score))
        state=init_state(frame);changed=state.copy()
        changed[0]=preprocess_frame(position.render_shift(game,env,dy=24))
        np.testing.assert_array_equal(changed[1:],state[1:])
        self.assertFalse(np.array_equal(changed[0],state[0]))

if __name__=='__main__': unittest.main()
