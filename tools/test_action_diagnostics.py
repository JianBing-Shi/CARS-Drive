"""Regression tests for causal interventions and fixed-context evaluation."""
import unittest
import torch
from metrics.action_diagnostics import replay
from model.CARSDirve import CARSDirve


class ActionDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1024)
        self.model=CARSDirve(history_step=3,future_step=5,feature_dim=4,hidden_dim=8,
                heads=2,layers=1,spatial_grid=(3,3),dropout=0).eval()
        features=torch.randn(2,9,4,6,6)
        poses=torch.eye(4,dtype=torch.float64).repeat(2,9,1,1)
        poses[:,:,0,3]=torch.arange(9)*.1
        self.inputs=dict(bev_features=features,bev_valid_mask=torch.ones(2,9,6,6,dtype=torch.bool),
                         timestamps_us=torch.arange(9)[None].repeat(2,1)*100000+1620000000000000,
                         reference_ego_to_global=poses,bev_range=torch.tensor([-5,-5,-1,5,5,1.]))

    @torch.no_grad()
    def test_fresh_matches_production_and_fixed5_ignores_future(self):
        torch.testing.assert_close(replay(self.model,self.inputs),self.model(**self.inputs))
        expected=replay(self.model,self.inputs,fixed_horizon=5)
        changed={**self.inputs,'bev_features':self.inputs['bev_features'].clone()}
        changed['bev_features'][:,4:]+=100
        torch.testing.assert_close(expected,replay(self.model,changed,fixed_horizon=5))

    @torch.no_grad()
    def test_no_vision_and_delayed_input_boundaries(self):
        changed={**self.inputs,'bev_features':self.inputs['bev_features']*100+50}
        torch.testing.assert_close(replay(self.model,self.inputs,vision='none'),
                                   replay(self.model,changed,vision='none'))
        delayed=replay(self.model,self.inputs,lag=2)
        changed={**self.inputs,'bev_features':self.inputs['bev_features'].clone()}
        changed['bev_features'][:,2:]+=100
        torch.testing.assert_close(delayed[:,:1],replay(self.model,changed,lag=2)[:,:1])
        teacher=torch.randn(2,5,3)
        first=replay(self.model,self.inputs,teacher_actions=teacher)
        teacher[:,2:]+=100
        torch.testing.assert_close(first[:,:3],replay(self.model,self.inputs,teacher_actions=teacher)[:,:3])

    @torch.no_grad()
    def test_history_boundary_and_age_controls_are_causal(self):
        changed={**self.inputs,'bev_features':self.inputs['bev_features'].clone()}
        changed['bev_features'][:,:2]+=100
        torch.testing.assert_close(replay(self.model,self.inputs,history_skip=2),
                                   replay(self.model,changed,history_skip=2))
        expected=replay(self.model,self.inputs,lag=2,zero_age=True,fixed_horizon=5)
        changed={**self.inputs,'bev_features':self.inputs['bev_features'].clone()}
        changed['bev_features'][:,2:]+=100
        torch.testing.assert_close(expected,replay(self.model,changed,lag=2,zero_age=True,fixed_horizon=5))


if __name__=='__main__':
    unittest.main()
