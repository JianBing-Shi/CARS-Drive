"""Production parity, causality and common-support error checks."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from metrics.refresh_comparison import RefreshComparisonEvaluator, rollout, window_errors
from model.FRESHDrive import FRESHDrive


class RefreshComparisonTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1024)
        self.model = FRESHDrive(mode='refresh', feature_dim=4, hidden_dim=8,
                                heads=2, layers=1, spatial_grid=(3, 3), dropout=0).eval()
        self.batch = dict(bev_features=torch.randn(2, 31, 4, 6, 6),
                          bev_valid_mask=torch.ones(2, 31, 6, 6, dtype=torch.bool),
                          timestamps_us=torch.arange(31)[None].expand(2, -1) * 100_000 + 1620000000000000)
        poses = torch.eye(4, dtype=torch.float64).repeat(2, 31, 1, 1)
        poses[:, :, 0, 3] = torch.arange(31) * .1
        self.batch.update(reference_ego_to_global=poses,
                          bev_range=torch.tensor([-5, -5, -1, 5, 5, 1.]))

    @torch.no_grad()
    def test_production_parity_and_causality(self):
        actual = rollout(self.model, self.batch, 'learned')
        expected = self.model(**self.batch, return_details=True)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        changed = dict(self.batch)
        changed['bev_features'] = self.batch['bev_features'].clone()
        changed['bev_features'][:, 21:] += 1000  # future of first five decisions
        later = rollout(self.model, changed, 'learned')
        torch.testing.assert_close(actual['planned_trajectory'][:, :5], later['planned_trajectory'][:, :5])
        none = rollout(self.model, self.batch, 'no_refresh')
        changed['bev_features'][:, 16:] += 1000
        torch.testing.assert_close(none['planned_trajectory'],
                                   rollout(self.model, changed, 'no_refresh')['planned_trajectory'])
        self.assertFalse(none['refresh_mask'].any())
        self.assertEqual(none['forced_replan'][0].nonzero().flatten().tolist(), [4, 9])
        self.assertEqual(none['plan_mask'][0].sum(-1).tolist(), [4, 3, 2, 1, 5, 4, 3, 2, 1, 5])
        replan = rollout(self.model, self.batch, 'replan_only')
        self.assertTrue(replan['plan_mask'].all())
        self.assertFalse(replan['refresh_mask'].any())
        self.assertTrue(rollout(self.model, self.batch, 'all_fresh')['refresh_mask'].all())

    def test_errors_ignore_padding_and_missing(self):
        prediction = torch.zeros(2, 2, 3, 3)
        prediction[0, :, :, 0] = torch.tensor([[1, 2, 999], [3, 999, 999]])
        mask = torch.tensor([[[1, 1, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0]]]).bool()
        scores = window_errors(prediction, torch.zeros_like(prediction), mask)
        self.assertEqual(scores['ade_m'][0], 2)
        self.assertEqual(scores['fde_m'][0], 2.5)
        self.assertTrue(torch.isnan(scores['ade_m'][1]))

    def test_lightning_export(self):
        batch = dict(self.batch, planning_targets=torch.zeros(2, 10, 5, 3),
                     planning_mask=torch.ones(2, 10, 5, dtype=torch.bool),
                     recording=['log1', 'log2'], event_type=['turn', 'stop'],
                     anchor_timestamp_us=torch.tensor([1, 2]))
        dataset = SimpleNamespace(_processed_file_names=['a.pt', 'b.pt'], processed_file_names_path='val.pt')
        with tempfile.TemporaryDirectory() as directory:
            config = {'trainer': {'ckpt_path': '/tmp/test.ckpt'}, 'evaluation': {'output_dir': directory}}
            evaluator = RefreshComparisonEvaluator(self.model, dataset, config, 'test.yaml', 2)
            pl.Trainer(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False).validate(
                           evaluator, DataLoader([batch], batch_size=None), verbose=False)
            report = json.loads((Path(directory) / 'metrics.json').read_text())
            self.assertEqual(report['policies']['learned']['samples'], 2)
            self.assertEqual(report['policies']['learned']['metrics']['common_endpoints'], 30)
            self.assertTrue((Path(directory) / 'per_sample.csv').exists())


if __name__ == '__main__':
    unittest.main()
