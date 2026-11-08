"""Exercise actual revision rollouts, cost export and adaptation controls."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from metrics.binary_refresh_evaluator import BinaryRefreshEvaluator
from metrics.refresh_comparison import RefreshComparisonEvaluator, rollout
from metrics.trajectory_revision import revision_metrics
from model import CARSDirve


class RevisionExportTests(unittest.TestCase):
    def test_revision_exports_costs_history_and_frozen_base_controls(self):
        self._check_exports('plan1')

    def test_quality_exports_continuous_metrics_and_frozen_base_controls(self):
        self._check_exports('plan2')

    def _check_exports(self, mode):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = CARSDirve(mode=mode, plan_horizon=10, refresh_version=3, architecture_version=2,
                           feature_dim=4, hidden_dim=8, heads=2, layers=1,
                           spatial_grid=(3, 3), dropout=0).eval()
        batch = dict(bev_features=torch.randn(1, 41, 4, 6, 6),
                     bev_valid_mask=torch.ones(1, 41, 6, 6, dtype=torch.bool),
                     timestamps_us=torch.arange(41)[None] * 100000,
                     planning_targets=torch.zeros(1, 10, 10, 3),
                     planning_mask=torch.ones(1, 10, 10, dtype=torch.bool),
                     action_history_sequence=torch.zeros(1, 11, 20, 3),
                     anchor_index=torch.tensor([25]), recording=['recording'], event_type=['event'], anchor_timestamp_us=torch.tensor([1]))
        dataset = SimpleNamespace(samples=[Path('/tmp/events/event.pt')], event_dir=Path('/tmp/events'),
                                  _processed_file_names=['event.pt'], processed_file_names_path='val.pt')
        with tempfile.TemporaryDirectory(prefix='fresh-revision-export-', dir='/tmp') as directory:
            config = dict(trainer=dict(ckpt_path='/tmp/revision.ckpt'), evaluation=dict(output_dir=directory))
            trainer = pl.Trainer(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
                                 enable_progress_bar=False, enable_model_summary=False)
            evaluator = BinaryRefreshEvaluator(model, dataset, config, 'test.yaml', 1)
            trainer.validate(evaluator, DataLoader([batch], batch_size=None), verbose=False)
            report = json.loads((Path(directory) / 'metrics.json').read_text())
            with (Path(directory) / 'per_decision.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(int(r['online_decode_calls']) for r in rows), 100)
            self.assertEqual(sum(int(r['initial_decode_calls']) for r in rows), 10)
            self.assertEqual(sum(int(r['total_online_decode_calls']) for r in rows), 110)
            self.assertEqual(sum(int(r['diagnostic_decode_calls']) for r in rows), 100)
            self.assertGreater(report['history_preferences']['valid_pairs'], 0)
            self.assertLessEqual(report['history_preferences']['valid_pairs'], 45)
            self.assertIn('total_candidate_gain_m', rows[0])
            self.assertIn('fixed10_candidate_ade_m', rows[0])
            if mode == 'plan2':
                self.assertIn('quality_l1', rows[0])
                self.assertIn('gain_l1', rows[0])
                self.assertNotIn('probability', rows[0])
                self.assertNotIn('replacement_ce', rows[0])
                self.assertNotIn('pr_auc_ap', report['global_metrics']['selection'])
            else:
                self.assertIn('replacement_ce', rows[0])
                self.assertNotIn('quality_l1', rows[0])
            with torch.no_grad():
                retained = rollout(model, batch, 'no_refresh')
                scores = revision_metrics(retained, batch['planning_targets'], batch['planning_mask'])
            self.assertFalse(retained['replace_decision'].any())
            self.assertEqual(float(scores['realized_gain']), 0.)
            self.assertEqual(float(scores['harmful_replacements']), 0.)
            self.assertEqual(report['protocol']['refresh_version'], 3)
            comparison = RefreshComparisonEvaluator(model, dataset, config, 'test.yaml', 1)
            trainer.validate(comparison, DataLoader([batch], batch_size=None), verbose=False)
            report = json.loads((Path(directory) / 'metrics.json').read_text())
            self.assertIn('all_fresh_base', report['policies'])
            self.assertIn('no_refresh_base', report['policies'])
            self.assertEqual(report['policies']['all_fresh']['metrics']['total_online_decode_calls'], 110)
            self.assertEqual(report['policies']['replan_only']['metrics']['total_online_decode_calls'], 210)
        with patch.object(model.cars_adapter, 'adapt', side_effect=AssertionError('base must bypass adaptation')):
            base = rollout(model, batch, 'all_fresh_base')
        self.assertTrue(base['candidate_generated'].all())


if __name__ == '__main__':
    unittest.main()
