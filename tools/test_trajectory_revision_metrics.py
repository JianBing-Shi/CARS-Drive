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
from model import CARSDirve


class RevisionExportTests(unittest.TestCase):
    def test_revision_exports_costs_history_and_frozen_base_controls(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = CARSDirve(mode='plan', refresh_version=3, architecture_version=2,
                           feature_dim=4, hidden_dim=8, heads=2, layers=1,
                           spatial_grid=(3, 3), dropout=0).eval()
        batch = dict(bev_features=torch.randn(1, 31, 4, 6, 6),
                     bev_valid_mask=torch.ones(1, 31, 6, 6, dtype=torch.bool),
                     timestamps_us=torch.arange(31)[None] * 100000,
                     planning_targets=torch.zeros(1, 10, 5, 3),
                     planning_mask=torch.ones(1, 10, 5, dtype=torch.bool),
                     recording=['recording'], event_type=['event'], anchor_timestamp_us=torch.tensor([1]))
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
            self.assertEqual(sum(int(r['online_decode_calls']) for r in rows), 50)
            self.assertEqual(sum(int(r['initial_decode_calls']) for r in rows), 5)
            self.assertEqual(sum(int(r['total_online_decode_calls']) for r in rows), 55)
            self.assertEqual(sum(int(r['diagnostic_decode_calls']) for r in rows), 50)
            self.assertEqual(report['history_preferences']['valid_pairs'], 34)
            self.assertIn('total_candidate_gain_m', rows[0])
            self.assertEqual(report['protocol']['refresh_version'], 3)
            comparison = RefreshComparisonEvaluator(model, dataset, config, 'test.yaml', 1)
            trainer.validate(comparison, DataLoader([batch], batch_size=None), verbose=False)
            report = json.loads((Path(directory) / 'metrics.json').read_text())
            self.assertIn('all_fresh_base', report['policies'])
            self.assertIn('no_refresh_base', report['policies'])
            self.assertEqual(report['policies']['all_fresh']['metrics']['total_online_decode_calls'], 55)
            self.assertEqual(report['policies']['replan_only']['metrics']['total_online_decode_calls'], 105)
        with patch.object(model.cars_adapter, 'adapt', side_effect=AssertionError('base must bypass adaptation')):
            base = rollout(model, batch, 'all_fresh_base')
        self.assertTrue(base['candidate_generated'].all())


if __name__ == '__main__':
    unittest.main()
