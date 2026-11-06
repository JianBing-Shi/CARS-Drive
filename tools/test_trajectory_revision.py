"""Revision gradients, causal candidate comparison and frozen action weights."""
import tempfile
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import lightning.pytorch as pl
import torch
import yaml
from torch.utils.data import DataLoader

from metrics.trajectory_revision import revision_metrics
from model import FRESHDrive


class TrajectoryRevisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = FRESHDrive(
            mode='refresh', refresh_version=3, architecture_version=2,
            feature_dim=4, hidden_dim=8, heads=2, layers=1,
            spatial_grid=(3, 3), dropout=0, refresh_monitor_frames=3)
        self.inputs = dict(
            bev_features=torch.randn(2, 31, 4, 6, 6),
            timestamps_us=torch.arange(31)[None].expand(2, -1) * 100000)
        self.targets = dict(planning_targets=torch.zeros(2, 10, 5, 3),
                            planning_mask=torch.ones(2, 10, 5, dtype=torch.bool))

    @staticmethod
    def keep_logits(context, *args, **kwargs):
        return context.new_tensor([10., -10.]).expand(context.shape[0], -1)

    def losses(self, output):
        return revision_metrics(output, self.targets['planning_targets'], self.targets['planning_mask'])

    def test_expert_l1_updates_adapter_through_frozen_decoder(self):
        self.model.train()
        before = {name: value.clone() for name, value in self.model.state_dict().items()
                  if name.startswith(('tokenizer.', 'memory.', 'decoder.'))}
        output = self.model(**self.inputs, **self.targets, return_details=True)
        loss = self.losses(output)['trajectory_l1']
        self.assertTrue(loss.requires_grad)
        loss.backward()
        adapter_gradients = [p.grad for p in self.model.refresh.revision.parameters()]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in adapter_gradients))
        for part in (self.model.tokenizer, self.model.memory, self.model.decoder):
            self.assertFalse(part.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in part.parameters()))
        optimizers, _ = self.model.configure_optimizers()
        optimizers[0].step()
        for name, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]), name)
        optimized = {id(parameter) for group in optimizers[0].param_groups for parameter in group['params']}
        self.assertTrue(all(name.startswith('refresh.') for name, parameter in self.model.named_parameters()
                            if id(parameter) in optimized))

    def test_every_decision_generates_even_rejected_candidates(self):
        self.model.eval()
        with patch.object(self.model.refresh, 'compare', side_effect=self.keep_logits):
            with patch.object(self.model.decoder, 'forward', wraps=self.model.decoder.forward) as decoder:
                output = self.model(**self.inputs, return_details=True)
        self.assertEqual(decoder.call_count, 55)  # initial five, ten five-step candidates
        self.assertTrue(output['candidate_valid'].all())
        self.assertEqual(tuple(output['candidate_trajectory'].shape), (2, 10, 5, 3))
        self.assertFalse(output['event_refresh'].any())
        self.assertEqual(output['deadline_replan'][0].nonzero().flatten().tolist(), [4, 9])
        self.assertEqual(output['online_decode_calls'][0].sum().item(), 50)

    def test_future_gt_and_future_observations_do_not_change_current_predictions(self):
        self.model.eval()
        output = self.model(**self.inputs, **self.targets, return_details=True)
        other_targets = dict(self.targets, planning_targets=torch.randn(2, 10, 5, 3) * 10)
        changed_gt = self.model(**self.inputs, **other_targets, return_details=True)
        for key in ('candidate_trajectory', 'refresh_logits', 'planned_trajectory'):
            torch.testing.assert_close(output[key], changed_gt[key])
        masked_targets = dict(self.targets, planning_mask=self.targets['planning_mask'].clone())
        masked_targets['planning_mask'][..., 2:] = False
        changed_mask = self.model(**self.inputs, **masked_targets, return_details=True)
        for key in ('candidate_trajectory', 'refresh_logits', 'planned_trajectory'):
            torch.testing.assert_close(output[key], changed_mask[key])
        changed_inputs = dict(self.inputs, bev_features=self.inputs['bev_features'].clone())
        changed_inputs['bev_features'][:, 21:] += 100
        changed_future = self.model(**changed_inputs, **self.targets, return_details=True)
        for key in ('candidate_trajectory', 'refresh_logits', 'planned_trajectory'):
            torch.testing.assert_close(output[key][:, :5], changed_future[key][:, :5])
        unlabeled = self.model(**self.inputs, return_details=True)
        for key in ('candidate_trajectory', 'refresh_logits', 'planned_trajectory'):
            torch.testing.assert_close(output[key], unlabeled[key])

    def test_history_only_compares_surviving_common_future(self):
        self.model.eval()
        with patch.object(self.model.refresh, 'compare', side_effect=self.keep_logits):
            output = self.model(**self.inputs, **self.targets, return_details=True)
        history_valid = output['history_valid']
        self.assertEqual(tuple(history_valid.shape), (2, 10, 4))
        # The initial plan and earlier candidates enter history while future points survive.
        self.assertEqual(history_valid[0].sum(-1).tolist(), [1, 2, 3, 4, 4, 4, 4, 4, 4, 4])
        if 'history_common_mask' in output:
            lengths = output['history_common_mask'][0, 4].sum(-1).tolist()
            self.assertEqual(sorted(lengths), [1, 2, 3, 4])
        self.assertFalse(output['decision_valid'][:, [4, 9]].any())
        self.assertTrue(self.losses(output)['history_valid_pairs'] > 0)

    def test_missing_candidates_preserve_history_lag_and_deadline_costs(self):
        self.model.eval()
        valid = torch.ones(2, 31, 6, 6, dtype=torch.bool)
        valid[:, 17] = False
        with patch.object(self.model.refresh, 'compare', side_effect=self.keep_logits):
            output = self.model(**self.inputs, bev_valid_mask=valid, **self.targets, return_details=True)
        self.assertEqual(output['history_valid'][0, 3].tolist(), [True, False, True, True])
        self.assertEqual(output['history_common_mask'][0, 3].sum(-1).tolist(), [4, 0, 2, 1])
        self.model.refresh_budget = 0
        with patch.object(self.model.decoder, 'forward', wraps=self.model.decoder.forward) as decoder:
            output = self.model(**self.inputs, return_details=True)
        self.assertEqual(decoder.call_count, 15)  # initial plus two stale fallback plans
        self.assertFalse(output['candidate_generated'].any())
        self.assertEqual(output['online_decode_calls'][0].sum().item(), 10)

    def test_invalid_observation_and_target_masks_remain_finite(self):
        self.model.refresh_budget = 0
        self.model.eval()
        invalid = dict(self.inputs, bev_valid_mask=torch.zeros(2, 31, 6, 6, dtype=torch.bool))
        targets = dict(self.targets, planning_mask=torch.zeros(2, 10, 5, dtype=torch.bool),
                       planning_targets=torch.full((2, 10, 5, 3), float('nan')))
        output = self.model(**invalid, **targets, return_details=True)
        losses = revision_metrics(output, targets['planning_targets'], targets['planning_mask'])
        self.assertFalse(output['candidate_valid'].any())
        self.assertFalse(output['decision_valid'].any())
        self.assertFalse(output['history_valid'].any())
        self.assertEqual(losses['loss'].item(), 0)
        self.assertTrue(losses['loss'].requires_grad)
        losses['loss'].backward()
        self.assertTrue(any(p.grad is not None for p in self.model.refresh.parameters()))
        self.assertTrue(torch.isfinite(output['planned_trajectory']).all())

    def test_actual_error_can_prefer_old_plan_and_comparison_is_symmetric(self):
        zero = torch.zeros(2, 5, 3)
        poor_candidate = zero.clone()
        poor_candidate[..., 0] = .4
        common = torch.ones(2, 5, dtype=torch.bool)
        labels, valid, gain = self.model.refresh.replacement_labels(
            zero, poor_candidate, zero, common, torch.ones(2, dtype=torch.bool),
            0., 0., torch.ones(2))
        self.assertTrue(valid.all())
        self.assertFalse(labels.any())
        self.assertTrue((gain < 0).all())
        context = torch.randn(2, 8)
        logits = self.model.refresh.compare(context, zero, poor_candidate, common)
        reversed_logits = self.model.refresh.compare(context, poor_candidate, zero, common)
        torch.testing.assert_close(logits, reversed_logits.flip(-1))
        tied = self.model.refresh.compare(context, zero, zero, common)
        torch.testing.assert_close(tied[:, 0], tied[:, 1])

    def test_current_visual_change_and_old_plan_affect_monitor_context(self):
        self.model.eval()
        state = dict(memory=torch.randn(2, 9, 8),
                     valid=torch.ones(2, 9, dtype=torch.bool), age_s=torch.zeros(2, 9))
        monitor = torch.randn(2, 3, 9, 8)
        token_valid = state['valid']
        arguments = dict(state=state, tokens=monitor[:, -1], token_valid=token_valid,
                         monitor_tokens=monitor, monitor_valid=token_valid[:, None].expand(-1, 3, -1),
                         monitor_lags=torch.tensor([[.2, .1, 0.], [.2, .1, 0.]]),
                         retained_plan=torch.zeros(2, 5, 3), plan_mask=torch.ones(2, 5, dtype=torch.bool),
                         previous_action=torch.zeros(2, 3), budget=torch.ones(2))
        context = self.model.refresh.context(**arguments)
        changed_monitor = monitor.clone()
        changed_monitor[:, -1, 0] += 5
        changed_visual = self.model.refresh.context(
            **dict(arguments, tokens=changed_monitor[:, -1], monitor_tokens=changed_monitor))
        changed_plan = arguments['retained_plan'].clone()
        changed_plan[..., 0] = 3
        changed_trajectory = self.model.refresh.context(**dict(arguments, retained_plan=changed_plan))
        self.assertFalse(torch.allclose(context, changed_visual))
        self.assertFalse(torch.allclose(context, changed_trajectory))

    def test_revision_loss_is_coordinate_l1_with_periodic_heading(self):
        candidate = torch.tensor([2., 0., 2 * torch.pi]).repeat(1, 1, 5, 1).requires_grad_()
        flag = torch.zeros(1, 1, dtype=torch.bool)
        output = dict(
            candidate_trajectory=candidate, candidate_valid=~flag,
            decision_valid=~flag, refresh_logits=torch.zeros(1, 1, 2, requires_grad=True),
            refresh_labels=flag.long(), refresh_probability=torch.full((1, 1), .5),
            replacement_gain=torch.zeros(1, 1),
            planned_trajectory=torch.zeros(1, 1, 5, 3), plan_mask=torch.ones(1, 1, 5, dtype=torch.bool),
            event_refresh=flag, refresh_request=flag, evidence_acquired=flag,
            deadline_replan=flag, replan_mask=flag, monitoring_reads=~flag,
            online_decode_calls=torch.full((1, 1), 5),
            history_logits=torch.zeros(1, 1, 4, 2, requires_grad=True),
            history_labels=torch.zeros(1, 1, 4, dtype=torch.long),
            history_valid=torch.zeros(1, 1, 4, dtype=torch.bool))
        scores = revision_metrics(output, torch.zeros(1, 1, 5, 3),
                                  torch.ones(1, 1, 5, dtype=torch.bool))
        self.assertAlmostEqual(scores['trajectory_xy_l1'].item(), 1.)
        self.assertAlmostEqual(scores['trajectory_heading_l1'].item(), 0., places=5)
        self.assertAlmostEqual(scores['candidate_ade'].item(), 2.)
        scores['trajectory_l1'].backward()
        self.assertTrue(candidate.grad[..., 0].abs().sum() > 0)

    def test_old_action_strict_load_and_new_revision_checkpoint_roundtrip(self):
        old_action = FRESHDrive(
            mode='action', refresh_version=2, architecture_version=2,
            feature_dim=4, hidden_dim=8, heads=2, layers=1,
            spatial_grid=(3, 3), dropout=0)
        with tempfile.TemporaryDirectory(prefix='fresh-revision-', dir='/tmp') as temporary:
            old_path = Path(temporary) / 'action-v2.ckpt'
            new_path = Path(temporary) / 'revision-v3.ckpt'
            for model, path in ((old_action, old_path), (self.model, new_path)):
                checkpoint = dict(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams),
                                  **{'pytorch-lightning_version': pl.__version__})
                model.on_save_checkpoint(checkpoint)
                torch.save(checkpoint, path)
            restored_action = FRESHDrive.load_from_checkpoint(old_path, map_location='cpu')
            self.assertEqual(restored_action.refresh_version, 2)
            for name in ('tokenizer', 'memory', 'decoder'):
                getattr(self.model, name).load_state_dict(getattr(restored_action, name).state_dict(), strict=True)
            restored_revision = FRESHDrive.load_from_checkpoint(new_path, map_location='cpu')
            self.assertEqual(restored_revision.refresh_version, 3)
            with self.assertRaises((ValueError, RuntimeError)):
                FRESHDrive.load_from_checkpoint(new_path, refresh_version=2)
            with self.assertRaisesRegex(ValueError, 'revision protocol differs'):
                FRESHDrive.load_from_checkpoint(new_path, refresh_history_weight=.5)

    def test_training_cli_migrates_action_and_resumes_revision(self):
        action = FRESHDrive(mode='action', refresh_version=2, architecture_version=2,
                            feature_dim=4, hidden_dim=8, heads=2, layers=1,
                            spatial_grid=(3, 3), dropout=0)
        with tempfile.TemporaryDirectory(prefix='fresh-revision-cli-', dir='/tmp') as temporary:
            root = Path(temporary)
            action_path = root / 'action.ckpt'
            state = dict(state_dict=action.state_dict(), hyper_parameters=dict(action.hparams),
                         **{'pytorch-lightning_version': pl.__version__})
            action.on_save_checkpoint(state)
            torch.save(state, action_path)
            cache = root / 'preprocessed_dataset'
            bev = Path('preprocessed_dataset/bev_features/frame.pt')
            (root / bev).parent.mkdir(parents=True)
            torch.save(dict(bev_features=torch.randn(4, 6, 6),
                            bev_valid_mask=torch.ones(6, 6, dtype=torch.bool),
                            reference_ego_to_global=torch.eye(4), image_timestamps_us=torch.ones(8, dtype=torch.long),
                            ego_state={}, tracked_objects={}, bev_xy=torch.zeros(6, 6, 2),
                            bev_range=torch.tensor([-10, -10, -1, 10, 10, 1.])), root / bev)
            poses = torch.zeros(31, 3)
            poses[:, 0] = (torch.arange(31) - 20) * .1
            event = dict(anchor_index=20, anchor_timestamp_us=2000000,
                         timestamps_us=torch.arange(31) * 100000, bev_paths=[bev.as_posix()] * 31,
                         future_trajectory=poses[21:], future_actions=torch.zeros(10, 3),
                         future_action_mask=torch.ones(10, dtype=torch.bool), route_condition={},
                         trajectory=dict(past_trajectory=poses[:20], anchor_trajectory=poses[20],
                                         future_expert_trajectory=poses[21:], trajectory_mask=torch.ones(31, dtype=torch.bool)))
            for split in ('train', 'val'):
                name = Path(split) / 'event/sample.pt'
                path = cache / 'events' / name
                path.parent.mkdir(parents=True)
                torch.save(event, path)
                torch.save([name.as_posix()], cache / f'mini-{split}.pt')
            config = yaml.safe_load(Path('config/train/refresh.yaml').read_text())
            config['datamodule'].update(root=str(root), train_batch_size=1, val_batch_size=1,
                                       num_workers=0, pin_memory=False, persistent_workers=False)
            config['model'].update(feature_dim=4, hidden_dim=8, heads=2, layers=1,
                                   spatial_grid=[3, 3], dropout=0)
            config['trainer'].update(ckpt_path=str(action_path), accelerator='cpu', max_epochs=1,
                                     num_sanity_val_steps=0, enable_progress_bar=False, enable_model_summary=False)
            config['trainer']['csv_logger']['save_dir'] = str(root / 'logs')
            path = root / 'config.yaml'
            def run():
                path.write_text(yaml.safe_dump(config))
                with patch.object(sys, 'argv', ['train.py', '--config', str(path)]):
                    return runpy.run_path('train.py', run_name='__main__')['trainer']
            trained = run()
            self.assertEqual(trained.global_step, 1)
            checkpoint = Path(trained.checkpoint_callback.last_model_path)
            config['trainer'].update(ckpt_path=str(checkpoint), max_epochs=2)
            resumed = run()
            self.assertEqual(resumed.global_step, 2)
            self.assertEqual(Path(resumed.checkpoint_callback.dirpath), checkpoint.parent)
            for name, value in action.state_dict().items():
                if name.startswith(('tokenizer.', 'memory.', 'decoder.')):
                    self.assertTrue(torch.equal(value, resumed.lightning_module.state_dict()[name]), name)

    def test_revision_training_and_resume_keep_action_state_unchanged(self):
        before = {name: value.clone() for name, value in self.model.state_dict().items()
                  if name.startswith(('tokenizer.', 'memory.', 'decoder.'))}
        batch = dict(self.inputs, **self.targets)
        loader = DataLoader([batch], batch_size=None)
        with tempfile.TemporaryDirectory(prefix='fresh-revision-fit-', dir='/tmp') as temporary:
            callback = pl.callbacks.ModelCheckpoint(
                dirpath=temporary, monitor='val/ade', mode='min', save_top_k=1, save_last=True)
            settings = dict(accelerator='cpu', devices=1, logger=False,
                            enable_progress_bar=False, enable_model_summary=False,
                            num_sanity_val_steps=0, callbacks=[callback])
            trainer = pl.Trainer(max_epochs=1, **settings)
            trainer.fit(self.model, train_dataloaders=loader, val_dataloaders=loader)
            self.assertTrue(Path(callback.best_model_path).is_file())
            restored = FRESHDrive.load_from_checkpoint(callback.last_model_path, map_location='cpu')
            resumed = pl.Trainer(max_epochs=2, **settings)
            resumed.fit(restored, train_dataloaders=loader, val_dataloaders=loader,
                        ckpt_path=callback.last_model_path)
            self.assertEqual(resumed.global_step, 2)
            for name, value in before.items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
