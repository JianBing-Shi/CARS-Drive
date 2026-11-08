"""Causal ten-by-ten planning, frozen pred weights, and both decision objectives."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightning.pytorch as pl
import torch

from model import CARSDirve


class TrajectoryRevisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = dict(architecture_version=2, feature_dim=4, hidden_dim=8,
                           heads=2, layers=1, spatial_grid=(3, 3), dropout=0)
        self.inputs = dict(bev_features=torch.randn(2, 41, 4, 6, 6),
                           timestamps_us=torch.arange(41)[None].expand(2, -1) * 100000,
                           action_history_sequence=torch.randn(2, 11, 20, 3))
        self.targets = dict(planning_targets=torch.zeros(2, 10, 10, 3),
                            planning_mask=torch.ones(2, 10, 10, dtype=torch.bool))

    def model(self, mode):
        return CARSDirve(mode=mode, plan_horizon=10, **self.config)

    @staticmethod
    def keep_scores(mode):
        pair = [10., -10.] if mode == 'plan1' else [-10., 10.]
        return lambda context, *args, **kwargs: context.new_tensor(pair).expand(context.shape[0], -1)

    def test_expert_gradient_reaches_revision_but_base_is_frozen(self):
        for mode in ('plan1', 'plan2'):
            with self.subTest(mode=mode):
                model = self.model(mode).train()
                base_names = ('tokenizer.', 'memory.', 'action_encoder.', 'decoder.')
                before = {n: v.clone() for n, v in model.state_dict().items() if n.startswith(base_names)}
                for module in (model.tokenizer, model.memory, model.action_encoder, model.decoder):
                    self.assertFalse(module.training)
                output = model(**self.inputs, **self.targets, return_details=True)
                metrics = model.revision_metrics(output, self.targets['planning_targets'], self.targets['planning_mask'])
                metrics['trajectory_l1'].backward()
                self.assertGreater(model.cars_adapter.revision[-1].weight.grad.abs().sum().item(), 0)
                optimizers, _ = model.configure_optimizers()
                optimizers[0].step()
                for n, value in before.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[n]), n)
                optimized = {id(p) for group in optimizers[0].param_groups for p in group['params']}
                self.assertTrue(all(n.startswith('cars_adapter.') for n, p in model.named_parameters() if id(p) in optimized))
                self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n.startswith(base_names)))

    def test_rejected_candidates_cost_ten_calls_and_deadline_is_independent(self):
        for mode in ('plan1', 'plan2'):
            with self.subTest(mode=mode):
                model = self.model(mode).eval()
                with patch.object(model.cars_adapter, 'compare', side_effect=self.keep_scores(mode)):
                    with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decoder:
                        output = model(**self.inputs, **self.targets, return_details=True)
                self.assertEqual(decoder.call_count, 110)
                self.assertEqual(tuple(output['candidate_trajectory'].shape), (2, 10, 10, 3))
                self.assertTrue(output['candidate_valid'].all())
                self.assertFalse(output['event_refresh'].any())
                self.assertEqual(output['deadline_replan'][0].nonzero().flatten().tolist(), [9])
                self.assertFalse(output['decision_valid'][:, 9].any())
                self.assertEqual(output['online_decode_calls'][0].sum().item(), 100)
                self.assertEqual(output['history_valid'][0].sum(-1).tolist(), list(range(10)))
                self.assertEqual(output['history_common_mask'][0, 9].sum(-1).tolist(), list(range(9, 0, -1)))

    def test_future_labels_observations_and_action_rows_never_change_earlier_decisions(self):
        for mode in ('plan1', 'plan2'):
            with self.subTest(mode=mode):
                model = self.model(mode).eval()
                output = model(**self.inputs, **self.targets, return_details=True)
                other = dict(self.targets, planning_targets=torch.randn(2, 10, 10, 3) * 10,
                             planning_mask=torch.zeros(2, 10, 10, dtype=torch.bool))
                changed_gt = model(**self.inputs, **other, return_details=True)
                unlabeled = model(**self.inputs, return_details=True)
                keys = ('candidate_trajectory', 'planned_trajectory',
                        'refresh_logits' if mode == 'plan1' else 'quality_errors')
                for key in keys:
                    torch.testing.assert_close(output[key], changed_gt[key])
                    torch.testing.assert_close(output[key], unlabeled[key])
                changed = dict(self.inputs, bev_features=self.inputs['bev_features'].clone(),
                               action_history_sequence=self.inputs['action_history_sequence'].clone())
                changed['bev_features'][:, 26:] += 100
                changed['action_history_sequence'][:, 6:] += 100
                future = model(**changed, **self.targets, return_details=True)
                for key in keys:
                    torch.testing.assert_close(output[key][:, :5], future[key][:, :5])
                changed = dict(self.inputs, action_history_sequence=self.inputs['action_history_sequence'].clone())
                changed['action_history_sequence'][:, 4] += 10
                action_changed = model(**changed, return_details=True)
                torch.testing.assert_close(output['candidate_trajectory'][:, :3], action_changed['candidate_trajectory'][:, :3])
                self.assertFalse(torch.allclose(output['candidate_trajectory'][:, 3], action_changed['candidate_trajectory'][:, 3]))

    def test_missing_candidates_and_nan_padding_have_zero_finite_loss(self):
        for mode in ('plan1', 'plan2'):
            with self.subTest(mode=mode):
                model = self.model(mode).eval()
                model.cars_adapter_budget = 0
                targets = dict(planning_mask=torch.zeros(2, 10, 10, dtype=torch.bool),
                               planning_targets=torch.full((2, 10, 10, 3), float('nan')))
                with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decoder:
                    output = model(**self.inputs, **targets, return_details=True)
                self.assertEqual(decoder.call_count, 20)
                metrics = model.revision_metrics(output, targets['planning_targets'], targets['planning_mask'])
                self.assertEqual(metrics['loss'].item(), 0)
                self.assertTrue(metrics['loss'].requires_grad)
                metrics['loss'].backward()
                self.assertFalse(output['candidate_generated'].any())
                self.assertFalse(output['history_valid'].any())
                self.assertTrue(torch.isfinite(output['planned_trajectory']).all())

    def test_partial_expert_support_does_not_relabel_a_longer_scored_plan(self):
        for mode in ('plan1', 'plan2'):
            with self.subTest(mode=mode):
                model = self.model(mode).eval()
                mask = self.targets['planning_mask'].clone()
                mask[..., 3:] = False
                with patch.object(model.cars_adapter, 'compare', side_effect=self.keep_scores(mode)):
                    output = model(**self.inputs, planning_targets=self.targets['planning_targets'],
                                   planning_mask=mask, return_details=True)
                self.assertEqual(output['decision_valid'][0].tolist(), [False] * 6 + [True] * 3 + [False])
                # The label mask never truncates the causal retained plan itself.
                self.assertEqual(output['retained_mask'][0].sum(-1).tolist(), list(range(9, -1, -1)))

    def test_shared_quality_is_symmetric_and_uses_coordinate_l1(self):
        model = self.model('plan2')
        zero = torch.zeros(2, 10, 3)
        poor = zero.clone()
        poor[..., 0] = .4
        poor[..., 2] = 2 * torch.pi
        mask = torch.ones(2, 10, dtype=torch.bool)
        labels, valid, gain = model.cars_adapter.replacement_labels(
            zero, poor, zero, mask, torch.ones(2, dtype=torch.bool), 0., 0., torch.ones(2))
        self.assertTrue(valid.all())
        self.assertFalse(labels.any())
        torch.testing.assert_close(gain, torch.full_like(gain, -.2))
        context = torch.randn(2, 8)
        scores = model.cars_adapter.compare(context, zero, poor, mask)
        swapped = model.cars_adapter.compare(context, poor, zero, mask)
        torch.testing.assert_close(scores, swapped.flip(-1))
        tied = model.cars_adapter.compare(context, zero, zero, mask)
        torch.testing.assert_close(tied[:, 0], tied[:, 1])

    def test_quality_training_has_no_cross_entropy_or_probability_outputs(self):
        model = self.model('plan2')
        with patch('torch.nn.functional.cross_entropy', side_effect=AssertionError('CE forbidden in plan2')):
            output = model(**self.inputs, **self.targets, return_details=True)
            metrics = model.revision_metrics(output, self.targets['planning_targets'], self.targets['planning_mask'])
            metrics['loss'].backward()
        self.assertNotIn('refresh_logits', output)
        self.assertNotIn('refresh_probability', output)
        self.assertNotIn('replacement_ce', metrics)
        self.assertTrue((output['quality_errors'] >= 0).all())
        self.assertGreater(model.cars_adapter.score[-1].weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(output['predicted_gain'], output['quality_errors'][..., 0] - output['quality_errors'][..., 1])
        torch.testing.assert_close(output['replacement_gain'], output['quality_targets'][..., 0] - output['quality_targets'][..., 1])

    def test_current_bev_and_old_plan_affect_context(self):
        adapter = self.model('plan1').cars_adapter
        state = dict(memory=torch.randn(2, 9, 8), valid=torch.ones(2, 9, dtype=torch.bool), age_s=torch.zeros(2, 9))
        monitor = torch.randn(2, 3, 9, 8)
        args = dict(state=state, tokens=monitor[:, -1], token_valid=state['valid'], monitor_tokens=monitor,
                    monitor_valid=state['valid'][:, None].expand(-1, 3, -1),
                    monitor_lags=torch.tensor([[.2, .1, 0.], [.2, .1, 0.]]),
                    retained_plan=torch.zeros(2, 10, 3), plan_mask=torch.ones(2, 10, dtype=torch.bool),
                    previous_action=torch.zeros(2, 3), budget=torch.ones(2))
        context = adapter.context(**args)
        changed = monitor.clone()
        changed[:, -1, 0] += 5
        visual = adapter.context(**dict(args, tokens=changed[:, -1], monitor_tokens=changed))
        plan = args['retained_plan'].clone()
        plan[..., 0] = 3
        trajectory = adapter.context(**dict(args, retained_plan=plan))
        self.assertFalse(torch.allclose(context, visual))
        self.assertFalse(torch.allclose(context, trajectory))

    def test_checkpoint_protocol_rejects_cross_objective_and_preserves_pred_compatibility(self):
        with tempfile.TemporaryDirectory(prefix='cars-modes-', dir='/tmp') as directory:
            for mode in ('pred', 'plan1', 'plan2'):
                with self.subTest(mode=mode):
                    model = CARSDirve(mode='pred', **self.config) if mode == 'pred' else self.model(mode)
                    checkpoint = dict(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams),
                                      **{'pytorch-lightning_version': pl.__version__})
                    model.on_save_checkpoint(checkpoint)
                    if mode == 'pred':
                        # Existing action checkpoints predate the new planning metadata.
                        for key in ('cache_anchor_step', 'quality_loss_weight', 'gain_loss_weight'):
                            checkpoint['hyper_parameters'].pop(key)
                        checkpoint['refresh_protocol'] = {'refresh_version': 3}
                    path = Path(directory) / f'{mode}.ckpt'
                    torch.save(checkpoint, path)
                    restored = CARSDirve.load_from_checkpoint(path, map_location='cpu')
                    self.assertEqual(restored.mode, mode)
                    for name, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
                    if mode != 'pred':
                        with self.assertRaisesRegex(ValueError, 'revision protocol differs'):
                            CARSDirve.load_from_checkpoint(path, mode='plan2' if mode == 'plan1' else 'plan1')
                        with self.assertRaisesRegex(ValueError, 'revision protocol differs'):
                            CARSDirve.load_from_checkpoint(path, refresh_history_weight=.5)

    def test_missing_causal_history_and_legacy_window_fail_explicitly(self):
        model = self.model('plan1')
        with self.assertRaisesRegex(ValueError, 'action_history_sequence'):
            model(self.inputs['bev_features'])
        with self.assertRaisesRegex(ValueError, '41-frame'):
            model(self.inputs['bev_features'][:, :31], action_history_sequence=self.inputs['action_history_sequence'])
        with self.assertRaisesRegex(ValueError, 'mode must be'):
            CARSDirve(mode='plan')


if __name__ == '__main__':
    unittest.main()
