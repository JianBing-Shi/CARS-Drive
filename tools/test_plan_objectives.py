"""Distinct plan objectives, causal selection diagnostics, and masked gradients."""
import unittest

import torch
from torch.nn import functional as F

from metrics.trajectory_revision import revision_metrics


def fixture(mode):
    valid = torch.ones(1, 2, 10, dtype=torch.bool)
    eligible = torch.tensor([[True, False]])
    candidate = torch.zeros(1, 2, 10, 3)
    candidate[0, 0, :, 0] = 2
    candidate[0, 1] = float('nan')
    candidate.requires_grad_()
    output = dict(planning_mode=mode, decision_valid=eligible,
        refresh_labels=torch.zeros(1, 2, dtype=torch.long),
        plan_mask=valid, planned_trajectory=torch.zeros_like(candidate),
        candidate_trajectory=candidate, candidate_valid=eligible,
        replacement_gain=torch.tensor([[0., float('nan')]]),
        replace_decision=eligible, event_refresh=eligible, refresh_request=eligible,
        evidence_acquired=eligible, deadline_replan=~eligible, replan_mask=eligible,
        monitoring_reads=torch.ones_like(eligible), online_decode_calls=eligible.long() * 10,
        history_valid=torch.tensor([[[True, True], [False, False]]]),
        history_labels=torch.zeros(1, 2, 2, dtype=torch.long),
        history_gain=torch.tensor([[[0., 0.], [float('nan'), float('nan')]]]))
    return output, torch.zeros_like(candidate), valid


class PlanObjectiveTests(unittest.TestCase):
    def test_plan2_uses_quality_gain_l1_and_equal_decision_history_weight(self):
        output, target, valid = fixture('plan2')
        output.update(quality_errors=torch.tensor([[[3., 1.], [float('nan'), float('nan')]]], requires_grad=True),
            quality_targets=torch.tensor([[[2., 2.], [float('nan'), float('nan')]]], requires_grad=True),
            predicted_gain=torch.tensor([[2., float('nan')]], requires_grad=True),
            history_quality_errors=torch.tensor([[[[4., 1.], [2., 1.]],
                [[float('nan'), float('nan')], [float('nan'), float('nan')]]]], requires_grad=True),
            history_quality_targets=torch.tensor([[[[2., 2.], [2., 2.]],
                [[float('nan'), float('nan')], [float('nan'), float('nan')]]]], requires_grad=True))
        metrics = revision_metrics(output, target, valid)
        self.assertEqual(float(metrics['trajectory_l1']), 1.)
        self.assertEqual(float(metrics['quality_l1']), 1.)
        self.assertEqual(float(metrics['gain_l1']), 2.)
        self.assertEqual(float(metrics['history_quality_l1']), 1.)
        self.assertEqual(float(metrics['history_gain_l1']), 2.)
        self.assertEqual(float(metrics['loss']), 4.75)
        self.assertNotIn('replacement_ce', metrics)
        self.assertNotIn('history_ce', metrics)
        self.assertNotIn('pr_auc_ap', metrics)
        self.assertIn('candidate_fixed10_ade', metrics)
        self.assertIn('candidate_fixed5_ade', metrics)
        metrics['loss'].backward()
        for key in ('candidate_trajectory', 'quality_errors', 'predicted_gain', 'history_quality_errors'):
            self.assertTrue(torch.isfinite(output[key].grad).all(), key)
        self.assertIsNone(output['quality_targets'].grad)
        self.assertIsNone(output['history_quality_targets'].grad)

    def test_plan1_has_only_preference_cross_entropy(self):
        output, target, valid = fixture('plan1')
        output.update(refresh_logits=torch.tensor([[[0., 1.], [float('nan'), float('nan')]]], requires_grad=True),
            refresh_probability=torch.tensor([[.7, .7]]),
            history_logits=torch.tensor([[[[0., 1.], [0., 1.]],
                [[float('nan'), float('nan')], [float('nan'), float('nan')]]]], requires_grad=True))
        metrics = revision_metrics(output, target, valid)
        ce = F.cross_entropy(torch.tensor([[0., 1.]]), torch.tensor([0]))
        torch.testing.assert_close(metrics['loss'], 1 + 1.25 * ce)
        self.assertNotIn('quality_l1', metrics)
        self.assertIn('replacement_ce', metrics)
        metrics['loss'].backward()
        self.assertTrue(torch.isfinite(output['history_logits'].grad).all())

    def test_empty_supervision_is_finite_zero_with_backward(self):
        for mode in ('plan1', 'plan2'):
            output, target, valid = fixture(mode)
            output['candidate_valid'].zero_()
            output['decision_valid'].zero_()
            output['history_valid'].zero_()
            valid.zero_()
            if mode == 'plan1':
                output.update(refresh_logits=torch.full((1, 2, 2), float('nan'), requires_grad=True),
                    refresh_probability=torch.zeros(1, 2),
                    history_logits=torch.full((1, 2, 2, 2), float('nan'), requires_grad=True))
            else:
                output.update(quality_errors=torch.full((1, 2, 2), float('nan'), requires_grad=True),
                    quality_targets=torch.full((1, 2, 2), float('nan')),
                    predicted_gain=torch.full((1, 2), float('nan'), requires_grad=True),
                    history_quality_errors=torch.full((1, 2, 2, 2), float('nan'), requires_grad=True),
                    history_quality_targets=torch.full((1, 2, 2, 2), float('nan')))
            metrics = revision_metrics(output, target, valid)
            self.assertEqual(float(metrics['loss']), 0.)
            metrics['loss'].backward()
            self.assertTrue(torch.isfinite(output['candidate_trajectory'].grad).all())


if __name__ == '__main__':
    unittest.main()
