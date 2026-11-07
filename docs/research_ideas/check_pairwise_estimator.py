"""Finite-action algebra check; this does not test driving performance."""
import json

pi = [0.2, 0.3, 0.5]
q_reference = [1.0, 4.0, -2.0]
q_model = [2.0, -1.0, 3.0]
n = len(pi)
mean_q = sum(p * q for p, q in zip(pi, q_reference))
analytic = [pi[k] * (q_reference[k] - mean_q) for k in range(n)]
corrected = [0.0] * n
uncorrected = [0.0] * n
for a in range(n):
    for b in range(n):
        p_audit = 0.15 + 0.1 * ((a + 2 * b) % 5)
        delta_model = q_model[a] - q_model[b]
        delta_reference = q_reference[a] - q_reference[b]
        for audited, probability in [(0, 1 - p_audit), (1, p_audit)]:
            estimate = delta_model + audited / p_audit * (delta_reference - delta_model)
            naive = delta_model + audited * (delta_reference - delta_model)
            for k in range(n):
                pair_score = 0.5 * (int(a == k) - int(b == k))
                weight = pi[a] * pi[b] * probability
                corrected[k] += weight * pair_score * estimate
                uncorrected[k] += weight * pair_score * naive
assert max(abs(x - y) for x, y in zip(analytic, corrected)) < 1e-12
assert max(abs(x - y) for x, y in zip(analytic, uncorrected)) > 0.1
print(json.dumps({
    "analytic_one_state_gradient": analytic,
    "enumerated_corrected_estimate": corrected,
    "without_inverse_selection_probability": uncorrected,
    "scope": "Exact finite-action algebra only; no simulator or training validation.",
}, ensure_ascii=False, indent=2))
