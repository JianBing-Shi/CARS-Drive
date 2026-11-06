"""Finite-value summaries and paired comparisons; no missing-as-zero scores."""
import math
import numpy as np


def aggregate(rows, metric_keys):
    result = {'samples': len(rows), 'metrics': {}, 'valid_samples_per_metric': {},
              'quantiles': {}, 'recordings': len({r['recording'] for r in rows}),
              'valid_recordings_per_metric': {}}
    if rows and 'log_id' in rows[0]:
        result['original_logs'] = len({r['log_id'] for r in rows})
        result['valid_original_logs_per_metric'] = {}
    for key in metric_keys:
        valid = [r for r in rows if r.get(key) is not None and math.isfinite(r[key])]
        values = [r[key] for r in valid]
        result['metrics'][key] = math.fsum(values)/len(values) if values else None
        result['valid_samples_per_metric'][key] = len(values)
        result['valid_recordings_per_metric'][key] = len({r['recording'] for r in valid})
        if 'valid_original_logs_per_metric' in result:
            result['valid_original_logs_per_metric'][key] = len({r['log_id'] for r in valid})
        result['quantiles'][key] = {f'p{q}': float(np.percentile(values, q)) if values else None
                                    for q in (50, 90, 95)}
    return result


def paired_comparison(rows, names, *, cluster_key='recording', bootstrap_samples=2000, seed=42):
    """Baseline minus model; wins and ties are fractions of all paired samples.

    Bootstrap resamples complete clusters and recomputes the window-weighted
    paired mean. Only authoritative log IDs support a log-level CI. Caller must
    label DB/recording fallback honestly instead of implying independent logs.
    """
    rng = np.random.default_rng(seed)
    result = {}
    for name in names:
        pairs = [r for r in rows if r.get('model_'+name) is not None
                 and r.get('constant_action_'+name) is not None
                 and math.isfinite(r['model_'+name]) and math.isfinite(r['constant_action_'+name])]
        delta = np.array([r['constant_action_'+name]-r['model_'+name] for r in pairs])
        clusters = sorted({r[cluster_key] for r in pairs})
        sums = np.array([sum(d for r, d in zip(pairs, delta) if r[cluster_key] == c) for c in clusters])
        counts = np.array([sum(r[cluster_key] == c for r in pairs) for c in clusters])
        ci = None
        if len(clusters) >= 2 and bootstrap_samples > 0:
            estimates = []
            for _ in range(bootstrap_samples):
                indices = rng.integers(len(clusters), size=len(clusters))
                estimates.append(sums[indices].sum()/counts[indices].sum())
            ci = [float(x) for x in np.percentile(estimates, [2.5, 97.5])]
        result[name] = {'paired_samples': len(pairs), 'clusters': len(clusters),
                        'mean_improvement': float(delta.mean()) if len(delta) else None,
                        'model_win_fraction': float((delta > 1e-9).mean()) if len(delta) else None,
                        'tie_fraction': float((abs(delta) <= 1e-9).mean()) if len(delta) else None,
                        'mean_improvement_ci95': ci}
    return result
