"""Fast mini-data diagnostic for the visual-planning hypothesis.

This script deliberately does *not* claim closed-loop or visual evidence. It uses
Plan-R1's processed nuPlan tensors and recorded future agents to build a cheap
frozen-traffic oracle replay proxy. Its purpose is to answer whether imitation
trajectory error and action ranking can disagree before implementing a new model.

Usage (after Plan-R1 preprocessing):
  python mini_action_ranking_diagnostic.py \
    --data-root D:/nuplan/dataset --dir mini --split val --max-scenarios 200

The resulting CSV/JSON are diagnostic artifacts. For a paper result, repeat the
same candidate pairs in nuPlan reactive simulation with independent branches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def _get(data: Any, key: str, default: Any = None) -> Any:
    """Read a field from a PyG HeteroData, dict, or object."""
    try:
        if isinstance(data, dict):
            return data.get(key, default)
        return data[key]
    except (KeyError, TypeError, AttributeError):
        return default


def _as_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _find_files(data_root: Path, directory: str, split: str) -> List[Path]:
    base = data_root / "nuplan-v1.1" / "splits"
    processed = base / f"{directory}-processed-plan-{split}-PlanR1"
    if not processed.exists():
        processed = base / f"{directory}-processed-pred-{split}-PlanR1"
    if not processed.exists():
        raise FileNotFoundError(
            f"No processed Plan-R1 directory found under {base}. "
            "Run preprocess_dataset.py first."
        )
    return sorted(processed.glob("*.pt"))


def _rotate_to_local(points: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]], dtype=np.float32)
    return points @ rot.T


def _candidate_set(gt: np.ndarray, heading: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    """Make small, reproducible local perturbations around logged ego future."""
    h = min(n, len(gt))
    base = gt[:h].copy()
    # Plan-R1 uses an ego-centered frame at the present state; y is approximately
    # lateral. We keep perturbations small so this remains a diagnostic, not a
    # new trajectory generator.
    candidates: Dict[str, np.ndarray] = {"expert": base}
    for off in (-1.0, -0.5, 0.5, 1.0):
        q = base.copy()
        q[:, 1] += off
        candidates[f"lateral_{off:+.1f}m"] = q
    for scale in (0.80, 1.20):
        q = base.copy()
        q[:, 0] *= scale
        candidates[f"longitudinal_{scale:.2f}x"] = q
    # A deterministic mild brake/advance pair. It changes the relative position
    # while avoiding a learned model or hidden future labels.
    for delta in (-1.0, 1.0):
        q = base.copy()
        alpha = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        q[:, 0] += (delta * alpha[:, 0])
        candidates[f"advance_{delta:+.1f}m"] = q
    return candidates


def _collision_proxy(
    ego_future: np.ndarray,
    ego_heading: np.ndarray,
    ego_box: np.ndarray,
    others_future: np.ndarray,
    others_heading: np.ndarray,
    others_box: np.ndarray,
    others_valid: np.ndarray,
) -> Tuple[bool, int]:
    """Conservative oriented-box proxy against recorded (frozen) other futures."""
    if others_future.size == 0:
        return False, 0
    h = min(len(ego_future), others_future.shape[1])
    ego_front, ego_rear, ego_left, ego_right = [float(x) for x in ego_box[:4]]
    ego_half_l = max(0.5, 0.5 * (ego_front + ego_rear))
    ego_half_w = max(0.5, 0.5 * (ego_left + ego_right))
    collision_steps = 0
    for t in range(h):
        e = ego_future[t]
        ey = float(ego_heading[min(t, len(ego_heading) - 1)])
        for j in range(others_future.shape[0]):
            if not bool(others_valid[j, min(t, others_valid.shape[1] - 1)]):
                continue
            rel = _rotate_to_local(others_future[j, t] - e, ey)
            of, or_, ol, oR = [float(x) for x in others_box[j, :4]]
            long_margin = ego_half_l + max(0.5, 0.5 * (of + or_))
            lat_margin = ego_half_w + max(0.4, 0.5 * (ol + oR))
            if abs(float(rel[0])) <= long_margin and abs(float(rel[1])) <= lat_margin:
                collision_steps += 1
                break
    return collision_steps > 0, collision_steps


def _scenario_metrics(data: Any, path: Path, horizon: int) -> Optional[Dict[str, Any]]:
    agent = _get(data, "agent")
    if agent is None:
        return None
    pos = _as_numpy(_get(agent, "position"))
    heading = _as_numpy(_get(agent, "heading"))
    visible = _as_numpy(_get(agent, "visible_mask"))
    box = _as_numpy(_get(agent, "box"))
    if pos is None or heading is None or box is None or pos.ndim != 3 or pos.shape[0] < 1:
        return None
    t_current = min(20, pos.shape[1] - 2)
    h = min(horizon, pos.shape[1] - t_current - 1)
    if h < 5:
        return None
    gt = pos[0, t_current + 1 : t_current + 1 + h].astype(np.float32)
    gt_heading = heading[0, t_current + 1 : t_current + 1 + h].astype(np.float32)
    other_pos = pos[1:, t_current + 1 : t_current + 1 + h].astype(np.float32)
    other_heading = heading[1:, t_current + 1 : t_current + 1 + h].astype(np.float32)
    other_valid = (
        visible[1:, t_current + 1 : t_current + 1 + h].astype(bool)
        if visible is not None
        else np.ones(other_pos.shape[:2], dtype=bool)
    )
    candidates = _candidate_set(gt, gt_heading, h)
    rows: List[Dict[str, Any]] = []
    for name, traj in candidates.items():
        ade = float(np.linalg.norm(traj - gt, axis=-1).mean())
        fde = float(np.linalg.norm(traj[-1] - gt[-1]))
        coll, coll_steps = _collision_proxy(
            traj, gt_heading, box[0], other_pos, other_heading, box[1:], other_valid
        )
        progress = float(traj[-1, 0] - traj[0, 0])
        # Frozen-traffic oracle replay proxy. This is intentionally transparent:
        # it is not a substitute for a reactive nuPlan simulation.
        oracle = progress - 100.0 * float(coll) - 2.0 * ade
        imitation_score = -ade
        rows.append(
            dict(
                scenario=str(_get(data, "scenario_name", path.stem)),
                candidate=name,
                ade=ade,
                fde=fde,
                collision=int(coll),
                collision_steps=coll_steps,
                progress=progress,
                oracle_proxy=oracle,
                imitation_score=imitation_score,
            )
        )
    pred_best = max(rows, key=lambda r: r["imitation_score"])
    oracle_best = max(rows, key=lambda r: r["oracle_proxy"])
    inversions = 0
    pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dp = rows[i]["imitation_score"] - rows[j]["imitation_score"]
            dq = rows[i]["oracle_proxy"] - rows[j]["oracle_proxy"]
            if abs(dp) < 1e-8 or abs(dq) < 1e-8:
                continue
            pairs += 1
            inversions += int(dp * dq < 0)
    return dict(
        path=str(path),
        scenario=str(_get(data, "scenario_name", path.stem)),
        num_candidates=len(rows),
        imitation_best=pred_best["candidate"],
        oracle_best=oracle_best["candidate"],
        best_mismatch=int(pred_best["candidate"] != oracle_best["candidate"]),
        pairwise_inversion_rate=(inversions / pairs if pairs else 0.0),
        rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dir", default="mini")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max-scenarios", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("mini_action_ranking_results"))
    args = parser.parse_args()
    files = _find_files(args.data_root, args.dir, args.split)[: args.max_scenarios]
    args.out.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for idx, path in enumerate(files, 1):
        try:
            # Plan-R1 stores ordinary dictionaries/HeteroData objects. Explicitly
            # opt out of the newer safe-only default for trusted local files.
            data = torch.load(path, map_location="cpu", weights_only=False)
            result = _scenario_metrics(data, path, args.horizon)
            if result is not None:
                summaries.append({k: v for k, v in result.items() if k != "rows"})
                rows.extend(result["rows"])
        except Exception as exc:  # retain per-file failure for auditability
            summaries.append({"path": str(path), "error": repr(exc)})
        if idx % 25 == 0 or idx == len(files):
            print(f"processed {idx}/{len(files)}")
    valid = [x for x in summaries if "error" not in x]
    report = {
        "protocol": "Plan-R1 processed mini data; frozen recorded-agent replay proxy",
        "not_closed_loop": True,
        "not_visual": True,
        "num_files": len(files),
        "num_valid": len(valid),
        "mismatch_rate": float(np.mean([x["best_mismatch"] for x in valid])) if valid else None,
        "mean_pairwise_inversion_rate": float(np.mean([x["pairwise_inversion_rate"] for x in valid])) if valid else None,
        "files": summaries,
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (args.out / "candidate_rows.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ["num_files", "num_valid", "mismatch_rate", "mean_pairwise_inversion_rate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
