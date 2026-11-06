"""Discrete, non-reactive collision proxies against recorded oriented boxes."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class EgoFootprint:
    """Distances from the cached ego pose origin, in metres (+x front, +y left)."""
    front: float
    rear: float
    left: float
    right: float

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.front, self.rear, self.left, self.right)):
            raise ValueError("Footprint distances must be finite and positive")


def rectangle_overlaps(center, yaw, length, width, boxes):
    """Separating-axis test for ego and N boxes [x,y,z,length,width,height,yaw].

    Boundary contact counts as overlap. No spatial rasterization or ego-centre
    point approximation. Inputs must already share a coordinate frame.
    """
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.bool)
    ego_u = torch.stack((yaw.cos(), yaw.sin()))
    ego_v = torch.stack((-yaw.sin(), yaw.cos()))
    other_u = torch.stack((boxes[:, 6].cos(), boxes[:, 6].sin()), -1)
    other_v = torch.stack((-boxes[:, 6].sin(), boxes[:, 6].cos()), -1)
    axes = torch.stack((ego_u.expand_as(other_u), ego_v.expand_as(other_v), other_u, other_v), 1)
    delta = boxes[:, :2] - center
    distance = (delta[:, None] * axes).sum(-1).abs()
    radius_ego = ((axes * ego_u).sum(-1).abs() * length / 2
                  + (axes * ego_v).sum(-1).abs() * width / 2)
    radius_other = ((axes * other_u[:, None]).sum(-1).abs() * boxes[:, 3:4] / 2
                    + (axes * other_v[:, None]).sum(-1).abs() * boxes[:, 4:5] / 2)
    return (distance <= radius_ego + radius_other + 1e-7).all(-1)


def collision_metrics(trajectory, prefix, annotations, anchor_poses, target_times,
                      footprint=None, max_annotation_age_us=100_000):
    """Per-window any-overlap requires a complete trajectory AND annotations.

    Annotation frames are in their own ego coordinates; candidate centres are
    mapped from the anchor into each annotation frame. Invalid frames remain
    NaN, including when their box list is empty. Valid empty lists mean clear.
    Missing footprint disables collision evaluation explicitly.
    """
    trajectory, prefix = trajectory.double().cpu(), prefix.bool().cpu()
    batch, steps = prefix.shape
    if (trajectory.shape != (batch, steps, 3) or target_times.shape != (batch, steps)
            or (annotations is not None and (len(annotations) != batch
                or any(len(frames) != steps for frames in annotations)))):
        raise ValueError("Collision inputs require [B,T,3], [B,T] and B annotation lists of length T")
    if not torch.isfinite(trajectory[prefix]).all():
        raise ValueError("Non-finite predicted trajectory at valid collision endpoints")
    collisions = torch.full((batch, steps), float('nan'), dtype=torch.float64)
    coverage = torch.zeros(batch, steps, dtype=torch.bool)
    for i in range(batch):
        for t in range(steps):
            frame = annotations[i][t] if annotations is not None else None
            if frame is None or not frame.get('valid', False):
                continue
            age = int(target_times[i, t]) - int(frame.get('annotation_timestamp_us', -1))
            # Offline labels may lie just after a nominal 10 Hz grid timestamp.
            # Symmetric matching is evaluation-only, never an input permission.
            if abs(age) > max_annotation_age_us:
                continue
            boxes = torch.as_tensor(frame.get('boxes', []), dtype=torch.float64)
            pose = frame.get('reference_ego_to_global')
            if boxes.ndim != 2 or boxes.shape[-1] != 7 or pose is None:
                continue
            if not torch.isfinite(boxes).all() or (boxes[:, 3:6] <= 0).any():
                continue
            pose = torch.as_tensor(pose, dtype=torch.float64)
            if pose.shape != (4, 4) or not torch.isfinite(pose).all():
                continue
            coverage[i, t] = True
            if footprint is None or anchor_poses is None or not prefix[i, t]:
                continue
            source = torch.as_tensor(anchor_poses[i], dtype=torch.float64)
            if source.shape != (4, 4) or not torch.isfinite(source).all():
                continue
            try:
                transform = torch.linalg.solve(pose, source)
            except RuntimeError:
                continue
            x, y, yaw = trajectory[i, t]
            offset_x = (footprint.front - footprint.rear) / 2
            offset_y = (footprint.left - footprint.right) / 2
            center = torch.stack((x + yaw.cos()*offset_x - yaw.sin()*offset_y,
                                  y + yaw.sin()*offset_x + yaw.cos()*offset_y,
                                  x.new_zeros(()), x.new_ones(())))
            center = (transform @ center)[:2]
            direction = transform[:2, :2] @ torch.stack((yaw.cos(), yaw.sin()))
            angle = torch.atan2(direction[1], direction[0])
            collisions[i, t] = rectangle_overlaps(
                center, angle, footprint.front + footprint.rear,
                footprint.left + footprint.right, boxes).any().double()
    complete = prefix.all(-1) & torch.isfinite(collisions).all(-1)
    result = {
        'collision_proxy_any': collisions.nan_to_num().bool().any(-1).double().masked_fill(~complete, float('nan')),
        'collision_proxy_step_fraction': collisions.nan_to_num().mean(-1).masked_fill(~complete, float('nan')),
        'collision_annotation_coverage': coverage.double().mean(-1),
        'collision_evaluated_coverage': torch.isfinite(collisions).double().mean(-1),
    }
    for step in range(steps):
        result[f'collision_proxy_step_{step + 1}'] = collisions[:, step]
    return result, collisions
