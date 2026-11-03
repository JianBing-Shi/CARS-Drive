"""nuPlan camera frame loading and frozen features; no tokenization or persistence.

Prepare a frame with ``load_nuplan_camera_frame`` and load the
image/BEV encoders once with the corresponding utilities in ``utils.camera``.
The preprocessing entry point supplies ``bevformer_root`` when loading the BEV
encoder; its Tiny configuration is resolved internally::

    images = extract_camera_features(frame["image_paths"], image_encoder,
                                     calibration=frame)
    bev = camera_bev_features(images, bev_encoder, frame)

``camera_features`` is CPU FP16 [8,256,H/32,W/32]. ``bev_features`` is
CPU FP16 [C,H_bev,W_bev] (Tiny: [256,50,50]), in the reference ego frame.
BEV visual tokens are created/saved later in model training.
"""

from __future__ import annotations

import sqlite3
from bisect import bisect_right
from collections.abc import Mapping
from contextlib import closing, ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from utils.camera import _camera_tensor
from utils.camera import _decode_calibration
from utils.camera import _ordered_cameras
from utils.camera import _pose_matrix
from utils.camera import _rectified_resize
from utils.camera import load_backbone_encoder
from utils.camera import load_bevformer_encoder
from utils.camera import build_ego_to_image_matrices
from utils.planning import wrap_angle, ego_pose_yaw, transform_points_to_local_coordinate_3d

EVENT_INTERVAL_US = 1_000_000
FRAME_INTERVAL_US = 100_000

WIDTH, HEIGHT = 640, 384
CAMERAS = ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2")

EVENT_TYPES = {
    # --- 纵向跟车与启停控制 ---
    "stopping_with_lead",                                         # 跟随前车并减速停车
    "following_lane_without_lead",                                # 在车道内无前车行驶
    "following_lane_with_lead",                                   # 在车道内跟随前车行驶
    "following_lane_with_slow_lead",                              # 在车道内跟随低速前车行驶
    "behind_long_vehicle",                                        # 行驶在大型车辆（如卡车、公交车）后方
    # "stationary_in_traffic",                                      # 在车流中保持静止
    # "near_multiple_vehicles",                                     # 靠近多辆车辆
    # "near_long_vehicle",                                          # 靠近大型车辆

    # --- 信号灯、停止线与停车标志交互 ---
    # "on_stopline_traffic_light",                                  # 自车位于红绿灯停止线处（通常指停车等待状态）
    # "accelerating_at_stopline",                                   # 在停止线处起步加速
    # "accelerating_at_traffic_light",                              # 在红绿灯处起步加速
    # "accelerating_at_traffic_light_with_lead",                    # 在红绿灯处跟随前车起步加速
    # "accelerating_at_traffic_light_without_lead",                 # 在红绿灯处无前车时起步加速
    # "stopping_at_traffic_light_with_lead",                        # 在红绿灯前跟随前车停车
    # "stopping_at_traffic_light_without_lead",                     # 在红绿灯前无前车时停车
    # "stationary_at_traffic_light_with_lead",                      # 在红绿灯前跟随前车静止
    # "stationary_at_traffic_light_without_lead",                   # 在红绿灯前无前车时静止
    # "accelerating_at_stop_sign",                                  # 在停车标志处起步加速
    # "accelerating_at_stop_sign_no_crosswalk",                     # 在无斑马线的停车标志处起步加速
    # "stopping_at_stop_sign_with_lead",                            # 在停车标志前跟随前车停车
    # "stopping_at_stop_sign_without_lead",                         # 在停车标志前无前车时停车
    # "stopping_at_stop_sign_no_crosswalk",                         # 在无斑马线的停车标志处停车
    # "on_stopline_stop_sign",                                      # 自车位于停车标志停止线处

    # --- 行人与斑马线交互 ---
    "waiting_for_pedestrian_to_cross",                            # 停车等待行人横穿马路
    "stopping_at_crosswalk",                                      # 在斑马线处停车
    "near_pedestrian_on_crosswalk",                               # 靠近斑马线上的行人
    "near_pedestrian_on_crosswalk_with_ego",                      # 自车与行人同时处于斑马线附近（存在潜在冲突）
    # "near_multiple_pedestrians",                                  # 靠近多名行人
    # "behind_pedestrian",                                          # 行驶在行人后方
    # "behind_pedestrian_on_driveable",                             # 行驶在可行驶区域行人后方
    # "behind_pedestrian_on_pickup_dropoff",                        # 行驶在上下客区域行人后方
    # "near_pedestrian_at_pickup_dropoff",                          # 靠近上下客区域的行人
    # "traversing_crosswalk",                                       # 正在穿越斑马线
    # "accelerating_at_crosswalk",                                  # 在斑马线处加速通过
    # "on_stopline_crosswalk",                                      # 自车位于斑马线停止线处
    # "stationary_at_crosswalk",                                    # 在斑马线处保持静止

    # --- 路口通行与转向行为 ---
    "starting_left_turn",                                         # 开始执行左转动作
    "starting_right_turn",                                        # 开始执行右转动作
    "starting_unprotected_cross_turn",                            # 开始执行无保护交叉转向（如无保护左转，需避让对向来车）
    "starting_straight_traffic_light_intersection_traversal",     # 开始在红绿灯路口直行通过
    "traversing_intersection",                                    # 正在穿越/通过交叉路口
    # "starting_low_speed_turn",                                    # 开始执行低速转弯
    # "starting_high_speed_turn",                                   # 开始执行高速转弯
    "starting_protected_cross_turn",                              # 开始执行有保护交叉转向（如绿灯左转）
    # "starting_protected_noncross_turn",                           # 开始执行有保护非交叉转向（如绿灯右转）
    # "starting_unprotected_noncross_turn",                         # 开始执行无保护非交叉转向
    # "starting_straight_stop_sign_intersection_traversal",         # 开始在停车标志路口直行通过
    # "traversing_traffic_light_intersection",                      # 正在穿越红绿灯路口
    # "on_intersection",                                            # 自车位于交叉路口内
    # "on_traffic_light_intersection",                              # 自车位于红绿灯交叉路口内
    # "on_all_way_stop_intersection",                               # 自车位于全向停车交叉路口

    # --- 车道变换与轨迹行为 ---
    # "changing_lane",                                              # 正在变换车道
    # "changing_lane_to_left",                                      # 正在向左变换车道
    # "changing_lane_to_right",                                     # 正在向右变换车道
    # "traversing_narrow_lane",                                     # 正在穿越狭窄车道

    # --- 车辆动力学与速度状态 ---
    # "low_magnitude_speed",                                        # 低速行驶状态
    # "medium_magnitude_speed",                                     # 中速行驶状态
    # "high_magnitude_speed",                                       # 高速行驶状态
    # "high_magnitude_jerk",                                        # 高加加速度（急加速/急减速）
    # "high_lateral_acceleration",                                  # 高横向加速度（急转弯）
    # "stationary",                                                 # 保持静止状态

    # --- 其他交通参与者交互（自行车等） ---
    # "behind_bike",                                                # 行驶在自行车后方
    # "near_high_speed_vehicle",                                    # 靠近高速行驶的车辆

    # --- 静态障碍物与特殊区域 ---
    "near_trafficcone_on_driveable",                              # 靠近可行驶区域内的交通锥（雪糕筒）
    "near_construction_zone",                                     # 靠近施工区域
    "near_construction_zone_sign",                                # 靠近施工区域标志
    # "near_barrier_on_driveable",                                  # 靠近可行驶区域内的路障
    # "on_pickup_dropoff",                                          # 自车位于上下客区域
    # "traversing_pickup_dropoff",                                  # 正在穿越上下客区域
    # "on_carpark",                                                 # 自车位于停车场内
}


def load_nuplan_events(database: str | Path) -> list[tuple[str, int]]:
    """Select internally configured scene types at the shared sample interval."""
    kinds = sorted(EVENT_TYPES)
    marks = ",".join("?" for _ in kinds)
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT DISTINCT st.type,lp.timestamp FROM scenario_tag st "
            "JOIN lidar_pc lp ON st.lidar_pc_token=lp.token "
            f"WHERE st.type IN ({marks}) ORDER BY st.type,lp.timestamp", kinds,
        ).fetchall()
    last, selected = {}, []
    for kind, timestamp in rows:
        if timestamp - last.get(kind, -10**30) >= EVENT_INTERVAL_US:
            selected.append((kind, int(timestamp)))
            last[kind] = timestamp
    return selected


def load_window_candidates(database: str | Path, history_us: int, future_us: int,
                           offsets_us=(-1_000_000, 0, 1_000_000), control_stride_us=3_000_000):
    """Event-relative anchors and background candidates, without reading images."""
    events = load_nuplan_events(database)
    candidates = {}
    for kind, timestamp in events:
        for offset in offsets_us:
            anchor = timestamp + offset
            key = (kind, anchor)
            # Prefer an exact event anchor when offsets from adjacent tags overlap.
            if key not in candidates or abs(offset) < abs(candidates[key][3]):
                candidates[key] = (kind, anchor, timestamp, offset)
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        tagged = [row[0] for row in connection.execute(
            "SELECT DISTINCT lp.timestamp FROM scenario_tag st "
            "JOIN lidar_pc lp ON st.lidar_pc_token=lp.token ORDER BY lp.timestamp")]
        times = [row[0] for row in connection.execute("SELECT timestamp FROM lidar_pc ORDER BY timestamp")]
    last = -10**30
    for anchor in times:
        if anchor - last < control_stride_us:
            continue
        lower, upper = anchor - history_us - 1_000_000, anchor + future_us + 1_000_000
        index = bisect_right(tagged, upper) - 1
        if index >= 0 and tagged[index] >= lower:
            continue
        candidates[("background_control", anchor)] = ("background_control", anchor, anchor, 0)
        last = anchor
    return sorted((item for item in candidates.values()
                   if times and times[0] <= item[1] - history_us
                   and item[1] + future_us <= times[-1]), key=lambda item: (item[1], item[0]))


def local_actions(trajectory: Tensor) -> Tensor:
    """Convert anchor-frame poses [..., T, 3] to preceding-ego-frame increments."""
    poses = F.pad(trajectory, (0, 0, 1, 0))
    delta = poses[..., 1:, :] - poses[..., :-1, :]
    yaw = poses[..., :-1, 2]
    return torch.stack((yaw.cos() * delta[..., 0] + yaw.sin() * delta[..., 1],
                        -yaw.sin() * delta[..., 0] + yaw.cos() * delta[..., 1],
                        torch.atan2(delta[..., 2].sin(), delta[..., 2].cos())), -1)


def load_nuplan_trajectory(
    database: str | Path,
    timestamp_us: int,
    *,
    history_seconds: float,
    future_seconds: float,
    max_pose_age_us: int = 100_000,
) -> dict[str, Any]:
    """Load past and recorded future expert poses in the anchor ego frame.

    The caller supplies horizons; FRAME_INTERVAL_US controls pose sampling.
    Future ego poses are demonstration targets, not original planner outputs.
    """
    interval_us = FRAME_INTERVAL_US
    if history_seconds < 0 or future_seconds < 0 or interval_us <= 0:
        raise ValueError("Trajectory horizons must be nonnegative and interval_us positive")
    database = Path(database).resolve()
    history_steps, history_remainder = divmod(round(history_seconds * 1_000_000), interval_us)
    future_steps, future_remainder = divmod(round(future_seconds * 1_000_000), interval_us)
    if history_remainder or future_remainder:
        raise ValueError("History and future horizons must be divisible by interval_us")
    anchor_targets = [timestamp_us + step * interval_us
                      for step in range(-history_steps, future_steps + 1)]
    lower, upper = anchor_targets[0] - max_pose_age_us, anchor_targets[-1]
    pose_columns = (
        "timestamp,x,y,z,qw,qx,qy,qz,vx,vy,vz,"
        "acceleration_x,acceleration_y,acceleration_z,"
        "angular_rate_x,angular_rate_y,angular_rate_z,epsg"
    )
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            f"SELECT {pose_columns} FROM ego_pose "
            "WHERE timestamp>=? AND timestamp<=? ORDER BY timestamp",
            (lower, upper),
        ).fetchall()
        if not rows:
            raise ValueError("No ego poses in the requested planning window")
        anchor_rows = [row for row in rows if row[0] <= timestamp_us]
        if not anchor_rows or timestamp_us - anchor_rows[-1][0] > max_pose_age_us:
            raise ValueError("Missing or stale anchor ego pose")
        anchor = anchor_rows[-1]
        anchor_timestamp = int(anchor[0])
        anchor_yaw = ego_pose_yaw(*map(float, anchor[4:8]))
        reference = _pose_matrix(
            [float(anchor[1]), float(anchor[2]), float(anchor[3])],
            [float(value) for value in anchor[4:8]],
        )
    trajectory = torch.zeros((len(anchor_targets), 3), dtype=torch.float32)
    trajectory_timestamps = torch.tensor(anchor_targets, dtype=torch.int64)
    trajectory_mask = torch.zeros(len(anchor_targets), dtype=torch.bool)
    pose_timestamps = torch.full((len(anchor_targets),), -1, dtype=torch.int64)
    row_times = [int(row[0]) for row in rows]
    for index, target in enumerate(anchor_targets):
        pose_index = bisect_right(row_times, target) - 1
        if pose_index < 0:
            continue
        row = rows[pose_index]
        if target - int(row[0]) > max_pose_age_us:
            continue
        point_global = torch.tensor(row[1:4], dtype=torch.float64)
        point_reference = transform_points_to_local_coordinate_3d(point_global[None], reference)[0]
        yaw = ego_pose_yaw(*map(float, row[4:8]))
        trajectory[index] = torch.tensor([
            float(point_reference[0]), float(point_reference[1]), wrap_angle(yaw - anchor_yaw)
        ])
        trajectory_mask[index] = True
        pose_timestamps[index] = int(row[0])
    anchor_index = history_steps
    trajectory[anchor_index] = 0.0
    trajectory_mask[anchor_index] = True

    return {
        "timestamp_us": int(timestamp_us),
        "anchor_pose_timestamp_us": anchor_timestamp,
        "trajectory_timestamps_us": trajectory_timestamps,
        "trajectory_mask": trajectory_mask,
        "pose_timestamps_us": pose_timestamps,
        "past_trajectory": trajectory[:anchor_index],
        "anchor_trajectory": trajectory[anchor_index],
        "future_expert_trajectory": trajectory[anchor_index + 1:],
        "future_expert_mask": trajectory_mask[anchor_index + 1:],
        "expert_trajectory_source": "recorded_ego_pose_proxy",
        "expert_planner_trajectory_available": False,
        "reference_ego_to_global": reference,
        "history_seconds": history_seconds,
        "future_seconds": future_seconds,
        "interval_us": interval_us,
    }

def load_nuplan_ego_state(
    database: str | Path, timestamp_us: int, *, max_pose_age_us: int = 100_000,
) -> dict[str, Any]:
    """Load current ego pose and dynamics; vectors are in the local ego frame."""
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        row = connection.execute(
            "SELECT timestamp,x,y,z,qw,qx,qy,qz,vx,vy,vz,"
            "acceleration_x,acceleration_y,acceleration_z,"
            "angular_rate_x,angular_rate_y,angular_rate_z FROM ego_pose "
            "WHERE timestamp<=? ORDER BY timestamp DESC LIMIT 1", (timestamp_us,),
        ).fetchone()
    if row is None or timestamp_us - row[0] > max_pose_age_us:
        raise ValueError("Missing or stale anchor ego pose")
    reference = _pose_matrix(row[1:4], row[4:8])
    global_to_ego = reference[:3, :3].T
    velocity = (global_to_ego @ torch.tensor(row[8:11], dtype=torch.float64)).float()
    acceleration = (global_to_ego @ torch.tensor(row[11:14], dtype=torch.float64)).float()
    return {
        "timestamp_us": timestamp_us,
        "anchor_pose_timestamp_us": int(row[0]),
        "position_global": torch.tensor(row[1:4], dtype=torch.float64),
        "yaw": torch.tensor(ego_pose_yaw(*row[4:8]), dtype=torch.float32),
        "velocity": velocity, "acceleration": acceleration,
        "angular_rate": torch.tensor(row[14:17], dtype=torch.float32),
        "speed": torch.linalg.vector_norm(velocity[:2]),
        "reference_ego_to_global": reference,
    }


def load_nuplan_route_condition(
    database: str | Path, timestamp_us: int,
) -> dict[str, Any]:
    """Read causal scenario metadata, not navigation route geometry.

    Scenario annotations are offline metadata; they are not inferred route
    commands and should not be treated as online sensor observations.
    """
    interval_us = EVENT_INTERVAL_US
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT DISTINCT st.type FROM scenario_tag st "
            "JOIN lidar_pc lp ON st.lidar_pc_token=lp.token "
            "WHERE lp.timestamp BETWEEN ? AND ? ORDER BY st.type",
            (timestamp_us - interval_us // 2, timestamp_us),
        ).fetchall()
    tags = tuple(row[0] for row in rows)
    selected = tuple(sorted(set(tags) & EVENT_TYPES))
    return {
        "timestamp_us": timestamp_us, "scenario_tags": tags,
        "selected_event_types": selected, "has_selected_event": bool(selected),
        "route_geometry_available": False, "route_command": None,
        "source": "scenario_tag",
    }



def load_nuplan_tracked_objects(
    database: str | Path, timestamp_us: int, *,
    reference_ego_to_global: Tensor, max_age_us: int = 100_000,
) -> dict[str, Any]:
    """Read the latest causal lidar annotation frame in a supplied ego frame.

    Track tokens associate objects across samples. Future calls are evaluation
    annotations only, never inputs to the planner at an earlier timestamp.
    Empty boxes with valid=True means an annotated frame without objects.
    """
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        frame = connection.execute(
            "SELECT token,timestamp FROM lidar_pc WHERE timestamp<=? "
            "ORDER BY timestamp DESC LIMIT 1", (timestamp_us,),
        ).fetchone()
        valid = frame is not None and timestamp_us - frame[1] <= max_age_us
        rows = connection.execute(
            "SELECT b.track_token,c.name,b.x,b.y,b.z,b.length,b.width,b.height,"
            "b.yaw,b.vx,b.vy,b.vz FROM lidar_box b "
            "JOIN track t ON b.track_token=t.token "
            "JOIN category c ON t.category_token=c.token WHERE b.lidar_pc_token=? "
            "ORDER BY b.track_token", (frame[0],),
        ).fetchall() if valid else []
    boxes = torch.empty((len(rows), 7), dtype=torch.float32)
    velocity = torch.empty((len(rows), 3), dtype=torch.float32)
    reference_yaw = torch.atan2(reference_ego_to_global[1, 0], reference_ego_to_global[0, 0])
    if rows:
        position = torch.tensor([r[2:5] for r in rows], dtype=torch.float64)
        local = transform_points_to_local_coordinate_3d(position, reference_ego_to_global)
        boxes[:, :3] = local.float()
        boxes[:, 3:6] = torch.tensor([r[5:8] for r in rows])
        boxes[:, 6] = wrap_angle(torch.tensor([r[8] for r in rows]) - reference_yaw).float()
        velocity[:] = (torch.tensor([r[9:12] for r in rows], dtype=torch.float64)
                       @ reference_ego_to_global[:3, :3]).float()
    return {
        "timestamp_us": timestamp_us,
        "annotation_timestamp_us": int(frame[1]) if frame else -1,
        "valid": bool(valid), "track_tokens": tuple(r[0] for r in rows),
        "categories": tuple(r[1] for r in rows), "boxes": boxes,
        "box_fields": ("x", "y", "z", "length", "width", "height", "yaw"),
        "velocity": velocity, "reference_ego_to_global": reference_ego_to_global,
    }


def load_nuplan_camera_frame(
    database: str | Path, sensor_blobs_root: str | Path, timestamp_us: int, *,
    max_image_age_us: int = 100_000,
    max_pose_age_us: int = 50_000,
    skip_unavailable: bool = False,
) -> dict[str, Any] | None:
    """Load the latest eight views at/before timestamp_us from a nuPlan log DB.

    sensor_blobs_root contains log-name/CAM_*/*.jpg. Each image uses its own
    camera calibration and capture pose; the reference is the latest ego pose
    at/before timestamp_us. Missing/stale images or poses raise ValueError,
    or return None when skip_unavailable=True for dataset filtering.
    Input JPGs are distorted. Extraction uses the DB intrinsics and distortion
    coefficients to rectify all eight views before encoding.
    """
    database = Path(database).resolve()
    root = Path(sensor_blobs_root)
    paths, rows = {}, []
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        reference = connection.execute(
            "SELECT timestamp,x,y,z,qw,qx,qy,qz FROM ego_pose "
            "WHERE timestamp<=? ORDER BY timestamp DESC LIMIT 1", (timestamp_us,)
        ).fetchone()
        if reference is None or timestamp_us-reference[0] > max_pose_age_us:
            if skip_unavailable:
                return None
            raise ValueError("Missing or stale reference ego pose")
        for camera in CAMERAS:
            row = connection.execute(
                "SELECT i.filename_jpg,i.timestamp,i.token,c.token,c.log_token,"
                "c.intrinsic,c.translation,c.rotation,c.distortion,c.width,c.height,"
                "p.timestamp,p.x,p.y,p.z,p.qw,p.qx,p.qy,p.qz "
                "FROM image i JOIN camera c ON i.camera_token=c.token "
                "JOIN ego_pose p ON i.ego_pose_token=p.token "
                "WHERE c.channel=? AND i.timestamp<=? "
                "ORDER BY i.timestamp DESC LIMIT 1", (camera, timestamp_us)
            ).fetchone()
            if row is None or timestamp_us-row[1] > max_image_age_us:
                if skip_unavailable:
                    return None
                raise ValueError(f"Missing or stale image for {camera}")
            if abs(row[11]-row[1]) > max_pose_age_us or row[11] > timestamp_us:
                if skip_unavailable:
                    return None
                raise ValueError(f"Invalid capture ego pose for {camera}")
            relative = Path(row[0])
            if relative.parts[0] != database.stem:
                relative = Path(database.stem) / relative
            paths[camera] = str(root / relative)
            rows.append(row)
    return {
        "database": str(database), "camera_order": CAMERAS,
        "image_paths": paths,
        "intrinsics": torch.stack([_decode_calibration(r[5]) for r in rows]),
        "camera_to_ego": torch.stack([_pose_matrix(_decode_calibration(r[6]), _decode_calibration(r[7])) for r in rows]),
        "distortion": tuple(_decode_calibration(r[8]).flatten() for r in rows),
        "original_shapes": tuple((int(r[10]), int(r[9]), 3) for r in rows),
        "image_tokens": tuple(r[2] for r in rows),
        "camera_tokens": tuple(r[3] for r in rows),
        "image_timestamps_us": torch.tensor([r[1] for r in rows], dtype=torch.int64),
        "camera_ego_to_global": torch.stack([_pose_matrix(r[12:15], r[15:19]) for r in rows]),
        "reference_ego_to_global": _pose_matrix(reference[1:4], reference[4:8]),
        "reference_pose_timestamp_us": int(reference[0]), "timestamp_us": timestamp_us,
    }

@torch.inference_mode()
def extract_camera_features(
    image_paths: Mapping[str, str | Path],
    image_encoder: nn.Module,
    *,
    image_size: tuple[int, int] = (WIDTH, HEIGHT),
    batch_size: int = 8,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract eight views from ONE time group in canonical camera order.

    Rectify raw images using calibration, resize with aspect ratio preserved,
    pad on the right/bottom, and normalize
    RGB with the Tiny checkpoint's ImageNet mean/std. ``image_size`` is (W, H),
    both multiples of 32. Missing views raise instead of becoming fake images.
    The returned CPU FP16 ``camera_features`` has shape [8, 256, H/32, W/32].
    Geometry metadata uses (height, width, channels); ``image_transforms`` maps
    original pixel coordinates to the resized/padded canvas. Nothing is saved.
    """
    if not isinstance(image_paths, Mapping):
        raise ValueError("image_paths must map camera names to image paths")
    paths = _ordered_cameras(image_paths, "image_paths", CAMERAS)
    if tuple(calibration["camera_order"]) != CAMERAS:
        raise ValueError("Calibration camera order differs from CAMERAS")
    expected_paths = _ordered_cameras(calibration["image_paths"], "calibration image_paths", CAMERAS)
    if any(Path(a).resolve() != Path(b).resolve() for a, b in zip(paths, expected_paths)):
        raise ValueError("Images do not match DB calibration frame")
    width, height = image_size
    if (not isinstance(batch_size, int) or batch_size < 1
            or any(not isinstance(size, int) or size <= 0 or size % 32
                   for size in image_size)):
        raise ValueError("batch_size must be positive; image_size must contain positive multiples of 32")
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensors, original_shapes, resized_shapes, transforms = [], [], [], []
    for camera_index, path in enumerate(paths):
        with Image.open(path) as image:
            original_w, original_h = image.size
            if tuple(calibration["original_shapes"][camera_index]) != (original_h, original_w, 3):
                raise ValueError(f"Image dimensions differ from DB for {CAMERAS[camera_index]}")
            scale = min(width / original_w, height / original_h)
            resized_w = min(width, max(1, round(original_w * scale)))
            resized_h = min(height, max(1, round(original_h * scale)))
            tensor = _rectified_resize(
                image, calibration["intrinsics"][camera_index],
                calibration["distortion"][camera_index], (resized_w, resized_h))
        tensors.append(F.pad((tensor - mean) / std,
                             (0, width - resized_w, 0, height - resized_h)))
        original_shapes.append((original_h, original_w, 3))
        resized_shapes.append((resized_h, resized_w, 3))
        transform = torch.diag(torch.tensor(
            [resized_w / original_w, resized_h / original_h, 1.0], dtype=torch.float64))
        transform[0, 2] = (resized_w / original_w - 1) / 2
        transform[1, 2] = (resized_h / original_h - 1) / 2
        transforms.append(transform)

    parameter = next(image_encoder.parameters())
    image_encoder.eval()
    features = []
    for start in range(0, 8, batch_size):
        batch = torch.stack(tensors[start:start + batch_size]).to(
            device=parameter.device, dtype=parameter.dtype)
        output = image_encoder(batch)
        if not isinstance(output, Tensor) or tuple(output.shape[1:]) != (256, height // 32, width // 32):
            raise ValueError("image_encoder must return Tiny FPN features [batch, 256, H/32, W/32]")
        features.append(output.to(device="cpu", dtype=torch.float16))
    return {
        "camera_features": torch.cat(features),
        "camera_order": CAMERAS,
        "source_paths": tuple(str(path) for path in paths),
        "original_shapes": tuple(original_shapes),
        "resized_shapes": tuple(resized_shapes),
        "image_shapes": ((height, width, 3),) * 8,
        "image_transforms": torch.stack(transforms),
        "calibration_database": calibration["database"],
        "image_tokens": calibration["image_tokens"],
        "rectification_applied": True,
    }

@torch.inference_mode()
def camera_bev_features(
    image_features: Mapping[str, Any] | list[Mapping[str, Any]],
    bev_encoder: nn.Module,
    calibration: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    batch_size: int = 1,
    max_image_age_us: int = 100_000,
) -> list[dict[str, Any]]:
    """Fuse several independent eight-camera frames in one BEVFormer call."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    single = isinstance(calibration, Mapping)
    image_features = [image_features] if single else list(image_features)
    calibrations = [calibration] if single else list(calibration)
    if len(image_features) != len(calibrations) or not image_features:
        raise ValueError("Batch image features and calibrations must be nonempty and aligned")
    if len(image_features) > batch_size:
        outputs = []
        for start in range(0, len(image_features), batch_size):
            outputs.extend(camera_bev_features(
                image_features[start:start + batch_size], bev_encoder,
                calibrations[start:start + batch_size], batch_size=batch_size,
                max_image_age_us=max_image_age_us))
        return outputs[0] if single else outputs
    batches = []
    projections = []
    metadata = []
    times = []
    for features, calibration in zip(image_features, calibrations):
        if tuple(calibration["camera_order"]) != CAMERAS:
            raise ValueError("Calibration camera order must equal CAMERAS")
        if (features.get("calibration_database") != calibration["database"]
                or features.get("image_tokens") != calibration["image_tokens"]
                or not features.get("rectification_applied")):
            raise ValueError("Features are not bound to the supplied DB calibration/images")
        value = torch.as_tensor(features["camera_features"])
        if value.ndim != 4 or value.shape[0] != 8:
            raise ValueError("camera_features must have shape [8,C,H,W]")
        batches.append(value)
        ego_to_image = build_ego_to_image_matrices(
            calibration["intrinsics"], calibration["camera_to_ego"],
            calibration["reference_ego_to_global"],
            camera_ego_to_global=calibration["camera_ego_to_global"], camera_order=CAMERAS)
        transform = _camera_tensor(features["image_transforms"], "image_transforms", (3, 3), CAMERAS)
        projection = ego_to_image.clone()
        projection[:, :3] = transform @ projection[:, :3]
        projections.append(projection.numpy())
        shape = _camera_tensor(features["image_shapes"], "image_shapes", (3,), CAMERAS)
        valid = _camera_tensor(features["resized_shapes"], "resized_shapes", (3,), CAMERAS)
        metadata.append({"lidar2img": projections[-1],
                         "img_shape": [tuple(int(v) for v in row) for row in shape],
                         "valid_shape": [tuple(int(v) for v in row) for row in valid],
                         "can_bus": np.zeros(18, dtype=np.float32)})
        capture_times = torch.as_tensor(calibration["image_timestamps_us"], dtype=torch.int64)
        if capture_times.shape != (8,) or (capture_times > calibration["timestamp_us"]).any() \
                or (calibration["timestamp_us"] - capture_times > max_image_age_us).any():
            raise ValueError("Invalid image capture times in BEV batch")
        times.append(capture_times)
    parameter = next(bev_encoder.parameters())
    transformer = bev_encoder.transformer
    if transformer.num_cams != 8 or batches[0].shape[1] != transformer.embed_dims:
        raise ValueError("BEV encoder and camera features have incompatible dimensions")
    values = torch.stack(batches).to(device=parameter.device, dtype=torch.float32)
    height, width = bev_encoder.bev_h, bev_encoder.bev_w
    pc_range = tuple(float(value) for value in bev_encoder.pc_range)
    tf32_matmul, tf32_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with ExitStack() as restore:
        restore.callback(setattr, torch.backends.cuda.matmul, "allow_tf32", tf32_matmul)
        restore.callback(setattr, torch.backends.cudnn, "allow_tf32", tf32_cudnn)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result = bev_encoder([values], metadata, prev_bev=None, only_bev=True)
        encoder = transformer.encoder
        references = encoder.get_reference_points(
            height, width, pc_range[5] - pc_range[2], encoder.num_points_in_pillar,
            dim="3d", bs=len(batches), device=parameter.device, dtype=torch.float32)
        _, visibility = encoder.point_sampling(references, pc_range, metadata)
    if result.shape[0] != len(batches):
        raise RuntimeError("BEVFormer batch output has an unexpected shape")
    x = pc_range[0] + (torch.arange(width, device="cpu") + 0.5) * (pc_range[3] - pc_range[0]) / width
    y = pc_range[1] + (torch.arange(height, device="cpu") + 0.5) * (pc_range[4] - pc_range[1]) / height
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    outputs = []
    for index in range(len(batches)):
        mask = visibility[:, index].any(dim=0).any(dim=-1).reshape(height, width)
        outputs.append({"bev_features": result[index].transpose(0, 1).reshape(transformer.embed_dims, height, width).cpu().half(),
                        "bev_valid_mask": mask.cpu(), "bev_xy": torch.stack((grid_x, grid_y), dim=-1),
                        "bev_range": pc_range, "bev_shape": (height, width),
                        "timestamp_us": int(calibrations[index]["timestamp_us"]),
                        "image_timestamps_us": times[index].cpu(), "camera_order": CAMERAS,
                        "calibration_database": calibrations[index]["database"],
                        "image_tokens": calibrations[index]["image_tokens"],
                        "reference_ego_to_global": calibrations[index]["reference_ego_to_global"],
                        "reference_pose_timestamp_us": calibrations[index]["reference_pose_timestamp_us"]})
    return outputs[0] if single else outputs


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    database_root = "../dataset/nuplan/nuplan-v1.1/splits/mini"
    sensor_blobs_root = "../dataset/nuplan/nuplan-v1.1/sensor_blobs/mini"
    bevformer_root = "3rdparty/BEVFormer"
    checkpoint = "3rdparty/BEVFormer/projects/ckpts/bevformer_tiny_epoch_24.pth"

    timestamp_us = 1_000_000
    frame = load_nuplan_camera_frame(database_root, sensor_blobs_root, timestamp_us)

    # 读取八路图像，完成校正和特征提取 
    image_encoder = load_backbone_encoder(checkpoint, device=device)
    features = extract_camera_features(frame["image_paths"], image_encoder, calibration=frame)

    # 利用内参、外参及各自拍摄位姿，融合到参考车辆坐标系
    source_indices = [0, 3, 2, 2, 4, 1, 1, 5]
    bev_encoder = load_bevformer_encoder(
        checkpoint,
        device=device,
        bevformer_root=bevformer_root,
        camera_embedding_indices=source_indices,
    )
    bev_features = camera_bev_features(features, bev_encoder, frame)
