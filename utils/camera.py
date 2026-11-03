"""nuPlan calibration, image geometry and frozen BEVFormer encoder utilities.

Load encoders once and reuse them across frames. Preprocessing derives a
target-to-source mapping from calibration; transferred weights still need
validation on nuPlan. Full BEV fusion requires the vendor OpenMMLab environment.
"""

from __future__ import annotations

import importlib
import io
import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F


class _CalibrationUnpickler(pickle.Unpickler):
    """Decode nuPlan list wrappers without importing its ORM or arbitrary globals."""

    def find_class(self, module, name):
        if module == "nuplan.database.common.data_types" and name in {
            "Translation", "Rotation", "CameraIntrinsic",
        }:
            return list
        allowed = {
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
        }
        if (module, name) in allowed:
            return allowed[module, name]
        raise ValueError(f"Unsupported calibration serialization: {module}.{name}")


def _decode_calibration(blob: bytes) -> Tensor:
    value = _CalibrationUnpickler(io.BytesIO(blob)).load()
    result = torch.as_tensor(np.asarray(value, dtype=np.float64))
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite nuPlan calibration")
    return result


def _pose_matrix(translation: Any, quaternion_wxyz: Any) -> Tensor:
    """nuPlan quaternion [w,x,y,z] and translation -> local-to-parent SE(3)."""
    t = torch.as_tensor(translation, dtype=torch.float64)
    q = torch.as_tensor(quaternion_wxyz, dtype=torch.float64)
    if (t.shape != (3,) or q.shape != (4,) or not torch.isfinite(t).all()
            or not torch.isfinite(q).all() or abs(q.norm().item() - 1) > 1e-3):
        raise ValueError("Expected finite translation and unit quaternion [w,x,y,z]")
    w, x, y, z = q / q.norm()
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    )).reshape(3, 3)
    matrix[:3, 3] = t
    return matrix


def _rectified_resize(image: Image.Image, intrinsic: Tensor, distortion: Tensor,
                      size: tuple[int, int]) -> Tensor:
    """Sample distorted RGB into a resized pinhole image with the SAME original K."""
    width, height = size
    original_w, original_h = image.size
    k = intrinsic.float()
    y, x = torch.meshgrid(torch.arange(height).float(), torch.arange(width).float(), indexing="ij")
    # Output raster pixel centers -> original pinhole raster pixel centers.
    u = (x + 0.5) * original_w / width - 0.5
    v = (y + 0.5) * original_h / height - 0.5
    rays = torch.linalg.solve(k, torch.stack((u, v, torch.ones_like(u))).flatten(1)).reshape(3, height, width)
    xn, yn = rays[0] / rays[2], rays[1] / rays[2]
    k1, k2, p1, p2, k3 = distortion.float()
    r2 = xn*xn + yn*yn
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xd = xn*radial + 2*p1*xn*yn + p2*(r2+2*xn*xn)
    yd = yn*radial + p1*(r2+2*yn*yn) + 2*p2*xn*yn
    projected = k @ torch.stack((xd, yd, torch.ones_like(xd))).flatten(1)
    projected = projected.reshape(3, height, width)
    grid = torch.stack((2*(projected[0]/projected[2]+0.5)/original_w-1,
                        2*(projected[1]/projected[2]+0.5)/original_h-1), dim=-1)
    source = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).float()/255
    return F.grid_sample(source[None], grid[None], mode="bilinear",
                         padding_mode="zeros", align_corners=False)[0]


def _checkpoint_state(checkpoint: str | Path) -> dict[str, Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    return {key.removeprefix("module."): value for key, value in state.items()}


def _ordered_cameras(value: Any, name: str, camera_order: Sequence[str] | None = None) -> Any:
    if isinstance(value, Mapping):
        if camera_order is None:
            raise ValueError(f"Specify camera_order for mapping input {name}")
        if set(value) != set(camera_order):
            raise ValueError(f"{name} must contain exactly these cameras: {camera_order}")
        return [value[camera] for camera in camera_order]
    if len(value) != 8:
        raise ValueError(f"{name} must have eight entries in the supplied camera order")
    return value


def _camera_tensor(value: Any, name: str, shape: tuple[int, ...],
                   camera_order: Sequence[str] | None = None) -> Tensor:
    value = _ordered_cameras(value, name, camera_order)
    if not isinstance(value, (Tensor, np.ndarray)):
        value = torch.stack([torch.as_tensor(item) for item in value])
    value = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if tuple(value.shape) != (8, *shape) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {(8, *shape)}")
    return value


def load_backbone_encoder(
    checkpoint: str | Path, *, device: str | torch.device = "cpu"
) -> nn.Module:
    """Load and freeze the Tiny checkpoint's ResNet-50 + single-level FPN.

    This stage needs torch/torchvision, not MMCV. Computation is FP32; extraction
    converts the returned cache to FP16. No pretrained downloads are performed.
    """
    from torchvision.models import resnet50

    state = _checkpoint_state(checkpoint)
    backbone = resnet50(weights=None)
    backbone.fc = nn.Identity()
    backbone.load_state_dict({
        key.removeprefix("img_backbone."): value
        for key, value in state.items() if key.startswith("img_backbone.")
    }, strict=True)
    lateral = nn.Conv2d(2048, 256, 1)
    fpn = nn.Conv2d(256, 256, 3, padding=1)
    for module, prefix in (
        (lateral, "img_neck.lateral_convs.0.conv."),
        (fpn, "img_neck.fpn_convs.0.conv."),
    ):
        module.load_state_dict({
            key.removeprefix(prefix): value
            for key, value in state.items() if key.startswith(prefix)
        }, strict=True)
    encoder = nn.Sequential(*list(backbone.children())[:-2], lateral, fpn)
    return encoder.to(device=device, dtype=torch.float32).requires_grad_(False).eval()


def build_ego_to_image_matrices(
    intrinsics: Any,
    camera_to_ego: Any,
    reference_ego_to_global: Any,
    *,
    camera_ego_to_global: Any | None = None,
    camera_order: Sequence[str] | None = None,
) -> Tensor:
    """Build [8, 4, 4] projections to ORIGINAL, rectified image pixels.

    ``intrinsics`` is [8,3,3]; ``camera_to_ego`` is [8,4,4], from optical camera
    coordinates (x right, y down, z forward) to the ego frame at capture time.
    ``reference_ego_to_global`` is [4,4], the BEV reference pose. When captures
    are asynchronous, pass their individual ego-to-global poses [8,4,4] through
    ``camera_ego_to_global``. Omitting them asserts all views share the reference
    pose. All camera-indexed arguments also accept dicts keyed by camera_order.
    No rear-axle/body-frame offset or distortion correction is guessed here.
    """
    intrinsic = _camera_tensor(intrinsics, "intrinsics", (3, 3), camera_order)
    extrinsic = _camera_tensor(camera_to_ego, "camera_to_ego", (4, 4), camera_order)
    reference_pose = torch.as_tensor(reference_ego_to_global, dtype=torch.float64, device="cpu")
    if reference_pose.shape != (4, 4) or not torch.isfinite(reference_pose).all():
        raise ValueError("reference_ego_to_global must be finite with shape [4,4]")
    capture_poses = (reference_pose.expand(8, -1, -1) if camera_ego_to_global is None
                     else _camera_tensor(camera_ego_to_global, "camera_ego_to_global", (4, 4), camera_order))
    # A point in the reference ego frame is taken to global, then to the
    # capture ego frame and finally to the optical camera frame.  ``extrinsic``
    # is camera -> ego, so its inverse is used here.
    transform = torch.linalg.solve(
        extrinsic, torch.linalg.solve(capture_poses, reference_pose.expand(8, -1, -1))
    )
    projection = torch.eye(4, dtype=torch.float64).repeat(8, 1, 1)
    projection[:, :3, :3] = intrinsic
    return projection @ transform


def _remap_camera_embeddings(
    embeddings: Tensor, source_indices: Sequence[int] | None
) -> Tensor:
    if source_indices is None:
        if embeddings.shape[0] != 8:
            raise ValueError(
                f"Checkpoint has {embeddings.shape[0]} camera embeddings. Supply eight "
                "camera_embedding_indices in the supplied camera order, based on the source "
                "checkpoint's camera order and the target camera directions."
            )
        return embeddings
    indices = torch.as_tensor(source_indices)
    if (indices.shape != (8,) or indices.dtype not in (torch.int32, torch.int64)
            or (indices < 0).any() or (indices >= embeddings.shape[0]).any()):
        raise ValueError("camera_embedding_indices must contain eight valid integer source indices")
    return embeddings[indices.long()].clone()


def camera_embedding_indices_from_calibration(camera_to_ego: Tensor) -> list[int]:
    """Match target optical +Z axes to nominal nuScenes camera headings.

    Source order follows the bundled nuscenes_converter: FRONT, FRONT_RIGHT,
    FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT. Headings are approximate, in an
    ego frame with X forward and Y left; these are not source calibrations.
    """
    axes = torch.as_tensor(camera_to_ego, dtype=torch.float64)[:, :2, 2]
    if axes.shape != (8, 2) or not torch.isfinite(axes).all() or (axes.norm(dim=1) < 1e-6).any():
        raise ValueError("Expected eight camera-to-ego poses with horizontal optical axes")
    headings = torch.atan2(axes[:, 1], axes[:, 0])
    source_headings = torch.deg2rad(headings.new_tensor([0, -55, 55, 180, 110, -110]))
    difference = headings[:, None] - source_headings
    distances = torch.atan2(difference.sin(), difference.cos()).abs()
    return distances.argmin(dim=1).tolist()


def _point_sampling_without_padding(encoder, reference_points, pc_range, img_metas):
    """Use vendor projection, excluding the right/bottom padded image region."""
    coordinates, visible = type(encoder).point_sampling(encoder, reference_points, pc_range, img_metas)
    ratios = coordinates.new_tensor([
        [(valid[1] / padded[1], valid[0] / padded[0])
         for valid, padded in zip(meta["valid_shape"], meta["img_shape"])]
        for meta in img_metas
    ]).permute(1, 0, 2)[:, :, None, None, :]
    visible = visible & (coordinates < ratios).all(-1)
    return coordinates, visible


def load_bevformer_encoder(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    bevformer_root: str | Path,
    camera_embedding_indices: Sequence[int] | None = None,
) -> nn.Module:
    """Load the official BEVFormer head for frozen, eight-view BEV extraction.

    Resolve the Tiny config and projects.mmdet3d_plugin from bevformer_root.

    Requires the vendored project's MMCV 1.x / MMDetection 2.x / MMDetection3D
    stack and compiled deformable-attention ops (see BEVFormer/docs/install.md).
    The full head is loaded strictly, but only its BEV encoder executes. Calls
    are stateless: prev_bev=None. The existing temporal-attention layers operate
    on current queries; there is no cross-frame memory or nuScenes CAN bus input.

    Six-camera weights need an explicit mapping; duplicated source embeddings
    are transfer initialization, not a calibrated/trained eight-view model.
    """
    bevformer_root = Path(bevformer_root).resolve()
    config_path = bevformer_root / "projects/configs/bevformer/bevformer_tiny.py"
    state = _checkpoint_state(checkpoint)
    prefix = "pts_bbox_head."
    head_state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    camera_key = "transformer.cams_embeds"
    if camera_key not in head_state:
        raise ValueError("Checkpoint must contain a full BEVFormer pts_bbox_head, not only backbone/FPN weights")
    source_camera_embeddings = head_state[camera_key].clone()
    head_state[camera_key] = _remap_camera_embeddings(source_camera_embeddings, camera_embedding_indices)
    from mmcv import Config
    from mmdet.models import build_head

    root = str(bevformer_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    importlib.import_module("projects.mmdet3d_plugin")
    config = Config.fromfile(str(config_path))
    head_config = config.model.pts_bbox_head.copy()
    # Building the head directly bypasses the detector that normally supplies
    # train_cfg. Inference needs no matcher; avoid DETR's default train config.
    head_config.train_cfg = None
    transformer = head_config.transformer
    transformer.num_cams = 8
    transformer.use_can_bus = False
    transformer.use_shift = False
    transformer.rotate_prev_bev = False
    transformer.encoder.return_intermediate = False
    layers = transformer.encoder.transformerlayers
    for layer in layers if isinstance(layers, (list, tuple)) else [layers]:
        for attention in layer.attn_cfgs:
            if attention.type == "SpatialCrossAttention":
                attention.num_cams = 8
    head = build_head(head_config)
    head.load_state_dict(head_state, strict=True)
    head.register_buffer("source_camera_embeddings", source_camera_embeddings, persistent=False)
    encoder = head.transformer.encoder
    encoder.point_sampling = MethodType(_point_sampling_without_padding, encoder)
    return head.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
