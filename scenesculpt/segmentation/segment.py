"""
Assign object IDs to Gaussians using SAM masks from training images.

Two modes:
  1. auto_segment(): fully automatic — runs SAM's automatic mask generator
     on every training image, then votes across views.
  2. project_mask(): project a single user-supplied binary mask from one
     view into 3D (useful when the editor triggers per-click SAM queries).

Requires the 'segmentation' extras:
  pip install 'scenesculpt[segmentation]'
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from tqdm import tqdm

if TYPE_CHECKING:
    from scenesculpt.pipeline.run_colmap import Camera
    from scenesculpt.scene.gaussian import GaussianScene

console = Console()


# ─────────────────────────────────────────────────── SAM helpers


def _load_sam(checkpoint: str, model_type: str = "vit_h"):
    """Load a Segment Anything Model."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "Install the segmentation extras: pip install 'scenesculpt[segmentation]'"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return sam


def _project_gaussians_to_image(
    means: np.ndarray,     # [N, 3]
    K: np.ndarray,         # [3, 3]
    w2c: np.ndarray,       # [4, 4]
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D Gaussian centres to pixel coordinates.

    Returns:
      uv    [N, 2] int32  — pixel (col, row), clipped to image bounds
      valid [N]   bool    — True if the point is in front of the camera
    """
    # World → camera
    ones = np.ones((len(means), 1), dtype=np.float32)
    pts_h = np.concatenate([means, ones], axis=1)   # [N, 4]
    pts_c = (w2c @ pts_h.T).T                       # [N, 4]
    xyz = pts_c[:, :3]

    valid = xyz[:, 2] > 0.01

    # Camera → image
    proj = (K @ xyz[valid].T).T                     # [M, 3]
    uv = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)  # [M, 2] u=col, v=row

    u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, width - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, height - 1)

    uv_out = np.zeros((len(means), 2), dtype=np.int32)
    uv_out[valid] = np.stack([u, v], axis=1)
    return uv_out, valid


# ─────────────────────────────────────────────────── public API


def project_mask(
    scene: GaussianScene,
    mask: np.ndarray,     # [H, W] bool — 2D mask in one training view
    camera: Camera,
    object_id: int = 0,
) -> torch.Tensor:
    """
    Return a boolean tensor [N] selecting all Gaussians that project into
    the provided 2D binary mask in the given camera view.
    """
    means = scene.means.cpu().numpy()
    uv, valid = _project_gaussians_to_image(
        means, camera.K, camera.w2c, camera.width, camera.height
    )

    result = np.zeros(len(means), dtype=bool)
    valid_uv = uv[valid]
    in_mask = mask[valid_uv[:, 1], valid_uv[:, 0]]
    result[valid] = in_mask
    return torch.from_numpy(result)


def auto_segment(
    scene: GaussianScene,
    cameras: list[Camera],
    sam_checkpoint: str,
    sam_model_type: str = "vit_h",
    subsample_cameras: int = 10,
    points_per_side: int = 32,
) -> GaussianScene:
    """
    Fully automatic segmentation:
      1. Run SAM on a subset of training images.
      2. For each image, assign each Gaussian the mask ID it falls into.
      3. Majority vote across images → final object_id per Gaussian.

    Modifies scene.object_ids in-place and returns the scene.
    """
    sam = _load_sam(sam_checkpoint, sam_model_type)
    try:
        from segment_anything import SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError("pip install 'scenesculpt[segmentation]'")

    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
    )

    N = len(scene)
    means = scene.means.cpu().numpy()

    # votes[i][label] = number of views that assigned Gaussian i to label
    # We use a 2-D array: rows = Gaussians, cols = object labels (capped at max_labels)
    max_labels = 512
    vote_matrix = np.zeros((N, max_labels), dtype=np.int32)

    selected_cams = cameras[::max(1, len(cameras) // subsample_cameras)]

    for cam in tqdm(selected_cams, desc="SAM segmentation"):
        img = np.array(Image.open(cam.image_path).convert("RGB"))

        # Resize to camera resolution
        if img.shape[1] != cam.width or img.shape[0] != cam.height:
            img = np.array(
                Image.fromarray(img).resize((cam.width, cam.height), Image.LANCZOS)
            )

        masks_data = mask_gen.generate(img)
        # Sort largest-first so later (smaller) masks overwrite in the label image
        masks_data.sort(key=lambda x: x["area"], reverse=True)

        label_img = np.zeros((cam.height, cam.width), dtype=np.int32)
        for label_id, m in enumerate(masks_data[:max_labels - 1], start=1):
            label_img[m["segmentation"]] = label_id

        uv, valid = _project_gaussians_to_image(
            means, cam.K, cam.w2c, cam.width, cam.height
        )
        valid_uv = uv[valid]
        labels = label_img[valid_uv[:, 1], valid_uv[:, 0]]

        for gi, label in zip(np.where(valid)[0], labels):
            if label > 0:
                vote_matrix[gi, label] += 1

    # Assign by majority vote; uncontested Gaussians remain -1
    assigned = vote_matrix.max(axis=1)
    final_ids = np.where(assigned > 0, vote_matrix.argmax(axis=1), -1).astype(np.int32)

    # Remap to contiguous IDs starting at 0
    unique = np.unique(final_ids[final_ids >= 0])
    remap = {old: new for new, old in enumerate(unique)}
    remapped = np.array(
        [remap.get(int(i), -1) for i in final_ids], dtype=np.int32
    )
    scene.object_ids = torch.from_numpy(remapped)

    n_objects = len(unique)
    console.print(f"[green]Segmented {N:,} Gaussians into {n_objects} objects[/]")
    return scene
