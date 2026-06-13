"""Run COLMAP Structure-from-Motion and parse the reconstruction."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from rich.console import Console

console = Console()


@dataclass
class Camera:
    """A calibrated camera with its pose and image path."""
    image_path: Path
    K: np.ndarray        # [3, 3] intrinsic matrix
    w2c: np.ndarray      # [4, 4] world-to-camera transform
    width: int
    height: int

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c)


def run_colmap(
    image_dir: str | Path,
    workspace_dir: str | Path,
    colmap_bin: str = "colmap",
    sequential: bool = True,
) -> Path:
    """
    Run the full COLMAP SfM pipeline:
      feature_extractor → matcher → mapper → model_converter

    Returns the path to the sparse reconstruction directory.
    """
    image_dir = Path(image_dir)
    workspace_dir = Path(workspace_dir)
    db_path = workspace_dir / "colmap.db"
    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # COLMAP is a Qt app — needs offscreen platform in headless environments
    headless_env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}

    def _run(step: str, args: list[str]) -> None:
        console.print(f"[cyan]COLMAP:[/] {step}")
        result = subprocess.run(
            [colmap_bin] + args,
            capture_output=True,
            text=True,
            env=headless_env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"COLMAP {step} failed:\n{result.stderr[-2000:]}"
            )

    _run("feature_extractor", [
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0",   # GPU SIFT needs OpenGL; CPU works headless
    ])

    matcher = "sequential_matcher" if sequential else "exhaustive_matcher"
    _run(matcher, [
        matcher,
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "0",
    ])

    _run("mapper", [
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ])

    # Pick the largest reconstruction (model 0 is usually the main one)
    recon_dir = sparse_dir / "0"
    if not recon_dir.exists():
        raise RuntimeError("COLMAP mapper produced no reconstruction in sparse/0")

    console.print(f"[green]COLMAP finished. Reconstruction at {recon_dir}[/]")
    return recon_dir


def load_cameras(recon_dir: str | Path, image_dir: str | Path) -> list[Camera]:
    """
    Parse a COLMAP binary reconstruction into a list of Camera objects.
    """
    recon_dir = Path(recon_dir)
    image_dir = Path(image_dir)

    recon = pycolmap.Reconstruction(str(recon_dir))
    cameras: list[Camera] = []

    for image_id, img in recon.images.items():
        if not img.has_pose:  # bool property in pycolmap >= 0.6
            continue
        cam = recon.cameras[img.camera_id]

        # Intrinsics
        K = _camera_to_K(cam)
        w, h = cam.width, cam.height

        # Extrinsics: COLMAP gives R, t such that x_cam = R @ x_world + t
        # pycolmap >= 0.6 exposes this via cam_from_world (a Rigid3d object)
        cam_from_world = img.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = np.array(cam_from_world.translation, dtype=np.float64)

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        img_path = image_dir / img.name
        if not img_path.exists():
            continue

        cameras.append(Camera(
            image_path=img_path,
            K=K.astype(np.float32),
            w2c=w2c,
            width=w,
            height=h,
        ))

    cameras.sort(key=lambda c: c.image_path.name)
    console.print(f"[green]Loaded {len(cameras)} cameras[/]")
    return cameras


def load_point_cloud(recon_dir: str | Path) -> np.ndarray:
    """Return the sparse COLMAP point cloud as [M, 3] float32."""
    recon = pycolmap.Reconstruction(str(recon_dir))
    pts = np.array(
        [p.xyz for p in recon.points3D.values()],
        dtype=np.float32,
    )
    return pts


# ------------------------------------------------------------------ helpers

def _camera_to_K(cam: pycolmap.Camera) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    model = cam.model.name if hasattr(cam.model, "name") else str(cam.model)

    if "SIMPLE_PINHOLE" in model:
        f, cx, cy = cam.params
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif "PINHOLE" in model:
        fx, fy, cx, cy = cam.params
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
    elif "SIMPLE_RADIAL" in model or "RADIAL" in model:
        # Use only the focal / principal point; ignore distortion
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif "OPENCV" in model:
        fx, fy, cx, cy = cam.params[:4]
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")

    return K
