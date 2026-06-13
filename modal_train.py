"""
Run COLMAP + Gaussian Splatting training on Modal cloud GPUs.

Setup (one-time):
    pip install modal
    modal setup          # authenticates via browser

Run:
    modal run modal_train.py --video my_video.mp4
    modal run modal_train.py --video my_video.mp4 --scene office --gpu A100 --iterations 30000

Download result:
    modal volume get scenesculpt-scenes <scene-name>/splats.ply ./splats.ply

Then edit locally:
    python scripts/launch_editor.py --scene splats.ply
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import modal

# ─────────────────────────────────────────────────── container image

# nvidia/cuda devel image is required so gsplat can compile its CUDA kernels
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .env({"QT_QPA_PLATFORM": "offscreen"})   # COLMAP headless
    .apt_install([
        "colmap",           # SfM
        "libgl1",           # OpenCV headless needs this
        "libglib2.0-0",
        "ffmpeg",
        "git",
        "ninja-build",      # speeds up gsplat CUDA compilation
    ])
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install([
        "gsplat==1.4.0",            # triggers CUDA compilation at image build time
        "pycolmap>=0.6.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24,<2",           # pycolmap compiled against NumPy 1.x
        "Pillow>=10.0",
        "tqdm>=4.65",
        "scipy>=1.11",
        "plyfile>=1.0",
        "rich>=13.0",
        "torchmetrics[image]>=1.2",
        "natsort>=8.4",
        "tyro>=0.7",
    ])
    .add_local_python_source("scenesculpt")   # copies your local package into the image
)

app = modal.App("scenesculpt-train", image=image)

# Persistent volume — scenes survive across runs
vol = modal.Volume.from_name("scenesculpt-scenes", create_if_missing=True)
REMOTE_ROOT = Path("/scenes")


# ─────────────────────────────────────────────────── remote function

@app.function(
    gpu="A10G",            # ~$0.76/hr; swap for "A100" (~$3/hr) for faster training
    timeout=10_800,        # 3 hours max
    volumes={REMOTE_ROOT: vol},
)
def run_pipeline(
    frames: list[tuple[str, bytes]],
    scene_name: str,
    train_iterations: int = 30_000,
) -> None:
    from scenesculpt.pipeline.run_colmap import run_colmap, load_cameras, load_point_cloud
    from scenesculpt.pipeline.train_splats import GaussianTrainer, TrainConfig

    frames_dir = Path(f"/tmp/{scene_name}/frames")
    frames_dir.mkdir(parents=True)

    for name, data in frames:
        (frames_dir / name).write_bytes(data)

    colmap_dir = Path(f"/tmp/{scene_name}/colmap")
    out_dir = REMOTE_ROOT / scene_name

    print(f"Running COLMAP on {len(frames)} frames…")
    recon_dir = run_colmap(frames_dir, colmap_dir, sequential=True)
    cameras = load_cameras(recon_dir, frames_dir)
    point_cloud = load_point_cloud(recon_dir)
    print(f"Point cloud: {len(point_cloud):,} points, {len(cameras)} cameras")

    cfg = TrainConfig(num_iterations=train_iterations)
    GaussianTrainer(cameras, out_dir, point_cloud=point_cloud, config=cfg).train()

    # Persist frames + COLMAP sparse reconstruction so segmentation can run later
    import shutil
    vol_frames = out_dir / "frames"
    vol_frames.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.iterdir():
        shutil.copy(f, vol_frames / f.name)
    shutil.copytree(str(recon_dir), str(out_dir / "colmap_sparse"), dirs_exist_ok=True)

    vol.commit()


# ─────────────────────────────────────────────────── local entrypoint

@app.local_entrypoint()
def main(
    video: str,
    scene: str = "scene01",
    fps: float = 2.0,
    max_frames: int = 200,
    iterations: int = 30_000,
    gpu: str = "A10G",
) -> None:
    from scenesculpt.pipeline.extract_frames import extract_frames

    # Override GPU dynamically
    run_pipeline.with_options(gpu=gpu)

    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Extract frames locally (no GPU needed)
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp) / "frames"
        paths = extract_frames(video_path, frames_dir, fps=fps, max_frames=max_frames)

        print(f"Uploading {len(paths)} frames to Modal…")
        frames_data = [(p.name, p.read_bytes()) for p in paths]

    # Ship to Modal
    run_pipeline.remote(frames_data, scene, train_iterations=iterations)

    print("\n✅  Training complete!")
    print(f"Download your scene:")
    print(f"  modal volume get scenesculpt-scenes {scene}/splats.ply ./splats.ply")
    print(f"Launch the editor:")
    print(f"  python scripts/launch_editor.py --scene splats.ply")
