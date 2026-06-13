"""
Run SAM-based object segmentation on a trained scene stored in Modal Volume.

This assigns an object_id to every Gaussian by projecting SAM masks from
multiple training views and taking a majority vote.

Setup (one-time):
    pip install segment-anything

Run (after modal_train.py has finished):
    modal run modal_segment.py --scene bicycle

Download result:
    modal volume get scenesculpt-scenes bicycle/splats_segmented.ply ./splats.ply

Then edit with object-select mode:
    python scripts/launch_editor.py --scene splats.ply
"""

from __future__ import annotations

from pathlib import Path

import modal

# Reuse the same CUDA image, plus segment-anything
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .env({"QT_QPA_PLATFORM": "offscreen"})
    .apt_install(["libgl1", "libglib2.0-0"])
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install([
        "segment-anything>=1.0",
        "pycolmap>=0.6.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24,<2",
        "Pillow>=10.0",
        "scipy>=1.11",
        "plyfile>=1.0",
        "rich>=13.0",
        "tqdm>=4.65",
        "natsort>=8.4",
        "pycocotools>=2.0",
    ])
    .add_local_python_source("scenesculpt")
)

app = modal.App("scenesculpt-segment", image=image)

vol = modal.Volume.from_name("scenesculpt-scenes", create_if_missing=True)
REMOTE_ROOT = Path("/scenes")
SAM_CHECKPOINT = REMOTE_ROOT / "sam_vit_h_4b8939.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


@app.function(
    gpu="A10G",
    timeout=7200,
    volumes={REMOTE_ROOT: vol},
)
def segment(scene_name: str, subsample_cameras: int = 10) -> None:
    import urllib.request
    from scenesculpt.scene.gaussian import GaussianScene
    from scenesculpt.pipeline.run_colmap import load_cameras
    from scenesculpt.segmentation.segment import auto_segment

    scene_dir = REMOTE_ROOT / scene_name

    # Download SAM checkpoint on first run (cached in Volume afterwards)
    if not SAM_CHECKPOINT.exists():
        print(f"Downloading SAM ViT-H checkpoint (2.4 GB) — only happens once…")
        urllib.request.urlretrieve(SAM_URL, str(SAM_CHECKPOINT))
        vol.commit()
        print("Checkpoint saved to volume.")

    # Load trained scene
    ply_path = scene_dir / "splats.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"No splats.ply found in {scene_dir}. Run training first.")
    scene = GaussianScene.load(ply_path)
    print(f"Loaded scene: {len(scene):,} Gaussians")

    # Load cameras from the persisted COLMAP reconstruction
    colmap_sparse = scene_dir / "colmap_sparse"
    frames_dir = scene_dir / "frames"
    if not colmap_sparse.exists() or not frames_dir.exists():
        raise FileNotFoundError(
            "frames/ or colmap_sparse/ not found in volume. "
            "Re-run training with the updated modal_train.py."
        )
    cameras = load_cameras(colmap_sparse, frames_dir)
    print(f"Loaded {len(cameras)} cameras")

    # Run SAM segmentation
    auto_segment(
        scene,
        cameras,
        sam_checkpoint=str(SAM_CHECKPOINT),
        sam_model_type="vit_h",
        subsample_cameras=subsample_cameras,
    )

    n_objects = len(scene.unique_object_ids())
    print(f"Found {n_objects} objects")

    # Save as a separate file so the original is preserved
    out_path = scene_dir / "splats_segmented.ply"
    scene.save(out_path)
    vol.commit()
    print(f"Saved → {out_path}")


@app.local_entrypoint()
def main(
    scene: str,
    subsample_cameras: int = 10,
) -> None:
    segment.remote(scene, subsample_cameras=subsample_cameras)

    print(f"\n✅  Segmentation complete!")
    print(f"Download:")
    print(f"  modal volume get scenesculpt-scenes {scene}/splats_segmented.ply ./splats.ply")
    print(f"Launch editor:")
    print(f"  python scripts/launch_editor.py --scene splats.ply")
