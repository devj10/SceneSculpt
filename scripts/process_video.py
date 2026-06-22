"""
End-to-end pipeline: phone video → trained Gaussian Splatting scene.

  python scripts/process_video.py --video path/to/video.mp4 --output ./my_scene
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.console import Console

console = Console()


@dataclass
class Config:
    video: Path
    """Input video file."""

    output: Path = Path("output")
    """Directory to write all artefacts into."""

    fps: float = 2.0
    """Frames to extract per second of video."""

    max_frames: int = 250
    """Hard cap on extracted frames."""

    colmap_bin: str = "colmap"
    """Path to COLMAP binary (must be on PATH or absolute)."""

    train_iterations: int = 30_000
    """Number of Gaussian Splatting training iterations."""

    sam_checkpoint: str | None = None
    """Optional: path to SAM checkpoint for automatic segmentation after training."""


def main() -> None:
    cfg = tyro.cli(Config)

    from scenesculpt.pipeline.extract_frames import extract_frames
    from scenesculpt.pipeline.run_colmap import run_colmap, load_cameras, load_point_cloud
    from scenesculpt.pipeline.train_splats import GaussianTrainer, TrainConfig

    frames_dir = cfg.output / "frames"
    colmap_dir = cfg.output / "colmap"
    splats_dir = cfg.output / "splats"

    # Extract frames
    console.rule("[bold cyan]Step 1 / 3 — Frame extraction[/]")
    extract_frames(cfg.video, frames_dir, fps=cfg.fps, max_frames=cfg.max_frames)

    # COLMAP SfM
    console.rule("[bold cyan]Step 2 / 3 — Structure from Motion (COLMAP)[/]")
    recon_dir = run_colmap(frames_dir, colmap_dir, colmap_bin=cfg.colmap_bin, sequential=True)
    cameras = load_cameras(recon_dir, frames_dir)
    point_cloud = load_point_cloud(recon_dir)
    console.print(f"Sparse point cloud: [green]{len(point_cloud):,}[/] points")

    # Train Gaussians
    console.rule("[bold cyan]Step 3 / 3 — Gaussian Splatting training[/]")
    train_cfg = TrainConfig(num_iterations=cfg.train_iterations)
    trainer = GaussianTrainer(cameras, splats_dir, point_cloud=point_cloud, config=train_cfg)
    scene = trainer.train()

    # Auto-segment with SAM
    if cfg.sam_checkpoint:
        console.rule("[bold cyan]Optional — SAM segmentation[/]")
        from scenesculpt.segmentation.segment import auto_segment
        auto_segment(scene, cameras, sam_checkpoint=cfg.sam_checkpoint)
        scene.save(splats_dir / "splats.ply")
        console.print("[green]Segmented scene saved (with object_ids).[/]")

    console.rule("[bold green]Done[/]")
    console.print(
        f"Launch the editor:\n"
        f"  [bold]python scripts/launch_editor.py --scene {splats_dir / 'splats.ply'}[/]"
    )


if __name__ == "__main__":
    main()
