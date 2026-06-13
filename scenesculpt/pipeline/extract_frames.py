"""Extract frames from a phone video for use with COLMAP."""

from __future__ import annotations

from pathlib import Path

import cv2
from rich.console import Console
from tqdm import tqdm

console = Console()


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 2.0,
    max_frames: int = 300,
    min_blur_threshold: float = 100.0,
) -> list[Path]:
    """
    Sample frames from a video at the target FPS, skipping blurry frames.

    Returns the list of saved image paths.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(video_fps / fps)))

    console.print(
        f"[cyan]Video:[/] {video_path.name}  "
        f"[cyan]FPS:[/] {video_fps:.1f}  "
        f"[cyan]Frames:[/] {total_frames}  "
        f"[cyan]Sample every:[/] {frame_interval}"
    )

    saved: list[Path] = []
    frame_idx = 0
    pbar = tqdm(total=min(max_frames, total_frames // frame_interval), desc="Extracting frames")

    while cap.isOpened() and len(saved) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Skip blurry frames (Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score >= min_blur_threshold:
                out_path = output_dir / f"frame_{len(saved):06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved.append(out_path)
                pbar.update(1)

        frame_idx += 1

    cap.release()
    pbar.close()
    console.print(f"[green]Saved {len(saved)} frames to {output_dir}[/]")
    return saved
