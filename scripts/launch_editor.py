"""
Launch the interactive 3D Gaussian scene editor.

  python scripts/launch_editor.py --scene output/splats/splats.ply
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.console import Console

console = Console()


@dataclass
class Config:
    scene: Path
    """Path to the .ply Gaussian scene file."""

    port: int = 8080
    """Port for the viser web UI."""


def main() -> None:
    cfg = tyro.cli(Config)

    if not cfg.scene.exists():
        console.print(f"[red]Scene file not found: {cfg.scene}[/]")
        raise SystemExit(1)

    from scenesculpt.editor.viewer import SceneEditor
    editor = SceneEditor(cfg.scene, port=cfg.port)
    editor.run()


if __name__ == "__main__":
    main()
