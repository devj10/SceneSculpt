"""
Interactive 3D Gaussian Splatting scene editor built on viser.

Usage:
    editor = SceneEditor("output/splats.ply")
    editor.run()   # opens browser at http://localhost:8080

Controls:
  Click   — select Gaussians within the configured radius
  Shift+click — expand current selection
  GUI panel — delete / recolor / translate / save
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch
import viser
from rich.console import Console
from scipy.spatial import KDTree

from scenesculpt.scene.gaussian import GaussianScene

console = Console()

# Highlight colour shown on selected Gaussians
HIGHLIGHT_RGB = np.array([1.0, 0.45, 0.0], dtype=np.float32)


class SceneEditor:
    def __init__(self, scene_path: str | Path, port: int = 8080) -> None:
        self.path = Path(scene_path)
        self.scene = GaussianScene.load(self.path)
        self.server = viser.ViserServer(port=port)
        self._lock = threading.Lock()

        N = len(self.scene)
        self._selected = np.zeros(N, dtype=bool)
        self._kdtree: KDTree | None = None

        console.print(
            f"[bold cyan]SceneEditor[/] loaded [green]{N:,}[/] Gaussians from "
            f"[dim]{self.path}[/]"
        )
        console.print(f"[bold]Open browser → [link=http://localhost:{port}]http://localhost:{port}[/][/]")

        self._upload_splats()
        self._setup_gui()

    # ──────────────────────────────────────── KDTree (lazy, invalidated on edit)

    @property
    def kdtree(self) -> KDTree:
        if self._kdtree is None:
            self._kdtree = KDTree(self.scene.means.numpy())
        return self._kdtree

    def _invalidate_kdtree(self) -> None:
        self._kdtree = None

    # ──────────────────────────────────────── rendering

    def _display_colors(self) -> np.ndarray:
        """Base colors with selection highlight. Returns [N, 3] float32."""
        colors = self.scene.sh0_colors().numpy().copy()
        colors[self._selected] = HIGHLIGHT_RGB
        return colors

    def _upload_splats(self) -> None:
        """(Re-)upload the full Gaussian scene and re-register the click handler."""
        covs = np.clip(self.scene.covariances(), -1e3, 1e3)  # prevent float16 overflow
        centers = self.scene.means.numpy()
        rgbs = self._display_colors()
        opacities = self.scene.real_opacities().numpy()

        handle = self.server.scene.add_gaussian_splats(
            name="/scene",
            centers=centers,
            covariances=covs,
            rgbs=rgbs,
            opacities=opacities[:, None],  # viser expects [N, 1]
        )

        # Click handler lives on the node handle, not on client.scene
        @handle.on_click
        def on_click(event) -> None:
            origin = np.array(event.ray_origin, dtype=np.float32)
            direction = np.array(event.ray_direction, dtype=np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            hit = self._ray_nearest_gaussian(origin, direction)
            with self._lock:
                self._select_at(hit, expand=False)

    # ──────────────────────────────────────── GUI

    def _setup_gui(self) -> None:
        with self.server.gui.add_folder("Selection"):
            self._radius_slider = self.server.gui.add_slider(
                "Radius",
                min=0.005,
                max=2.0,
                step=0.005,
                initial_value=0.15,
            )
            self._status_label = self.server.gui.add_markdown("*No Gaussians selected*")
            clear_btn = self.server.gui.add_button("Clear Selection")

        with self.server.gui.add_folder("Edit"):
            self._color_picker = self.server.gui.add_rgb(
                "Recolor", initial_value=(0.9, 0.3, 0.1)
            )
            recolor_btn = self.server.gui.add_button("Recolor Selected")
            delete_btn = self.server.gui.add_button(
                "Delete Selected", color="red"
            )

        with self.server.gui.add_folder("Transform"):
            self._tx = self.server.gui.add_number("Δx", initial_value=0.0, step=0.01)
            self._ty = self.server.gui.add_number("Δy", initial_value=0.0, step=0.01)
            self._tz = self.server.gui.add_number("Δz", initial_value=0.0, step=0.01)
            translate_btn = self.server.gui.add_button("Translate Selected")

        with self.server.gui.add_folder("File"):
            self._save_label = self.server.gui.add_markdown(f"*{self.path.name}*")
            save_btn = self.server.gui.add_button("Save Scene")

        # ── callbacks

        @clear_btn.on_click
        def _(_) -> None:
            with self._lock:
                self._clear_selection()

        @recolor_btn.on_click
        def _(_) -> None:
            with self._lock:
                self._recolor_selected()

        @delete_btn.on_click
        def _(_) -> None:
            with self._lock:
                self._delete_selected()

        @translate_btn.on_click
        def _(_) -> None:
            with self._lock:
                self._translate_selected()

        @save_btn.on_click
        def _(_) -> None:
            with self._lock:
                self._save()

    # ──────────────────────────────────────── selection logic

    def _ray_nearest_gaussian(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Return the position of the Gaussian whose centre is closest to the ray."""
        means = self.scene.means.numpy()           # [N, 3]
        to_means = means - origin                  # [N, 3]
        t = np.dot(to_means, direction).clip(0)    # [N]
        closest = origin + t[:, None] * direction  # [N, 3]
        return means[np.argmin(np.linalg.norm(means - closest, axis=1))]

    def _select_at(self, point: np.ndarray, expand: bool = False) -> None:
        idxs = self.kdtree.query_ball_point(point, r=float(self._radius_slider.value))
        if not expand:
            self._selected[:] = False
        self._selected[idxs] = True
        self._refresh_status()
        self._upload_splats()

    def _clear_selection(self) -> None:
        self._selected[:] = False
        self._refresh_status()
        self._upload_splats()

    def _refresh_status(self) -> None:
        n = int(self._selected.sum())
        if n == 0:
            self._status_label.content = "*No Gaussians selected*"
        else:
            self._status_label.content = f"**{n:,}** Gaussians selected"

    # ──────────────────────────────────────── edit operations

    def _recolor_selected(self) -> None:
        if not self._selected.any():
            console.print("[yellow]Nothing selected[/]")
            return
        rgb = tuple(self._color_picker.value)
        mask = torch.from_numpy(self._selected)
        self.scene.recolor_mask(mask, rgb)
        self._upload_splats()
        console.print(f"[green]Recolored {self._selected.sum():,} Gaussians → RGB {rgb}[/]")

    def _delete_selected(self) -> None:
        if not self._selected.any():
            console.print("[yellow]Nothing selected[/]")
            return
        n = int(self._selected.sum())
        mask = torch.from_numpy(self._selected)
        self.scene = self.scene.delete_mask(mask)
        self._selected = np.zeros(len(self.scene), dtype=bool)
        self._invalidate_kdtree()
        self._refresh_status()
        self._upload_splats()
        console.print(f"[red]Deleted {n:,} Gaussians — {len(self.scene):,} remaining[/]")

    def _translate_selected(self) -> None:
        if not self._selected.any():
            console.print("[yellow]Nothing selected[/]")
            return
        delta = torch.tensor(
            [self._tx.value, self._ty.value, self._tz.value], dtype=torch.float32
        )
        mask = torch.from_numpy(self._selected)
        self.scene.translate_mask(mask, delta)
        self._invalidate_kdtree()
        self._upload_splats()
        console.print(f"[green]Translated {self._selected.sum():,} Gaussians by {delta.tolist()}[/]")

    def _save(self) -> None:
        self.scene.save(self.path)
        console.print(f"[bold green]Scene saved → {self.path}[/]")
        self._save_label.content = f"*Saved: {self.path.name}*"

    # ──────────────────────────────────────── public

    def run(self) -> None:
        """Block the main thread, keeping the server alive."""
        console.print("[dim]Press Ctrl-C to exit.[/]")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            console.print("\n[bold]Editor closed.[/]")

    def save_to(self, path: str | Path) -> None:
        self.scene.save(path)
