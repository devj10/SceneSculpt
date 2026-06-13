"""
3D Gaussian Splatting training loop built on gsplat.

Reference hyperparameters follow the original 3DGS paper (Kerbl et al. 2023).
Requires a CUDA GPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rich.console import Console
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from gsplat import rasterization

from scenesculpt.pipeline.run_colmap import Camera
from scenesculpt.scene.gaussian import GaussianScene

console = Console()


# ─────────────────────────────────────────────────── hyperparameters


@dataclass
class TrainConfig:
    num_iterations: int = 30_000
    # Learning rates
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_sh_dc: float = 2.5e-3
    lr_sh_rest: float = 2.5e-3 / 20
    lr_opacities: float = 5e-2
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    # Loss weights
    lambda_ssim: float = 0.2
    # Densification
    densify_from: int = 500
    densify_until: int = 15_000
    densify_every: int = 100
    grad_threshold: float = 2e-4
    split_scale_threshold: float = 0.01
    prune_opacity_threshold: float = 5e-3
    prune_scale_factor: float = 0.1
    # Opacity reset
    reset_opacity_every: int = 3_000
    # Logging / checkpoints
    log_every: int = 500
    save_every: int = 5_000
    sh_degree_up_every: int = 1_000   # increase SH degree every N steps (max 3)
    sh_degree_max: int = 3


# ─────────────────────────────────────────────────── model


class GaussianModel(nn.Module):
    """Trainable wrapper around a Gaussian scene."""

    def __init__(self, means: torch.Tensor, rgb_init: torch.Tensor | None = None) -> None:
        super().__init__()
        N = len(means)

        self.means = nn.Parameter(means.float().cuda())

        # Unit quaternions (wxyz), identity rotation = [1, 0, 0, 0]
        quats = torch.zeros(N, 4)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats.cuda())

        # Log-scales: start at a small uniform size
        scales = torch.full((N, 3), -4.0)
        self.scales = nn.Parameter(scales.cuda())

        # Logit-opacities: sigmoid(0.1) ≈ 0.52
        self.opacities = nn.Parameter(torch.full((N,), 0.1).cuda())

        # Spherical harmonics: DC + rest for degree 3
        K = (3 + 1) ** 2 - 1  # 15
        if rgb_init is not None:
            from scenesculpt.scene.gaussian import SH_C0
            sh_dc = ((rgb_init.float().cuda() - 0.5) / SH_C0).unsqueeze(1)  # [N, 1, 3]
        else:
            sh_dc = torch.zeros(N, 1, 3).cuda()
        self.sh_dc = nn.Parameter(sh_dc)
        self.sh_rest = nn.Parameter(torch.zeros(N, K, 3).cuda())

        # Gradient accumulator for densification (not a parameter)
        self.register_buffer("grad_accum", torch.zeros(N).cuda())
        self.register_buffer("grad_count", torch.zeros(N, dtype=torch.long).cuda())
        self.active_sh_degree = 0

    def __len__(self) -> int:
        return self.means.shape[0]

    def get_colors(self) -> torch.Tensor:
        """Return full SH color tensor [N, (deg+1)^2, 3] up to active degree."""
        return torch.cat([self.sh_dc, self.sh_rest], dim=1)

    def to_scene(self) -> GaussianScene:
        """Export to a CPU GaussianScene for saving/editing."""
        with torch.no_grad():
            return GaussianScene(
                means=self.means.cpu(),
                quats=F.normalize(self.quats, dim=-1).cpu(),
                scales=self.scales.cpu(),
                opacities=self.opacities.cpu(),
                sh_dc=self.sh_dc.squeeze(1).cpu(),
                sh_rest=self.sh_rest.cpu(),
                object_ids=torch.full((len(self),), -1, dtype=torch.int32),
            )

    # ──────── densification helpers

    def accumulate_grads(self, means2d_grad: torch.Tensor) -> None:
        """Accumulate 2D gradient norms for densification decisions."""
        # grad shape is [C, N, 2] when rendered with a batch dim — flatten to [N, 2]
        if means2d_grad.dim() == 3:
            means2d_grad = means2d_grad.squeeze(0)
        norms = means2d_grad.norm(dim=-1)  # [N]
        self.grad_accum += norms.detach()
        self.grad_count += 1

    def densify_and_prune(self, scene_scale: float) -> int:
        """Clone, split, and prune Gaussians. Returns change in count."""
        cfg = self._cfg
        grads = self.grad_accum / self.grad_count.clamp(min=1).float()
        scales_exp = torch.exp(self.scales)  # [N, 3]
        max_scale = scales_exp.max(dim=-1).values  # [N]

        over_thresh = grads > cfg.grad_threshold
        small = max_scale < cfg.split_scale_threshold
        large = ~small

        clone_mask = over_thresh & small
        split_mask = over_thresh & large
        prune_mask = (
            (torch.sigmoid(self.opacities) < cfg.prune_opacity_threshold)
            | (max_scale > cfg.prune_scale_factor * scene_scale)
        )

        delta = 0
        if clone_mask.any():
            delta += self._clone(clone_mask)
        if split_mask.any():
            delta += self._split(split_mask)
            prune_mask = torch.cat([prune_mask, torch.ones(delta, dtype=torch.bool, device=prune_mask.device)])
        if prune_mask.any():
            delta -= self._prune(prune_mask)

        # Reset accumulators after densification
        N = len(self)
        self.grad_accum = torch.zeros(N, device=self.means.device)
        self.grad_count = torch.zeros(N, dtype=torch.long, device=self.means.device)
        return delta

    def reset_opacities(self) -> None:
        with torch.no_grad():
            # Reset to sigmoid^-1(0.01) ≈ -4.6
            self.opacities.data.clamp_max_(-4.6)

    def _clone(self, mask: torch.Tensor) -> int:
        n = mask.sum().item()
        self.means = nn.Parameter(torch.cat([self.means, self.means[mask]]))
        self.quats = nn.Parameter(torch.cat([self.quats, self.quats[mask]]))
        self.scales = nn.Parameter(torch.cat([self.scales, self.scales[mask]]))
        self.opacities = nn.Parameter(torch.cat([self.opacities, self.opacities[mask]]))
        self.sh_dc = nn.Parameter(torch.cat([self.sh_dc, self.sh_dc[mask]]))
        self.sh_rest = nn.Parameter(torch.cat([self.sh_rest, self.sh_rest[mask]]))
        return n

    def _split(self, mask: torch.Tensor) -> int:
        """Replace each selected Gaussian with 2 smaller daughters."""
        n = mask.sum().item()
        scales_exp = torch.exp(self.scales[mask])  # [n, 3]

        # Sample 2 offsets along the principal axis of each Gaussian
        noise = torch.randn(2, n, 3, device=self.means.device) * scales_exp.unsqueeze(0)
        # [2, n, 3] → use as position deltas

        # Shrink scale by factor ~1.6 (as in original paper)
        new_scales = (self.scales[mask] - np.log(1.6)).unsqueeze(0).expand(2, -1, -1)

        new_means = self.means[mask].unsqueeze(0) + noise  # [2, n, 3]

        for attr, new_val in [
            ("means", new_means.reshape(-1, 3)),
            ("quats", self.quats[mask].unsqueeze(0).expand(2, -1, -1).reshape(-1, 4)),
            ("scales", new_scales.reshape(-1, 3)),
            ("opacities", self.opacities[mask].unsqueeze(0).expand(2, -1).reshape(-1)),
            ("sh_dc", self.sh_dc[mask].unsqueeze(0).expand(2, -1, -1, -1).reshape(-1, 1, 3)),
            ("sh_rest", self.sh_rest[mask].unsqueeze(0).expand(2, -1, -1, -1).reshape(-1, *self.sh_rest.shape[1:])),
        ]:
            old = getattr(self, attr)
            setattr(self, attr, nn.Parameter(torch.cat([old[~mask], new_val])))

        return n  # net gain: 2*n new - n removed = n (handled together with prune)

    def _prune(self, mask: torch.Tensor) -> int:
        keep = ~mask
        n = mask.sum().item()
        for attr in ("means", "quats", "scales", "opacities", "sh_dc", "sh_rest"):
            old = getattr(self, attr)
            setattr(self, attr, nn.Parameter(old[keep]))
        return n

    # set by GaussianTrainer before first densification call
    _cfg: TrainConfig = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────── training


class GaussianTrainer:
    def __init__(
        self,
        cameras: list[Camera],
        output_dir: str | Path,
        point_cloud: np.ndarray | None = None,
        config: TrainConfig | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Gaussian Splatting training requires a CUDA GPU.")

        self.cameras = cameras
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config or TrainConfig()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

        # Initialize Gaussians
        if point_cloud is None or len(point_cloud) < 10:
            console.print("[yellow]No point cloud; initialising from random points.[/]")
            point_cloud = np.random.randn(10_000, 3).astype(np.float32) * 0.5

        means_init = torch.from_numpy(point_cloud).float()
        self.model = GaussianModel(means_init).cuda()
        self.model._cfg = self.cfg

        # Compute scene scale for pruning (max distance from centroid)
        c = means_init.mean(0)
        self.scene_scale = float((means_init - c).norm(dim=-1).max().item())

        self.optimizer = self._build_optimizer()

    # ──────── optimizer

    def _build_optimizer(self) -> torch.optim.Adam:
        cfg = self.cfg
        params = [
            {"params": [self.model.means],      "lr": cfg.lr_means,      "name": "means"},
            {"params": [self.model.quats],       "lr": cfg.lr_quats,      "name": "quats"},
            {"params": [self.model.scales],      "lr": cfg.lr_scales,     "name": "scales"},
            {"params": [self.model.opacities],   "lr": cfg.lr_opacities,  "name": "opacities"},
            {"params": [self.model.sh_dc],       "lr": cfg.lr_sh_dc,      "name": "sh_dc"},
            {"params": [self.model.sh_rest],     "lr": cfg.lr_sh_rest,    "name": "sh_rest"},
        ]
        return torch.optim.Adam(params, eps=1e-15)

    def _update_means_lr(self, step: int) -> None:
        """Exponential decay for position learning rate."""
        cfg = self.cfg
        t = np.clip(step / cfg.num_iterations, 0.0, 1.0)
        lr = np.exp(
            np.log(cfg.lr_means) * (1.0 - t) + np.log(cfg.lr_means_final) * t
        )
        for group in self.optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = lr

    def _rebuild_optimizer_for_new_params(self) -> None:
        """Rebuild optimizer after densification (parameter count changed)."""
        # Copy state for surviving parameters by index — easiest is to just reset
        self.optimizer = self._build_optimizer()

    # ──────── training step

    def _render(self, cam: Camera) -> tuple[torch.Tensor, dict]:
        """Render one camera view. Returns [H, W, 3] and info dict."""
        model = self.model
        quats_norm = F.normalize(model.quats, dim=-1)

        colors = model.get_colors()  # [N, (deg+1)^2, 3]

        viewmat = torch.from_numpy(cam.w2c).float().cuda().unsqueeze(0)  # [1, 4, 4]
        K = torch.from_numpy(cam.K).float().cuda().unsqueeze(0)          # [1, 3, 3]

        renders, alphas, info = rasterization(
            means=model.means,
            quats=quats_norm,
            scales=torch.exp(model.scales),
            opacities=torch.sigmoid(model.opacities),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=cam.width,
            height=cam.height,
            sh_degree=model.active_sh_degree,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
        )
        return renders[0], info  # [H, W, 3]

    def _load_image(self, cam: Camera) -> torch.Tensor:
        """Load ground-truth image as [H, W, 3] float32 CUDA tensor."""
        img = Image.open(cam.image_path).convert("RGB")
        img = img.resize((cam.width, cam.height), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).cuda()

    def _loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """L1 + SSIM loss. pred/gt: [H, W, 3]."""
        p = pred.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        g = gt.permute(2, 0, 1).unsqueeze(0)
        l1 = F.l1_loss(p, g)
        ssim_val = self.ssim(p, g)
        return (1.0 - self.cfg.lambda_ssim) * l1 + self.cfg.lambda_ssim * (1.0 - ssim_val)

    # ──────── main loop

    def train(self) -> GaussianScene:
        cfg = self.cfg
        console.print(f"[bold]Training for {cfg.num_iterations:,} iterations — {len(self.model):,} initial Gaussians[/]")

        pbar = tqdm(range(cfg.num_iterations), desc="Training")
        metrics: dict[str, float] = {}

        for step in pbar:
            self._update_means_lr(step)

            # Anneal SH degree
            target_sh = min(step // cfg.sh_degree_up_every, cfg.sh_degree_max)
            self.model.active_sh_degree = target_sh

            cam = random.choice(self.cameras)
            gt = self._load_image(cam)

            # Forward
            pred, info = self._render(cam)
            info["means2d"].retain_grad()

            loss = self._loss(pred, gt)
            loss.backward()

            # Accumulate 2D gradients for densification
            if info["means2d"].grad is not None:
                self.model.accumulate_grads(info["means2d"].grad)

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Densification
            if cfg.densify_from <= step <= cfg.densify_until and step % cfg.densify_every == 0:
                self.model.densify_and_prune(self.scene_scale)
                self._rebuild_optimizer_for_new_params()

            # Opacity reset
            if step > 0 and step % cfg.reset_opacity_every == 0:
                self.model.reset_opacities()

            # Logging
            if step % cfg.log_every == 0:
                metrics = {"loss": loss.item(), "N": len(self.model)}
                pbar.set_postfix(metrics)

            # Checkpoint
            if step > 0 and step % cfg.save_every == 0:
                ckpt_path = self.output_dir / f"splats_{step:06d}.ply"
                self.model.to_scene().save(ckpt_path)
                console.print(f"[dim]Saved checkpoint → {ckpt_path}[/]")

        # Final save
        final_path = self.output_dir / "splats.ply"
        scene = self.model.to_scene()
        scene.save(final_path)
        console.print(f"[green bold]Training complete. Scene saved → {final_path}[/]")
        return scene
