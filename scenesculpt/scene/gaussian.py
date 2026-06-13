"""Core Gaussian scene data structure with PLY I/O and in-place edit operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import plyfile


SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


@dataclass
class GaussianScene:
    """
    A 3D Gaussian Splatting scene.

    All tensors live on CPU by default; move to GPU for training.
    Stored values are pre-activation:
      - opacities: logit  (sigmoid → real opacity)
      - scales:    log    (exp    → real scale)
      - quats:     raw wxyz quaternion (normalised on load)
    """

    means: torch.Tensor       # [N, 3]   world-space positions
    quats: torch.Tensor       # [N, 4]   rotation quaternions, wxyz
    scales: torch.Tensor      # [N, 3]   log-scales
    opacities: torch.Tensor   # [N]      logit-opacities
    sh_dc: torch.Tensor       # [N, 3]   DC spherical-harmonic coefficients
    sh_rest: torch.Tensor     # [N, K, 3] higher-order SH (K = (deg+1)^2 - 1)
    object_ids: torch.Tensor  # [N]      int32, -1 = unassigned

    # ------------------------------------------------------------------ I/O

    @classmethod
    def load(cls, path: str | Path) -> GaussianScene:
        """Load from the standard 3DGS .ply format (+ optional object_id field)."""
        ply = plyfile.PlyData.read(str(path))
        v = ply["vertex"]
        names = v.data.dtype.names

        means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

        sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)

        rest_keys = sorted([k for k in names if k.startswith("f_rest_")])
        if rest_keys:
            # stored as [N, 3*K]: K coeffs for ch0, then ch1, then ch2
            flat = np.stack([v[k] for k in rest_keys], axis=1).astype(np.float32)
            K = len(rest_keys) // 3
            # reshape to [N, 3, K] then transpose to [N, K, 3]
            sh_rest = flat.reshape(-1, 3, K).transpose(0, 2, 1)
        else:
            sh_rest = np.zeros((len(means), 0, 3), dtype=np.float32)

        opacities = v["opacity"].astype(np.float32)
        scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
        quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

        if "object_id" in names:
            object_ids = v["object_id"].astype(np.int32)
        else:
            object_ids = np.full(len(means), -1, dtype=np.int32)

        return cls(
            means=torch.from_numpy(means),
            quats=torch.from_numpy(quats),
            scales=torch.from_numpy(scales),
            opacities=torch.from_numpy(opacities),
            sh_dc=torch.from_numpy(sh_dc),
            sh_rest=torch.from_numpy(sh_rest),
            object_ids=torch.from_numpy(object_ids),
        )

    def save(self, path: str | Path) -> None:
        """Save to standard 3DGS .ply format with object_id extension."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        N = len(self)
        means = self.means.cpu().numpy()
        quats = self.quats.cpu().numpy()
        scales = self.scales.cpu().numpy()
        opacities = self.opacities.cpu().numpy()
        sh_dc = self.sh_dc.cpu().numpy()
        sh_rest = self.sh_rest.cpu().numpy()   # [N, K, 3]
        object_ids = self.object_ids.cpu().numpy()

        # Build attribute list matching the 3DGS convention
        attrs: list[tuple[str, np.ndarray]] = [
            ("x", means[:, 0]), ("y", means[:, 1]), ("z", means[:, 2]),
            ("nx", np.zeros(N, np.float32)),
            ("ny", np.zeros(N, np.float32)),
            ("nz", np.zeros(N, np.float32)),
            ("f_dc_0", sh_dc[:, 0]), ("f_dc_1", sh_dc[:, 1]), ("f_dc_2", sh_dc[:, 2]),
        ]

        # [N, K, 3] → [N, 3, K] → [N, 3*K], stored as f_rest_0..3K-1
        K = sh_rest.shape[1]
        flat_rest = sh_rest.transpose(0, 2, 1).reshape(N, -1)
        for i in range(flat_rest.shape[1]):
            attrs.append((f"f_rest_{i}", flat_rest[:, i]))

        attrs += [
            ("opacity", opacities),
            ("scale_0", scales[:, 0]), ("scale_1", scales[:, 1]), ("scale_2", scales[:, 2]),
            ("rot_0", quats[:, 0]), ("rot_1", quats[:, 1]),
            ("rot_2", quats[:, 2]), ("rot_3", quats[:, 3]),
            ("object_id", object_ids),
        ]

        dtype = [(name, "f4" if name != "object_id" else "i4") for name, _ in attrs]
        arr = np.empty(N, dtype=dtype)
        for name, data in attrs:
            arr[name] = data

        plyfile.PlyData([plyfile.PlyElement.describe(arr, "vertex")]).write(str(path))

    # ------------------------------------------------------------------ derived quantities

    def __len__(self) -> int:
        return self.means.shape[0]

    def real_opacities(self) -> torch.Tensor:
        """Sigmoid-activated opacities in [0, 1]."""
        return torch.sigmoid(self.opacities)

    def real_scales(self) -> torch.Tensor:
        """Exp-activated scales."""
        return torch.exp(self.scales)

    def sh0_colors(self) -> torch.Tensor:
        """View-independent RGB colors from the DC SH term, clamped to [0, 1]."""
        return (self.sh_dc * SH_C0 + 0.5).clamp(0.0, 1.0)

    def covariances(self) -> np.ndarray:
        """
        Return [N, 3, 3] covariance matrices from quaternions + scales.
        cov = R @ diag(s^2) @ R^T
        """
        q = self.quats.cpu().numpy()           # [N, 4] wxyz
        s = self.real_scales().cpu().numpy()   # [N, 3]

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.stack([
            1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x),
            2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x**2 + y**2),
        ], axis=-1).reshape(-1, 3, 3)           # [N, 3, 3]

        # R @ diag(s^2) @ R^T  =  (R * s^2) @ R^T
        cov = (R * (s**2)[:, None, :]) @ R.transpose(0, 2, 1)
        return cov.astype(np.float32)

    # ------------------------------------------------------------------ selection

    def mask_by_object_id(self, object_id: int) -> torch.Tensor:
        return (self.object_ids == object_id)

    def mask_by_radius(self, center: np.ndarray, radius: float) -> torch.Tensor:
        c = torch.from_numpy(center.astype(np.float32))
        return torch.norm(self.means - c, dim=-1) <= radius

    def unique_object_ids(self) -> list[int]:
        ids = self.object_ids[self.object_ids >= 0].unique()
        return [int(i) for i in ids.tolist()]

    # ------------------------------------------------------------------ edit ops (return new scene or mutate in-place)

    def delete_mask(self, mask: torch.Tensor) -> GaussianScene:
        """Return a new scene with selected Gaussians removed."""
        keep = ~mask
        return GaussianScene(
            means=self.means[keep].clone(),
            quats=self.quats[keep].clone(),
            scales=self.scales[keep].clone(),
            opacities=self.opacities[keep].clone(),
            sh_dc=self.sh_dc[keep].clone(),
            sh_rest=self.sh_rest[keep].clone(),
            object_ids=self.object_ids[keep].clone(),
        )

    def recolor_mask(self, mask: torch.Tensor, rgb: tuple[float, float, float]) -> None:
        """Recolor selected Gaussians to a solid color (in-place).
        Zeroes higher-order SH so the color is view-independent."""
        color = torch.tensor(rgb, dtype=torch.float32)
        self.sh_dc[mask] = (color - 0.5) / SH_C0
        if self.sh_rest.shape[1] > 0:
            self.sh_rest[mask] = 0.0

    def translate_mask(self, mask: torch.Tensor, delta: torch.Tensor) -> None:
        """Translate selected Gaussians by delta [3] (in-place)."""
        self.means[mask] += delta.to(self.means)

    def assign_object_id(self, mask: torch.Tensor, object_id: int) -> None:
        """Tag selected Gaussians with an object_id (in-place)."""
        self.object_ids[mask] = object_id
