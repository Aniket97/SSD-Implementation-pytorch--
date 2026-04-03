from __future__ import annotations

import math
import torch

from config import SSD300Config


class DefaultBoxGenerator:
    def __init__(self, cfg: SSD300Config) -> None:
        self.cfg = cfg
        self._cached: torch.Tensor | None = None

    def generate(self) -> torch.Tensor:
        if self._cached is not None:
            return self._cached

        cfg = self.cfg
        S = cfg.input_size
        rows: list[list[float]] = []

        for level, fm_size in enumerate(cfg.feature_map_dims):
            f_k = S / cfg.anchor_strides[level]
            s_k = cfg.anchor_min_scales[level] / S
            s_k_next = cfg.anchor_max_scales[level] / S

            for row in range(fm_size):
                for col in range(fm_size):
                    cx = (col + 0.5) / f_k
                    cy = (row + 0.5) / f_k

                    rows.append([cx, cy, s_k, s_k])

                    s_mid = math.sqrt(s_k * s_k_next)
                    rows.append([cx, cy, s_mid, s_mid])

                    for ar in cfg.anchor_aspect_ratios[level]:
                        sqrt_ar = math.sqrt(ar)
                        rows.append([cx, cy, s_k * sqrt_ar, s_k / sqrt_ar])
                        rows.append([cx, cy, s_k / sqrt_ar, s_k * sqrt_ar])

        boxes = torch.tensor(rows, dtype=torch.float32).clamp_(0.0, 1.0)
        self._cached = boxes
        return boxes
