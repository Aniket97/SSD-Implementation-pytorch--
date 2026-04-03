from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ChannelL2Norm(nn.Module):
    def __init__(self, num_channels: int, initial_scale: float = 20.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((num_channels,), initial_scale))
        self._eps = 1e-10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l2_norm = x.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=self._eps)
        return self.gamma.view(1, -1, 1, 1) * (x / l2_norm)


_VGG16_CHANNELS = [
    64, 64, 'M',
    128, 128, 'M',
    256, 256, 256, 'C',
    512, 512, 512, 'M',
    512, 512, 512,
]


def _make_vgg_layers(spec: List, in_channels: int = 3) -> List[nn.Module]:
    layers: List[nn.Module] = []
    ch = in_channels

    for entry in spec:
        if entry == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif entry == 'C':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            layers.append(nn.Conv2d(ch, entry, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch = entry

    return layers


def build_vgg_backbone() -> List[nn.Module]:
    layers = _make_vgg_layers(_VGG16_CHANNELS, in_channels=3)

    # replace stride-2 pool5 with size-preserving version
    layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

    # FC6 → atrous conv, FC7 → 1x1 conv
    layers.append(nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(1024, 1024, kernel_size=1))
    layers.append(nn.ReLU(inplace=True))

    return layers


def build_extra_layers() -> List[nn.Module]:
    return [
        nn.Conv2d(1024, 256, kernel_size=1),
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(512, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(256, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3),
        nn.Conv2d(256, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3),
    ]
