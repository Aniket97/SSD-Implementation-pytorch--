from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SSD300Config:
    name: str
    num_classes: int
    input_size: int
    feature_map_dims: List[int]
    anchor_strides: List[int]
    anchor_min_scales: List[int]
    anchor_max_scales: List[int]
    anchor_aspect_ratios: List[List[int]]
    box_variances: Tuple[float, float]
    max_iterations: int
    lr_decay_at: Tuple[int, ...]


COCO_CONFIG = SSD300Config(
    name='COCO',
    num_classes=81,
    input_size=300,
    feature_map_dims=[38, 19, 10, 5, 3, 1],
    anchor_strides=[8, 16, 32, 64, 100, 300],
    anchor_min_scales=[21, 45, 99, 153, 207, 261],
    anchor_max_scales=[45, 99, 153, 207, 261, 315],
    anchor_aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    box_variances=(0.1, 0.2),
    max_iterations=4000,
    lr_decay_at=(60000, 75000),
)

VGG_PIXEL_MEAN: Tuple[float, float, float] = (104.0, 117.0, 123.0)
