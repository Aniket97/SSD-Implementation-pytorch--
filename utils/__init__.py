# utils package – bounding-box operations
from .box_ops import (
    pairwise_iou,
    center_to_corner,
    corner_to_center,
    encode_gt_to_offsets,
    decode_offsets_to_boxes,
    assign_anchors,
    nms,
)
