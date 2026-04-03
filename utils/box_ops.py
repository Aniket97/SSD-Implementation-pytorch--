from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple


def center_to_corner(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        boxes[:, :2] - boxes[:, 2:] / 2.0,
        boxes[:, :2] + boxes[:, 2:] / 2.0,
    ], dim=1)


def corner_to_center(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        (boxes[:, :2] + boxes[:, 2:]) / 2.0,
        boxes[:, 2:] - boxes[:, :2],
    ], dim=1)


def pairwise_iou(set_a: torch.Tensor, set_b: torch.Tensor) -> torch.Tensor:
    A = set_a.size(0)
    B = set_b.size(0)

    tl = torch.max(
        set_a[:, :2].unsqueeze(1).expand(A, B, 2),
        set_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )
    br = torch.min(
        set_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        set_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )

    wh = (br - tl).clamp(min=0.0)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    area_a = ((set_a[:, 2] - set_a[:, 0]) * (set_a[:, 3] - set_a[:, 1]))
    area_b = ((set_b[:, 2] - set_b[:, 0]) * (set_b[:, 3] - set_b[:, 1]))
    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - intersection

    return intersection / union


def encode_gt_to_offsets(
    gt_corners: torch.Tensor,
    anchors_xywh: torch.Tensor,
    variances: Tuple[float, float],
) -> torch.Tensor:
    var_c, var_s = variances

    gt_cx_cy = (gt_corners[:, :2] + gt_corners[:, 2:]) / 2.0
    gt_wh = gt_corners[:, 2:] - gt_corners[:, :2]

    delta_cxcy = (gt_cx_cy - anchors_xywh[:, :2]) / (var_c * anchors_xywh[:, 2:])
    delta_wh = torch.log(gt_wh / anchors_xywh[:, 2:]) / var_s

    return torch.cat([delta_cxcy, delta_wh], dim=1)


def decode_offsets_to_boxes(
    predicted: torch.Tensor,
    anchors_xywh: torch.Tensor,
    variances: Tuple[float, float],
) -> torch.Tensor:
    var_c, var_s = variances

    pred_cx_cy = anchors_xywh[:, :2] + predicted[:, :2] * var_c * anchors_xywh[:, 2:]
    pred_wh = anchors_xywh[:, 2:] * torch.exp(predicted[:, 2:] * var_s)

    return torch.cat([
        pred_cx_cy - pred_wh / 2.0,
        pred_cx_cy + pred_wh / 2.0,
    ], dim=1)


def assign_anchors(
    gt_boxes: torch.Tensor,
    anchors_xywh: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
    variances: Tuple[float, float],
    out_loc: torch.Tensor,
    out_cls: torch.Tensor,
    batch_idx: int,
) -> None:
    anchors_corner = center_to_corner(anchors_xywh)
    iou = pairwise_iou(gt_boxes, anchors_corner)

    _, best_anchor_per_gt = iou.max(dim=1)
    best_iou_per_anchor, best_gt_per_anchor = iou.max(dim=0)

    for gt_idx in range(gt_boxes.size(0)):
        best_gt_per_anchor[best_anchor_per_gt[gt_idx]] = gt_idx
    best_iou_per_anchor[best_anchor_per_gt] = 2.0

    matched_gt_boxes = gt_boxes[best_gt_per_anchor]
    matched_labels = gt_labels[best_gt_per_anchor].long() + 1
    matched_labels[best_iou_per_anchor < iou_threshold] = 0

    out_loc[batch_idx] = encode_gt_to_offsets(matched_gt_boxes, anchors_xywh, variances)
    out_cls[batch_idx] = matched_labels


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float,
    top_k: int,
) -> Tuple[torch.Tensor, int]:
    keep = scores.new_zeros(scores.size(0), dtype=torch.long)

    if boxes.numel() == 0:
        return keep, 0

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()
    order = order[-top_k:]

    count = 0
    while order.numel() > 0:
        current = order[-1]
        keep[count] = current
        count += 1

        if order.numel() == 1:
            break

        order = order[:-1]

        ix1 = x1[order].clamp(min=x1[current].item())
        iy1 = y1[order].clamp(min=y1[current].item())
        ix2 = x2[order].clamp(max=x2[current].item())
        iy2 = y2[order].clamp(max=y2[current].item())

        iw = (ix2 - ix1).clamp(min=0.0)
        ih = (iy2 - iy1).clamp(min=0.0)
        inter = iw * ih

        union = areas[order] + areas[current] - inter
        iou_vals = inter / union

        order = order[iou_vals <= iou_thresh]

    return keep, count
