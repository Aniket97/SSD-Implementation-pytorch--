from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import assign_anchors


class SSDLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        neg_pos_ratio: int = 3,
        variances: Tuple[float, float] = (0.1, 0.2),
    ) -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variances     = variances

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loc_preds, cls_preds, anchors = predictions
        B, N = loc_preds.shape[:2]

        loc_targets = torch.zeros(B, N, 4,          device=loc_preds.device)
        cls_targets = torch.zeros(B, N, dtype=torch.long, device=loc_preds.device)

        for b in range(B):
            gt = targets[b]
            assign_anchors(
                gt_boxes      = gt[:, :4],
                anchors_xywh  = anchors,
                gt_labels     = gt[:, 4].long(),
                iou_threshold = self.iou_threshold,
                variances     = self.variances,
                out_loc       = loc_targets,
                out_cls       = cls_targets,
                batch_idx     = b,
            )

        pos_mask = cls_targets > 0
        num_pos  = pos_mask.sum()

        if num_pos.item() == 0:
            return loc_preds.sum() * 0.0, cls_preds.sum() * 0.0

        pos_3d   = pos_mask.unsqueeze(-1).expand_as(loc_preds)
        loc_loss = F.smooth_l1_loss(
            loc_preds[pos_3d].view(-1, 4),
            loc_targets[pos_3d].view(-1, 4),
            reduction='sum',
        )

        flat_logits = cls_preds.view(-1, self.num_classes)
        log_sum_exp = (
            torch.log(torch.exp(flat_logits - flat_logits.max(dim=1, keepdim=True).values).sum(dim=1, keepdim=True))
            + flat_logits.max(dim=1, keepdim=True).values
        )
        neg_scores = (log_sum_exp - flat_logits.gather(1, cls_targets.view(-1, 1)))
        neg_scores = neg_scores.view(B, N)

        neg_scores[pos_mask] = 0.0

        _,      sort_by_loss = neg_scores.sort(dim=1, descending=True)
        _, rank_in_image     = sort_by_loss.sort(dim=1)

        n_pos_per_image = pos_mask.long().sum(dim=1, keepdim=True)
        n_neg_per_image = (self.neg_pos_ratio * n_pos_per_image).clamp(max=N - 1)
        neg_mask = rank_in_image < n_neg_per_image

        selected = (pos_mask | neg_mask)
        cls_loss = F.cross_entropy(
            cls_preds[selected],
            cls_targets[selected],
            reduction='sum',
        )

        norm = num_pos.float()
        return loc_loss / norm, cls_loss / norm
