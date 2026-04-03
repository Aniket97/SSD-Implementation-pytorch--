from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg_backbone import build_vgg_backbone, build_extra_layers, ChannelL2Norm
from prior_boxes import DefaultBoxGenerator
from config import SSD300Config


_ANCHORS_PER_CELL = [4, 6, 6, 6, 4, 4]
_FEATURE_CHANNELS = [512, 1024, 512, 256, 256, 256]
_CONV4_3_END = 23


def _build_heads(
    channels: List[int],
    n_anchors: List[int],
    num_classes: int,
) -> Tuple[nn.ModuleList, nn.ModuleList]:
    loc, cls = [], []
    for ch, k in zip(channels, n_anchors):
        loc.append(nn.Conv2d(ch, k * 4,          kernel_size=3, padding=1))
        cls.append(nn.Conv2d(ch, k * num_classes, kernel_size=3, padding=1))
    return nn.ModuleList(loc), nn.ModuleList(cls)


class SSD300(nn.Module):
    def __init__(self, cfg: SSD300Config, inference_mode: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.inference_mode = inference_mode

        self.backbone = nn.ModuleList(build_vgg_backbone())
        self.l2_norm  = ChannelL2Norm(512, initial_scale=20.0)
        self.extras   = nn.ModuleList(build_extra_layers())

        self.loc_heads, self.cls_heads = _build_heads(
            _FEATURE_CHANNELS, _ANCHORS_PER_CELL, cfg.num_classes
        )

        anchors = DefaultBoxGenerator(cfg).generate()
        self.register_buffer('anchors', anchors)

    def _extract_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        fmaps: List[torch.Tensor] = []

        for layer in self.backbone[:_CONV4_3_END]:
            x = layer(x)
        fmaps.append(self.l2_norm(x))

        for layer in self.backbone[_CONV4_3_END:]:
            x = layer(x)
        fmaps.append(x)

        for idx, layer in enumerate(self.extras):
            x = F.relu(layer(x), inplace=True)
            if idx % 2 == 1:
                fmaps.append(x)

        return fmaps

    def forward(self, images: torch.Tensor):
        fmaps = self._extract_feature_maps(images)
        B = images.size(0)

        loc_list, cls_list = [], []
        for fm, loc_head, cls_head in zip(fmaps, self.loc_heads, self.cls_heads):
            loc_list.append(loc_head(fm).permute(0, 2, 3, 1).contiguous())
            cls_list.append(cls_head(fm).permute(0, 2, 3, 1).contiguous())

        loc_preds = torch.cat([t.view(B, -1) for t in loc_list], dim=1)
        cls_preds = torch.cat([t.view(B, -1) for t in cls_list], dim=1)

        loc_preds = loc_preds.view(B, -1, 4)
        cls_preds = cls_preds.view(B, -1, self.cfg.num_classes)

        if self.inference_mode:
            return self._post_process(loc_preds, cls_preds)

        return loc_preds, cls_preds, self.anchors

    def _post_process(
        self,
        loc_preds: torch.Tensor,
        cls_preds: torch.Tensor,
        conf_threshold: float = 0.01,
        nms_threshold: float = 0.45,
        top_k: int = 200,
    ) -> torch.Tensor:
        from utils.box_ops import decode_offsets_to_boxes
        import torchvision

        B = loc_preds.size(0)
        num_cls = self.cfg.num_classes
        output = loc_preds.new_zeros(B, num_cls, top_k, 5)

        cls_probs = F.softmax(cls_preds, dim=-1)

        for b in range(B):
            decoded = decode_offsets_to_boxes(
                loc_preds[b], self.anchors, self.cfg.box_variances
            )
            for c in range(1, num_cls):
                scores = cls_probs[b, :, c]
                mask   = scores > conf_threshold
                if not mask.any():
                    continue
                kept_scores = scores[mask]
                kept_boxes  = decoded[mask]
                keep_idx = torchvision.ops.nms(kept_boxes, kept_scores, nms_threshold)
                keep_idx = keep_idx[:top_k]
                n_kept   = keep_idx.size(0)
                output[b, c, :n_kept, 0]  = kept_scores[keep_idx]
                output[b, c, :n_kept, 1:] = kept_boxes[keep_idx]

        return output

    @classmethod
    def from_vgg_weights(
        cls,
        cfg: SSD300Config,
        vgg_weights_path: str,
        inference_mode: bool = False,
    ) -> 'SSD300':
        model = cls(cfg, inference_mode)

        backbone_state = torch.load(vgg_weights_path, map_location='cpu')
        model.backbone.load_state_dict(backbone_state)
        print(f'Loaded VGG16 backbone weights from {vgg_weights_path}')

        for module_list in [model.extras, model.loc_heads, model.cls_heads]:
            for layer in module_list:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        return model
