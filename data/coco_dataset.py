from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# non-sequential COCO category IDs → contiguous [1, 80]
COCO_ID_TO_CONTIGUOUS: dict[int, int] = {
    1: 1,  2: 2,  3: 3,  4: 4,  5: 5,  6: 6,  7: 7,  8: 8,  9: 9,  10: 10,
    11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19,
    21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28,
    33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37,
    42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46,
    52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55,
    61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
    74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73,
    84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80,
}


class COCODetection(Dataset):
    def __init__(
        self,
        root: str,
        image_set: str,
        image_ids: List[int],
        transform: Optional[Callable] = None,
    ) -> None:
        from pycocotools.coco import COCO

        self.root       = root
        self.image_set  = image_set
        self.image_ids  = image_ids
        self.transform  = transform
        self.name       = 'COCO'

        ann_file   = os.path.join(root, 'annotations', f'instances_{image_set}.json')
        self._coco = COCO(ann_file)
        self._img_dir = os.path.join(root, 'images', image_set)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id   = self.image_ids[idx]
        img_info = self._coco.loadImgs(img_id)[0]
        img_path = os.path.join(self._img_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        ann_ids     = self._coco.getAnnIds(imgIds=img_id)
        annotations = self._coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in annotations:
            if ann.get('ignore', 0) or ann.get('iscrowd', 0):
                continue
            cat_id = ann['category_id']
            if cat_id not in COCO_ID_TO_CONTIGUOUS:
                continue

            bx, by, bw, bh = ann['bbox']
            x1 = max(0.0, bx) / w
            y1 = max(0.0, by) / h
            x2 = min(w, bx + bw) / w
            y2 = min(h, by + bh) / h

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(COCO_ID_TO_CONTIGUOUS[cat_id] - 1)

        if boxes:
            boxes_arr  = np.array(boxes,  dtype=np.float32)
            labels_arr = np.array(labels, dtype=np.int64)
        else:
            boxes_arr  = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros(0,      dtype=np.int64)

        if self.transform is not None:
            img_t, targets_arr = self.transform(img, boxes_arr, labels_arr)
        else:
            img_t       = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1))
            targets_arr = np.hstack([boxes_arr, labels_arr.reshape(-1, 1)]).astype(np.float32)

        return img_t, torch.FloatTensor(targets_arr)
