from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


Boxes = np.ndarray


def _clip_to_image(boxes: Boxes, img_h: int, img_w: int) -> Boxes:
    out = boxes.copy()
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, img_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, img_h)
    return out


class _Brightness:
    def __init__(self, max_delta: float = 32.0) -> None:
        self.max_delta = max_delta

    def __call__(self, img, boxes, labels):
        if random.random() < 0.5:
            delta = random.uniform(-self.max_delta, self.max_delta)
            img = img.astype(np.float32) + delta
        return img, boxes, labels


class _Contrast:
    def __init__(self, low: float = 0.5, high: float = 1.5) -> None:
        self.low, self.high = low, high

    def __call__(self, img, boxes, labels):
        if random.random() < 0.5:
            img = img.astype(np.float32) * random.uniform(self.low, self.high)
        return img, boxes, labels


class _Saturation:
    def __init__(self, low: float = 0.5, high: float = 1.5) -> None:
        self.low, self.high = low, high

    def __call__(self, img, boxes, labels):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(self.low, self.high)
            img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        return img, boxes, labels


class _Hue:
    def __init__(self, max_delta: float = 18.0) -> None:
        self.max_delta = max_delta

    def __call__(self, img, boxes, labels):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-self.max_delta, self.max_delta)) % 360.0
            img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        return img, boxes, labels


class _ChannelShuffle:
    def __call__(self, img, boxes, labels):
        if random.random() < 0.5:
            perm = np.random.permutation(3)
            img = img[:, :, perm]
        return img, boxes, labels


class PhotometricDistort:
    def __init__(self) -> None:
        self._brightness  = _Brightness(32.0)
        self._contrast    = _Contrast(0.5, 1.5)
        self._saturation  = _Saturation(0.5, 1.5)
        self._hue         = _Hue(18.0)
        self._ch_shuffle  = _ChannelShuffle()

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        img, boxes, labels = self._brightness(img, boxes, labels)

        if random.random() < 0.5:
            img, boxes, labels = self._contrast(img, boxes, labels)
            img, boxes, labels = self._saturation(img, boxes, labels)
            img, boxes, labels = self._hue(img, boxes, labels)
        else:
            img, boxes, labels = self._saturation(img, boxes, labels)
            img, boxes, labels = self._hue(img, boxes, labels)
            img, boxes, labels = self._contrast(img, boxes, labels)

        img, boxes, labels = self._ch_shuffle(img, boxes, labels)
        return img, boxes, labels


class RandomExpand:
    def __init__(
        self,
        pixel_mean: Tuple[float, float, float],
        max_ratio: float = 4.0,
    ) -> None:
        self.pixel_mean = pixel_mean
        self.max_ratio  = max_ratio

    def __call__(self, img, boxes, labels):
        if random.random() < 0.5:
            return img, boxes, labels

        h, w = img.shape[:2]
        ratio  = random.uniform(1.0, self.max_ratio)
        new_w  = int(w * ratio)
        new_h  = int(h * ratio)
        off_x  = random.randint(0, new_w - w)
        off_y  = random.randint(0, new_h - h)

        canvas = np.full((new_h, new_w, 3), self.pixel_mean, dtype=np.float32)
        canvas[off_y:off_y + h, off_x:off_x + w] = img

        if boxes is not None and len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] += off_x
            boxes[:, [1, 3]] += off_y

        return canvas, boxes, labels


def _box_iou_with_patch(boxes: Boxes, patch: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(boxes[:, 0], patch[0])
    iy1 = np.maximum(boxes[:, 1], patch[1])
    ix2 = np.minimum(boxes[:, 2], patch[2])
    iy2 = np.minimum(boxes[:, 3], patch[3])

    inter_w = np.maximum(0.0, ix2 - ix1)
    inter_h = np.maximum(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_patch = (patch[2] - patch[0]) * (patch[3] - patch[1])

    return inter / (area_boxes + area_patch - inter + 1e-8)


class RandomSampleCrop:
    _MIN_IOU_CHOICES = (None, 0.1, 0.3, 0.7, 0.9, None)
    _MAX_ATTEMPTS    = 50

    def __call__(self, img, boxes, labels):
        h, w = img.shape[:2]

        while True:
            min_iou = random.choice(self._MIN_IOU_CHOICES)

            if min_iou is None and random.random() < 1.0 / len(self._MIN_IOU_CHOICES):
                return img, boxes, labels

            for _ in range(self._MAX_ATTEMPTS):
                crop_w = random.uniform(0.3 * w, w)
                crop_h = random.uniform(0.3 * h, h)

                if not (0.5 <= crop_h / crop_w <= 2.0):
                    continue

                x1 = random.uniform(0, w - crop_w)
                y1 = random.uniform(0, h - crop_h)
                patch = np.array([x1, y1, x1 + crop_w, y1 + crop_h])

                if boxes is None or len(boxes) == 0:
                    crop = img[int(y1):int(y1 + crop_h), int(x1):int(x1 + crop_w)]
                    return crop, boxes, labels

                iou_vals = _box_iou_with_patch(boxes, patch)
                if min_iou is not None and iou_vals.min() < min_iou:
                    continue

                cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
                cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
                inside = (cx > x1) & (cx < patch[2]) & (cy > y1) & (cy < patch[3])

                if not inside.any():
                    continue

                crop        = img[int(y1):int(y1 + crop_h), int(x1):int(x1 + crop_w)]
                kept_boxes  = boxes[inside].copy()
                kept_boxes[:, [0, 2]] -= x1
                kept_boxes[:, [1, 3]] -= y1
                kept_boxes  = _clip_to_image(kept_boxes, crop.shape[0], crop.shape[1])
                kept_labels = labels[inside]

                return crop, kept_boxes, kept_labels

            return img, boxes, labels


class RandomHFlip:
    def __call__(self, img, boxes, labels):
        if random.random() < 0.5:
            w = img.shape[1]
            img = img[:, ::-1]
            if boxes is not None and len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return img, boxes, labels


class SSDTrainTransform:
    def __init__(
        self,
        target_size: int = 300,
        pixel_mean: Tuple[float, float, float] = (104.0, 117.0, 123.0),
    ) -> None:
        self.target_size = target_size
        self.pixel_mean  = pixel_mean

        self._distort = PhotometricDistort()
        self._expand  = RandomExpand(pixel_mean)
        self._crop    = RandomSampleCrop()
        self._flip    = RandomHFlip()

    def __call__(
        self,
        img:    np.ndarray,
        boxes:  np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        img    = img.astype(np.float32)
        boxes  = boxes.astype(np.float32).copy()
        labels = labels.astype(np.int64).copy()

        h, w = img.shape[:2]
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

        img, boxes, labels = self._distort(img, boxes, labels)
        img, boxes, labels = self._expand(img, boxes, labels)
        img, boxes, labels = self._crop(img, boxes, labels)
        img, boxes, labels = self._flip(img, boxes, labels)

        new_h, new_w = img.shape[:2]
        img = cv2.resize(img, (self.target_size, self.target_size))

        boxes[:, [0, 2]] /= new_w
        boxes[:, [1, 3]] /= new_h
        boxes = boxes.clip(0.0, 1.0)

        img -= np.array(self.pixel_mean, dtype=np.float32)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        targets = np.hstack([boxes, labels.reshape(-1, 1)]).astype(np.float32)
        return img_tensor, targets


class SSDValTransform:
    def __init__(
        self,
        target_size: int = 300,
        pixel_mean: Tuple[float, float, float] = (104.0, 117.0, 123.0),
    ) -> None:
        self.target_size = target_size
        self.pixel_mean  = pixel_mean

    def __call__(
        self,
        img:    np.ndarray,
        boxes:  np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        img    = cv2.resize(img.astype(np.float32), (self.target_size, self.target_size))
        img   -= np.array(self.pixel_mean, dtype=np.float32)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        boxes  = boxes.astype(np.float32).copy().clip(0.0, 1.0)
        labels = labels.astype(np.int64).copy()
        targets = np.hstack([boxes, labels.reshape(-1, 1)]).astype(np.float32)
        return img_tensor, targets
