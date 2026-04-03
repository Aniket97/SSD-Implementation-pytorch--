from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from tqdm import tqdm

from config import COCO_CONFIG, VGG_PIXEL_MEAN
from data.coco_dataset import COCO_ID_TO_CONTIGUOUS
from model.ssd_net import SSD300


KAGGLE_BASE_DIR = '/kaggle/working'

_DEFAULTS = dict(
    trained_model        = os.path.join(KAGGLE_BASE_DIR, 'weights/ssd300_COCO_final.pth'),
    test_ids_path        = os.path.join(KAGGLE_BASE_DIR, 'weights/coco_test_ids.npy'),
    dataset_root         = os.path.join(KAGGLE_BASE_DIR, 'data/coco'),
    coco_image_set       = 'val2014',
    save_folder          = os.path.join(KAGGLE_BASE_DIR, 'eval/'),
    confidence_threshold = 0.01,
    top_k                = 200,
    use_cuda             = True,
    batch_size           = 32,
    num_workers          = 4,
)

args = SimpleNamespace(**_DEFAULTS)

_CONTIGUOUS_TO_COCO_ID: dict[int, int] = {
    v: k for k, v in COCO_ID_TO_CONTIGUOUS.items()
}


class _COCOEvalDataset(data.Dataset):
    def __init__(self, root: str, image_set: str, image_ids: list) -> None:
        from pycocotools.coco import COCO
        ann_file      = os.path.join(root, 'annotations', f'instances_{image_set}.json')
        self._coco    = COCO(ann_file)
        self._img_dir = os.path.join(root, 'images', image_set)
        self._ids     = image_ids
        self._mean    = np.array(VGG_PIXEL_MEAN, dtype=np.float32)
        self._size    = COCO_CONFIG.input_size

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int):
        img_id   = self._ids[idx]
        img_info = self._coco.loadImgs(img_id)[0]
        img_path = os.path.join(self._img_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        orig_h, orig_w = img.shape[:2]

        img_f = cv2.resize(img.astype(np.float32), (self._size, self._size))
        img_f -= self._mean
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1))

        return img_t, img_id, orig_h, orig_w


def _collate_eval(batch):
    imgs, img_ids, hs, ws = zip(*batch)
    return torch.stack(imgs, dim=0), list(img_ids), list(hs), list(ws)


def evaluate() -> None:
    os.makedirs(args.save_folder, exist_ok=True)

    device = torch.device(
        'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    )

    if not os.path.exists(args.test_ids_path):
        sys.exit(
            f'ERROR: test IDs file not found at {args.test_ids_path}.\n'
            f'Run train.py first – it saves coco_test_ids.npy automatically.'
        )
    test_ids = np.load(args.test_ids_path, allow_pickle=True).tolist()
    print(f'Evaluating on {len(test_ids)} test images')

    if not os.path.exists(args.trained_model):
        sys.exit(f'ERROR: trained model not found: {args.trained_model}')

    model = SSD300(COCO_CONFIG, inference_mode=True)
    model.load_state_dict(
        torch.load(args.trained_model, map_location='cpu')
    )
    model.eval()
    model = model.to(device)
    if device.type == 'cuda':
        cudnn.benchmark = True
    print(f'Loaded model from {args.trained_model}')

    eval_dataset = _COCOEvalDataset(
        root       = args.dataset_root,
        image_set  = args.coco_image_set,
        image_ids  = test_ids,
    )
    eval_loader = data.DataLoader(
        eval_dataset,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        collate_fn  = _collate_eval,
        pin_memory  = (device.type == 'cuda'),
    )

    coco_results = []
    conf_thresh  = args.confidence_threshold

    with torch.no_grad():
        for imgs, img_ids, orig_hs, orig_ws in tqdm(eval_loader, desc='Evaluating'):
            imgs = imgs.to(device)
            detections = model(imgs)

            for b in range(detections.size(0)):
                img_id = img_ids[b]
                orig_h = orig_hs[b]
                orig_w = orig_ws[b]

                for cls_idx in range(1, detections.size(1)):
                    if cls_idx not in _CONTIGUOUS_TO_COCO_ID:
                        continue

                    dets = detections[b, cls_idx]
                    scores = dets[:, 0]
                    valid  = scores > conf_thresh
                    if not valid.any():
                        continue

                    kept_scores = scores[valid].cpu().numpy()
                    kept_boxes  = dets[valid, 1:].cpu().numpy()

                    kept_boxes[:, [0, 2]] *= orig_w
                    kept_boxes[:, [1, 3]] *= orig_h

                    coco_cat_id = _CONTIGUOUS_TO_COCO_ID[cls_idx]

                    for score, box in zip(kept_scores, kept_boxes):
                        x1, y1, x2, y2 = box.tolist()
                        coco_results.append({
                            'image_id':    img_id,
                            'category_id': coco_cat_id,
                            'bbox':  [x1, y1, x2 - x1, y2 - y1],
                            'score': float(score),
                        })

    print(f'Total detections collected: {len(coco_results)}')

    results_json = os.path.join(args.save_folder, 'detections.json')
    with open(results_json, 'w') as fh:
        json.dump(coco_results, fh)
    print(f'Detections written to {results_json}')

    if not coco_results:
        print('WARNING: no detections above threshold – mAP will be 0.')
        return

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_file = os.path.join(
        args.dataset_root, 'annotations',
        f'instances_{args.coco_image_set}.json'
    )
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results_json)

    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    evaluator.params.imgIds = test_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluator.summarize()
    summary_path = os.path.join(args.save_folder, 'map_results.txt')
    with open(summary_path, 'w') as fh:
        fh.write(buf.getvalue())
    print(f'\nSummary metrics saved -> {summary_path}')
    print('\nEvaluation complete.')


if __name__ == '__main__':
    evaluate()
