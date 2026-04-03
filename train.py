from __future__ import annotations

import os
import sys
import time
import tarfile
import urllib.request
import zipfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

from config import COCO_CONFIG, VGG_PIXEL_MEAN
from data.augmentation import SSDTrainTransform
from data.coco_dataset import COCODetection
from model.ssd_net import SSD300
from multibox_loss import SSDLoss


_DEFAULTS = dict(
    dataset_root   = '',
    vgg_weights    = 'vgg16_reducedfc.pth',
    weights_dir    = '/kaggle/working/weights/',
    resume         = None,
    start_iter     = 0,
    batch_size     = 8,
    num_workers    = 2,
    initial_lr     = 1e-4,
    momentum       = 0.9,
    weight_decay   = 5e-4,
    lr_decay_gamma = 0.1,
    use_cuda       = True,
    coco_image_set = 'val2014',
    log_interval   = 10,
    save_interval  = 5000,
)

args = SimpleNamespace(**_DEFAULTS)


def _download(url: str, dest: str, chunk: int = 8192) -> None:
    if os.path.exists(dest):
        print(f'[cached] {dest}')
        return
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    print(f'Downloading {url} ...')
    with urllib.request.urlopen(url) as src, open(dest, 'wb') as dst:
        while buf := src.read(chunk):
            dst.write(buf)
    print('Done.')


def _extract(path: str, target: str) -> None:
    os.makedirs(target, exist_ok=True)
    if path.endswith('.tar'):
        with tarfile.open(path) as tf:
            tf.extractall(target)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
    os.remove(path)


def prepare_env() -> None:
    data_dir = os.path.join('/kaggle/working', 'data')
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(data_dir,         exist_ok=True)

    vgg_path = os.path.join(args.weights_dir, args.vgg_weights)
    _download('https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth', vgg_path)

    args.dataset_root = os.path.join(data_dir, 'coco')
    ann_dir = os.path.join(args.dataset_root, 'annotations')
    if not os.path.isdir(ann_dir):
        img_zip = os.path.join(args.dataset_root, 'val2014.zip')
        ann_zip = os.path.join(args.dataset_root, 'annotations.zip')
        _download('http://images.cocodataset.org/zips/val2014.zip', img_zip)
        _download('http://images.cocodataset.org/annotations/annotations_trainval2014.zip', ann_zip)
        _extract(img_zip, os.path.join(args.dataset_root, 'images'))
        _extract(ann_zip, args.dataset_root)


def build_train_dataset():
    from pycocotools.coco import COCO

    ann_file = os.path.join(
        args.dataset_root, 'annotations',
        f'instances_{args.coco_image_set}.json'
    )
    if not os.path.exists(ann_file):
        sys.exit(f'ERROR: annotation file not found: {ann_file}')

    coco_api = COCO(ann_file)
    all_ids  = list(coco_api.imgToAnns.keys())
    print(f'Total annotated images in COCO {args.coco_image_set}: {len(all_ids)}')

    np.random.seed(42)
    np.random.shuffle(all_ids)

    train_ids = all_ids[:25000]
    test_ids  = all_ids[25000:26000]

    test_ids_path = os.path.join(args.weights_dir, 'coco_test_ids.npy')
    np.save(test_ids_path, test_ids)
    print(f'Train: {len(train_ids)} images | Test IDs saved -> {test_ids_path}')

    tfm = SSDTrainTransform(target_size=COCO_CONFIG.input_size, pixel_mean=VGG_PIXEL_MEAN)
    return COCODetection(
        root=args.dataset_root,
        image_set=args.coco_image_set,
        image_ids=train_ids,
        transform=tfm,
    )


def collate_detections(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)


def _set_lr(optimizer, new_lr: float) -> None:
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr


def train() -> None:
    prepare_env()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    cfg    = COCO_CONFIG

    vgg_path = os.path.join(args.weights_dir, args.vgg_weights)
    if args.resume:
        model = SSD300(cfg, inference_mode=False)
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print(f'Resumed from checkpoint: {args.resume}')
    else:
        model = SSD300.from_vgg_weights(cfg, vgg_path, inference_mode=False)

    model = model.to(device)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = SSDLoss(
        num_classes   = cfg.num_classes,
        iou_threshold = 0.5,
        neg_pos_ratio = 3,
        variances     = cfg.box_variances,
    )
    optimizer = optim.SGD(
        model.parameters(),
        lr           = args.initial_lr,
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
    )

    dataset = build_train_dataset()
    loader  = data.DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_detections,
        pin_memory  = True,
    )
    print(f'Training SSD300 on COCO | device: {device}')

    import traceback
    log_path = os.path.join(args.weights_dir, 'train_log.txt')
    log_fh   = open(log_path, 'a', buffering=1)

    def _log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + '\n')

    _log(f'Training started. Log -> {log_path}')

    step_count = 0
    batch_iter = iter(loader)

    for iteration in range(args.start_iter, cfg.max_iterations):
        if iteration in cfg.lr_decay_at:
            step_count += 1
            new_lr = args.initial_lr * (args.lr_decay_gamma ** step_count)
            _set_lr(optimizer, new_lr)
            _log(f'[iter {iteration}] LR -> {new_lr:.2e}')

        try:
            imgs, tgts = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            imgs, tgts = next(batch_iter)

        imgs = imgs.to(device)
        tgts = [t.to(device) for t in tgts]

        try:
            t0    = time.time()
            loc_preds, cls_preds, _ = model(imgs)
            core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            anchors = core_model.anchors
            preds = (loc_preds, cls_preds, anchors)

            optimizer.zero_grad()
            loc_loss, cls_loss = criterion(preds, tgts)
            total = loc_loss + cls_loss
            total.backward()
            optimizer.step()
            elapsed = time.time() - t0

            if iteration % args.log_interval == 0:
                _log('iter ' + repr(iteration) + ' || Loss: %.4f ||' % total.item() +
                     ' timer: %.4f sec.' % elapsed)

        except Exception:
            err_msg = (
                f'ERROR at iter {iteration}:\n' +
                traceback.format_exc()
            )
            _log(err_msg)
            continue

        if iteration != 0 and iteration % args.save_interval == 0:
            core = model.module if isinstance(model, torch.nn.DataParallel) else model
            ckpt = os.path.join(args.weights_dir, f'ssd300_COCO_{iteration}.pth')
            torch.save(core.state_dict(), ckpt)
            _log(f'Checkpoint -> {ckpt}')

    core  = model.module if isinstance(model, torch.nn.DataParallel) else model
    final = os.path.join(args.weights_dir, 'ssd300_COCO_final.pth')
    torch.save(core.state_dict(), final)
    _log(f'Training complete.  Weights -> {final}')
    log_fh.close()


if __name__ == '__main__':
    train()
