# SSD300 – My Implementation

Re-implementation of SSD300 (Liu et al., ECCV 2016) trained on a 25k-image subset of MS-COCO val2014.

Tested on **Kaggle with 2× T4 GPUs** (~4000 iterations, batch size 8).

---

## Running on Kaggle

1. Create a new Kaggle notebook and set the accelerator to **GPU T4 x2**.

2. Upload all files in this folder to the notebook or clone the repo:
   ```bash
   !git clone https://github.com/Aniket97/GNR-638-Assignment-3-SSD.git
   %cd GNR-638-Assignment-3-SSD/aniket_ssd_implementation
   ```

3. Install dependencies:
   ```bash
   !pip install pycocotools opencv-python-headless tqdm
   ```

4. Run training (downloads VGG weights and COCO data automatically):
   ```bash
   !python train.py
   ```
   Weights and logs are saved to `/kaggle/working/weights/`.

5. After training, run evaluation:
   ```bash
   !python test.py
   ```
   Results are saved to `/kaggle/working/eval/map_results.txt`.

---

## Running Locally or on a Separate Server

The scripts hardcode `/kaggle/working/` as the output directory. Before running, create that path manually:

```bash
# Linux / Mac
sudo mkdir -p /kaggle/working/weights
sudo mkdir -p /kaggle/working/data
sudo mkdir -p /kaggle/working/eval
sudo chmod -R 777 /kaggle/working

# Windows (run as Administrator)
mkdir C:\kaggle\working\weights
mkdir C:\kaggle\working\data
mkdir C:\kaggle\working\eval
```

Then install dependencies and run the same way:

```bash
pip install torch torchvision pycocotools opencv-python tqdm
python train.py
python test.py
```

> If you want to change the output path, edit `weights_dir` and `dataset_root` in the `_DEFAULTS` dict at the top of `train.py` and `test.py`.

---

## File Structure

```
aniket_ssd_implementation/
├── config.py              # SSD300 hyperparameters and anchor config
├── prior_boxes.py         # Default box / anchor generation
├── multibox_loss.py       # SSD loss (loc + cls with hard negative mining)
├── train.py               # Training script
├── test.py                # Evaluation script (COCO mAP)
├── model/
│   ├── ssd_net.py         # SSD300 model
│   └── vgg_backbone.py    # VGG16 backbone + extra layers
├── data/
│   ├── coco_dataset.py    # COCO dataset loader
│   └── augmentation.py    # Training augmentations
└── utils/
    └── box_ops.py         # IoU, box encoding, NMS
```
