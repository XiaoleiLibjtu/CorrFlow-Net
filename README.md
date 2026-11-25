# CorrFlow-Net

**CorrFlow-Net: Continuous Error Correction via Flow Matching for Medical Image Segmentation**

This repository provides the PyTorch implementation of CorrFlow-Net. The network combines cue-guided feature correction with continuous prediction-space refinement for binary medical image segmentation.

## Overview

- **DC2M — Decouple-then-Calibrate Feature Correction Module:** disentangles over-segmentation suppression, under-segmentation recovery, and auxiliary spatial localization cues.
- **ACGC — Adaptive Cue-Guided Calibration:** routes the correction cues into four feature stages using spatial/channel gating and region-level cross-attention.
- **CPFC — Continuous Prediction-Space Flow Correction:** predicts a state- and time-conditioned velocity field and refines the direct segmentation through projected Euler integration.

```text
Image → PVTv2-B2 → DC2M → Decoder → Direct mask
                                      ↓
                     Decoder features + CPFC → Refined mask
```

The model returns the refined probability map, the auxiliary contour prediction, and a dictionary of intermediate predictions used for training.

## Installation

Use Python 3.9 or later and install a PyTorch build suitable for your device. Then install the project dependencies:

```bash
git clone https://github.com/XiaoleiLibjtu/CorrFlow-Net.git
cd CorrFlow-Net
pip install -r requirements.txt
```

Local smoke checks used PyTorch 2.0.1, timm 1.0.28, OpenCV 4.12.0, and Albumentations 2.0.8. The release does not include datasets, pretrained backbone weights, or trained segmentation checkpoints.

## Data preparation

Organize each dataset as follows:

```text
datasets/GlaS/
├── images/
│   ├── sample_001.jpg
│   └── sample_002.jpg
├── masks/
│   ├── sample_001.jpg
│   └── sample_002.jpg
├── train.txt
├── val.txt
├── test.txt                     # optional
└── error_masks/
    ├── over_seg_masks/
    ├── under_seg_masks/
    └── contour_k5/
```

Each split file contains one sample name per line, without a file extension. Training uses `.jpg` images and ground-truth masks. Evaluation also supports PNG, JPEG, BMP, and TIFF files. Ground-truth masks should encode background as 0 and foreground as 255.

Training and training-time validation require three auxiliary supervision maps. Given a reference probability map `R` and binary ground truth `G`, the maps are:

```text
over-segmentation:  R * (1 - G)
under-segmentation: G * (1 - R)
contour:           dilate(G, 5×5) - erode(G, 5×5)
```

Generate the three supervision maps from reference predictions for the training and validation samples using the formulas above, and save them in the corresponding `error_masks/` directories as 8-bit images with values from 0 to 255. Filenames must match the ground-truth stems. The paper initializes reference predictions with MedSAM2; reference inference and auxiliary-map generation are performed outside this repository.

The trainer refreshes training FP/FN maps from the best validation checkpoint during its configured refresh window. Evaluation does not need reference predictions or auxiliary supervision maps.

## Training

Download the PVTv2-B2 pretrained backbone from the [PVT project](https://github.com/whai362/PVT/tree/v2), then specify its path explicitly:

```bash
python train.py --dataset GlaS --data-root ./datasets --backbone-weights ./pretrained_pth/pvt_v2_b2.pth
```

If `--backbone-weights` is omitted, the backbone is initialized randomly. Training from random initialization is not equivalent to the pretrained setup.

| Setting | Default |
| --- | --- |
| Input size | 256 × 256 |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Maximum epochs | 300 |
| Early-stopping patience | 100 epochs |
| Base cue weights: contour / over / under | 0.1 / 0.1 / 0.1 |
| Direct segmentation loss weight | 0.3 |
| Flow-matching loss weight | 0.2 |

Checkpoints and logs are saved under `run_files/<dataset>/<run>/`. Use `python train.py --help` for configuration options. `--pretrained` loads matching model tensors; `--resume` reloads model weights only and does not restore optimizer, scheduler, epoch, or best validation score.

## Evaluation

```bash
python test.py --dataset GlaS --data-root ./datasets --checkpoint ./run_files/GlaS/YOUR_RUN/checkpoint.pth
```

Replace `YOUR_RUN` with your actual run directory. The script uses a nonempty `test.txt` when available, otherwise `val.txt`. Predicted masks and per-image-averaged metrics are saved under `results/<dataset>/CorrFlow-Net/<split>/`.

The refinement schedule is configured inside the model; the evaluation command does not expose a correction-step option.

## Python inference

```python
import torch
from network import CorrFlowNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CorrFlowNet()
model.load_state_dict(torch.load("checkpoint.pth", map_location="cpu"), strict=True)
model = model.to(device).eval()

# Match the training preprocessing: OpenCV BGR, values in [0, 1].
image = torch.rand(1, 3, 256, 256, device=device)
with torch.no_grad():
    probability, contour, auxiliary = model(image)
    binary_mask = probability > 0.5
```

The loader supports legacy CorrFlow module names, DataParallel `module.` prefixes, and common checkpoint wrappers.

## Acknowledgements

The PVTv2 backbone is adapted from [PVT](https://github.com/whai362/PVT/tree/v2). The implementation also uses [timm](https://github.com/huggingface/pytorch-image-models), [PyTorch](https://pytorch.org/), and [Albumentations](https://albumentations.ai/). Reference-mask initialization in the paper uses [MedSAM2](https://github.com/bowang-lab/MedSAM2).

## License

A license for the original CorrFlow-Net code has not yet been selected. Third-party components retain their original licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
