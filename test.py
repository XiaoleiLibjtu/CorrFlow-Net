import argparse
import os

from pathlib import Path
from operator import add

import numpy as np
import cv2
from tqdm import tqdm
import torch

from network.model import CorrFlowNet
from utils.utils import create_dir, seeding, calculate_metrics


IMAGE_EXTS = [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]


def clean_stem(name):

    name = Path(name).name

    while True:
        p = Path(name)
        if p.suffix.lower() in IMAGE_EXTS:
            name = p.stem
        else:
            break

    return name


def find_file_by_stem(folder, stem):

    folder = Path(folder)

    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return str(p)

    hits = list(folder.glob(stem + ".*"))
    if len(hits) > 0:
        return str(hits[0])

    raise FileNotFoundError(f"Cannot find file with stem '{stem}' in {folder}")


def txt_has_samples(txt_path):

    txt_path = Path(txt_path)

    if not txt_path.exists():
        return False

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True

    return False


def choose_eval_split(data_path):

    data_path = Path(data_path)

    test_txt = data_path / "test.txt"

    if txt_has_samples(test_txt):
        return "test"

    return "val"


def load_split_data(data_path, split):

    data_path = Path(data_path)

    list_path = data_path / f"{split}.txt"
    image_dir = data_path / "images"
    mask_dir = data_path / "masks"

    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")

    xs, ys = [], []

    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue

            stem = clean_stem(item)

            x_path = find_file_by_stem(image_dir, stem)
            y_path = find_file_by_stem(mask_dir, stem)

            xs.append(x_path)
            ys.append(y_path)

    if len(xs) == 0:
        raise RuntimeError(f"No samples found in {list_path}")

    return xs, ys


def process_mask(y_pred):

    y_pred = y_pred[0].detach().cpu().numpy()   # [1, H, W]
    y_pred = np.squeeze(y_pred, axis=0)         # [H, W]
    y_pred = (y_pred > 0.5).astype(np.uint8) * 255
    y_pred = np.expand_dims(y_pred, axis=-1)    # [H, W, 1]
    return np.concatenate([y_pred, y_pred, y_pred], axis=2)


def print_score(metrics_score, num_samples):
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples
    acc = metrics_score[4] / num_samples
    f2 = metrics_score[5] / num_samples

    print(
        f'Jaccard: {jaccard:1.4f} - '
        f'F1: {f1:1.4f} - '
        f'Recall: {recall:1.4f} - '
        f'Precision: {precision:1.4f} - '
        f'Acc: {acc:1.4f} - '
        f'F2: {f2:1.4f}'
    )


def save_score(metrics_score, num_samples, save_file, eval_split):
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples
    acc = metrics_score[4] / num_samples
    f2 = metrics_score[5] / num_samples

    with open(save_file, "w", encoding="utf-8") as f:
        f.write(f"Eval split: {eval_split}\n")
        f.write(f"Samples: {num_samples}\n")
        f.write(f"Jaccard: {jaccard:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Acc: {acc:.4f}\n")
        f.write(f"F2: {f2:.4f}\n")


def load_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(path)

    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    return torch.from_numpy(image)


def load_mask(path, size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(path)

    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(mask, axis=0).astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    mask = np.expand_dims(mask, axis=0)

    return torch.from_numpy(mask)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CorrFlow-Net")
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=project_root / "datasets")
    parser.add_argument("--output-root", type=Path, default=project_root / "results")
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    seeding(42)

    dataset_name = args.dataset
    image_size = args.image_size
    size = (image_size, image_size)

    base_dir = args.data_root
    data_path = os.path.join(base_dir, dataset_name)

    checkpoint_path = args.checkpoint



    refine_noise_std = 0

    eval_split = choose_eval_split(data_path)
    print(f"Evaluating split: {eval_split}")

    save_path = os.path.join(args.output_root, dataset_name, 'CorrFlow-Net', eval_split)

    create_dir(save_path)
    create_dir(os.path.join(save_path, 'mask'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = CorrFlowNet()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)

    print(f'Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}')

    model = model.to(device)
    model.eval()

    eval_x, eval_y = load_split_data(data_path, eval_split)

    print(f"Number of eval samples: {len(eval_x)}")

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    with torch.no_grad():
        for x_path, y_path in tqdm(list(zip(eval_x, eval_y)), total=len(eval_x)):
            name = clean_stem(y_path)

            image = load_image(x_path, size).to(device, dtype=torch.float32)
            mask = load_mask(y_path, size).to(device, dtype=torch.float32)

            mask_pred, _, _ = model(image, refine_noise_std=refine_noise_std)

            score = calculate_metrics(mask, mask_pred)
            metrics_score = list(map(add, metrics_score, score))

            cv2.imwrite(
                os.path.join(save_path, 'mask', f'{name}.jpg'),
                process_mask(mask_pred)
            )

    print()
    print_score(metrics_score, len(eval_x))

    result_file = os.path.join(save_path, f'result_{eval_split}.txt')
    save_score(metrics_score, len(eval_x), result_file, eval_split)

    print(f"\nSaved results to: {save_path}")
    print(f"Saved score to: {result_file}")
