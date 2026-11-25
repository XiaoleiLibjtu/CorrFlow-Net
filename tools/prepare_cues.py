"""Generate correction-cue targets from existing reference predictions."""

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_gray(path):
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_mask(path, probability):
    image = np.round(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    path.write_bytes(encoded.tobytes())


def prepare_cues(dataset_dir, reference_dir):
    mask_dir = dataset_dir / "masks"
    if not mask_dir.is_dir() or not reference_dir.is_dir():
        raise FileNotFoundError("Dataset masks/ and reference directory must exist.")
    masks = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not masks:
        raise ValueError(f"No ground-truth masks found in {mask_dir}")
    references = {}
    for path in reference_dir.iterdir():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in references:
                raise ValueError(f"Ambiguous reference stem: {path.stem}")
            references[path.stem] = path
    missing = [p.stem for p in masks if p.stem not in references]
    if missing:
        raise FileNotFoundError(f"Missing reference predictions: {missing[:10]}")
    if len({p.stem for p in masks}) != len(masks):
        raise ValueError("Ground-truth mask stems must be unique.")
    output_dirs = {
        key: dataset_dir / "error_masks" / folder
        for key, folder in (("over", "over_seg_masks"), ("under", "under_seg_masks"), ("contour", "contour_k5"))
    }
    for folder in output_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    kernel = np.ones((5, 5), dtype=np.uint8)
    for path in masks:
        gt = (read_gray(path) > 127).astype(np.uint8)
        reference = read_gray(references[path.stem]).astype(np.float32) / 255.0
        if reference.shape != gt.shape:
            raise ValueError(f"Reference and GT sizes differ for {path.stem}")
        # Zero padding matches get_contour_mask() in the training engine.
        dilated = cv2.dilate(gt, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        eroded = cv2.erode(gt, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        write_mask(output_dirs["over"] / f"{path.stem}.png", reference * (1.0 - gt))
        write_mask(output_dirs["under"] / f"{path.stem}.png", gt * (1.0 - reference))
        write_mask(output_dirs["contour"] / f"{path.stem}.png", dilated - eroded)
    return len(masks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    count = prepare_cues(args.dataset_dir, args.reference_dir)
    print(f"Prepared correction cues for {count} samples.")
