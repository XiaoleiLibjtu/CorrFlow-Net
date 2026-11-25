import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import calculate_metrics


def get_contour_mask(y, k=5):
    y = y.float()
    kernel = torch.ones(1, 1, k, k, device=y.device, dtype=y.dtype)
    y_d = F.conv2d(y, kernel, padding=k // 2)
    y_d = (y_d > 0).float()
    y_e = F.conv2d(y, kernel, padding=k // 2)
    y_e = (y_e == k * k).float()
    contour = y_d - y_e
    return (contour > 0).float()


def bce_dice_prob_loss(pred, target, smooth=1.0):
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    target = target.float()

    bce = F.binary_cross_entropy(pred, target)

    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(1)
    dice = 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)

    return bce + dice.mean()


def load_names(path, file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [line.strip() for line in file.readlines() if line.strip()]

    images = [os.path.join(path, "images", name) + ".jpg" for name in data]
    masks = [os.path.join(path, "masks", name) + ".jpg" for name in data]

    return images, masks


def load_data(path, val_name=None):
    train_names_path = os.path.join(path, "train.txt")

    if val_name is None:
        valid_names_path = os.path.join(path, "val.txt")
    else:
        valid_names_path = os.path.join(path, f"val_{val_name}.txt")

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)


def resolve_auxiliary_mask_path(mask_path, subdirs):
    """Resolve an auxiliary mask by the GT-mask filename."""

    if isinstance(subdirs, str):
        subdirs = [subdirs]

    mask_name = os.path.basename(mask_path)
    stem, _ = os.path.splitext(mask_name)
    dataset_root = os.path.dirname(os.path.dirname(mask_path))
    candidates = []

    for subdir in subdirs:
        folder = os.path.join(dataset_root, "error_masks", subdir)

        candidates.extend([
            os.path.join(folder, mask_name),
            os.path.join(folder, stem + ".png"),
            os.path.join(folder, stem + ".jpg"),
            os.path.join(folder, stem + ".jpeg"),
            os.path.join(folder, stem + ".bmp")
        ])

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Cannot find auxiliary mask for {mask_path}. Tried: {candidates}")


class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None, use_dynamic_masks=True):
        super().__init__()

        if len(images_path) != len(masks_path):
            raise ValueError("images_path and masks_path must have equal lengths.")

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size
        self.use_dynamic_masks = bool(use_dynamic_masks)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        mask_path = self.masks_path[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)


        if self.use_dynamic_masks:
            over_subdirs = ["dynamic_over_seg_masks", "over_seg_masks", "fp_masks"]
            under_subdirs = ["dynamic_under_seg_masks", "under_seg_masks", "fn_masks"]
        else:
            # Strict ablation: never read stale dynamic masks.
            over_subdirs = ["over_seg_masks", "fp_masks"]
            under_subdirs = ["under_seg_masks", "fn_masks"]

        over_seg_path = resolve_auxiliary_mask_path(mask_path, over_subdirs)
        under_seg_path = resolve_auxiliary_mask_path(mask_path, under_subdirs)
        contour_path = resolve_auxiliary_mask_path(mask_path, ["contour_k5", "boundary_k5"])

        over_seg_mask = cv2.imread(over_seg_path, cv2.IMREAD_GRAYSCALE)
        under_seg_mask = cv2.imread(under_seg_path, cv2.IMREAD_GRAYSCALE)
        contour_mask = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)

        if over_seg_mask is None:
            raise FileNotFoundError(over_seg_path)
        if under_seg_mask is None:
            raise FileNotFoundError(under_seg_path)
        if contour_mask is None:
            raise FileNotFoundError(contour_path)

        if self.transform is not None:
            aug = self.transform(
                image=image,
                mask=mask,
                over=over_seg_mask,
                under=under_seg_mask,
                contour=contour_mask
            )

            image = aug["image"]
            mask = aug["mask"]
            over_seg_mask = aug["over"]
            under_seg_mask = aug["under"]
            contour_mask = aug["contour"]

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0

        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        over_seg_mask = cv2.resize(over_seg_mask, self.size, interpolation=cv2.INTER_LINEAR)
        over_seg_mask = np.expand_dims(over_seg_mask, axis=0).astype(np.float32) / 255.0
        over_seg_mask = np.clip(over_seg_mask, 0.0, 1.0)

        under_seg_mask = cv2.resize(under_seg_mask, self.size, interpolation=cv2.INTER_LINEAR)
        under_seg_mask = np.expand_dims(under_seg_mask, axis=0).astype(np.float32) / 255.0
        under_seg_mask = np.clip(under_seg_mask, 0.0, 1.0)

        contour_mask = cv2.resize(contour_mask, self.size, interpolation=cv2.INTER_NEAREST)
        contour_mask = np.expand_dims(contour_mask, axis=0).astype(np.float32) / 255.0
        contour_mask = (contour_mask > 0.5).astype(np.float32)

        return image, (mask, over_seg_mask, under_seg_mask, contour_mask)

    def __len__(self):
        return self.n_samples


def correction_cue_adaptive_weights(
    contour_gt,
    over_seg_gt,
    under_seg_gt,
    base_contour=0.3,
    base_over=0.3,
    base_under=0.5,
    eps=1e-6,
    max_beta=100.0
):
    """Area-adaptive cue weights with empty-mask protection."""

    with torch.no_grad():
        rho_contour = contour_gt.float().mean()
        rho_over = over_seg_gt.float().mean()
        rho_under = under_seg_gt.float().mean()

        def raw_weight(rho, base):
            valid = (rho > eps).float()
            beta = (1.0 / torch.tanh(rho + eps)).clamp(max=max_beta)
            return float(base) * beta * valid

        w_contour = raw_weight(rho_contour, base_contour)
        w_over = raw_weight(rho_over, base_over)
        w_under = raw_weight(rho_under, base_under)

        total_base = float(base_contour) + float(base_over) + float(base_under)
        total_w = w_contour + w_over + w_under

        if float(total_w.cpu()) <= eps:
            device = contour_gt.device
            dtype = contour_gt.dtype

            lambda_contour = torch.tensor(float(base_contour), device=device, dtype=dtype)
            lambda_over = torch.tensor(float(base_over), device=device, dtype=dtype)
            lambda_under = torch.tensor(float(base_under), device=device, dtype=dtype)
        else:
            lambda_contour = total_base * w_contour / (total_w + eps)
            lambda_over = total_base * w_over / (total_w + eps)
            lambda_under = total_base * w_under / (total_w + eps)

    return lambda_contour, lambda_over, lambda_under


def compute_total_loss(
    mask_pred,
    contour_pred,
    aux,
    y,
    over_seg_gt=None,
    under_seg_gt=None,
    contour_gt=None,
    lambda_contour=0.3,
    lambda_over=0.3,
    lambda_under=0.5,
    lambda_direct=0.3,
    lambda_correction_flow=0.1,
    return_details=False
):
    if contour_gt is None:
        contour_gt = get_contour_mask(y, k=5)

    contour_gt_for_contour = F.interpolate(
        contour_gt.float(),
        size=contour_pred.shape[-2:],
        mode="nearest"
    )

    loss_mask = bce_dice_prob_loss(mask_pred, y)
    loss_contour = bce_dice_prob_loss(contour_pred, contour_gt_for_contour)

    total = loss_mask

    details = {
        "refinement_mode": aux.get("refinement_mode", "unknown"),
        "loss_mask": float(loss_mask.detach().cpu()),
        "loss_contour": float(loss_contour.detach().cpu())
    }

    direct_mask = aux.get("direct_mask")

    if direct_mask is not None:
        direct_gt = y.float()

        if direct_gt.shape[-2:] != direct_mask.shape[-2:]:
            direct_gt = F.interpolate(direct_gt, size=direct_mask.shape[-2:], mode="nearest")

        loss_direct = bce_dice_prob_loss(direct_mask, direct_gt)
        total = total + lambda_direct * loss_direct

        details.update({
            "loss_direct": float(loss_direct.detach().cpu()),
            "lambda_direct": float(lambda_direct)
        })

    has_cue_supervision = (
        over_seg_gt is not None
        and under_seg_gt is not None
        and aux.get("over_seg_pred") is not None
        and aux.get("under_seg_pred") is not None
    )

    if has_cue_supervision:
        over_seg_for_loss = F.interpolate(
            over_seg_gt.float(),
            size=aux["over_seg_pred"].shape[-2:],
            mode="bilinear",
            align_corners=False
        ).clamp(0.0, 1.0)

        under_seg_for_loss = F.interpolate(
            under_seg_gt.float(),
            size=aux["under_seg_pred"].shape[-2:],
            mode="bilinear",
            align_corners=False
        ).clamp(0.0, 1.0)

        contour_for_adapt = F.interpolate(
            contour_gt.float(),
            size=over_seg_for_loss.shape[-2:],
            mode="nearest"
        )

        loss_over = bce_dice_prob_loss(aux["over_seg_pred"], over_seg_for_loss)
        loss_under = bce_dice_prob_loss(aux["under_seg_pred"], under_seg_for_loss)

        lambda_contour_adapt, lambda_over_adapt, lambda_under_adapt = correction_cue_adaptive_weights(
            contour_gt=contour_for_adapt,
            over_seg_gt=over_seg_for_loss,
            under_seg_gt=under_seg_for_loss,
            base_contour=lambda_contour,
            base_over=lambda_over,
            base_under=lambda_under
        )

        total = (
            total
            + lambda_contour_adapt * loss_contour
            + lambda_over_adapt * loss_over
            + lambda_under_adapt * loss_under
        )

        details.update({
            "loss_over_seg": float(loss_over.detach().cpu()),
            "loss_under_seg": float(loss_under.detach().cpu()),
            "lambda_contour_adapt": float(lambda_contour_adapt.detach().cpu()),
            "lambda_over_adapt": float(lambda_over_adapt.detach().cpu()),
            "lambda_under_adapt": float(lambda_under_adapt.detach().cpu())
        })
    else:
        total = total + lambda_contour * loss_contour

    correction_field_pred = aux.get("correction_field_pred")
    correction_field_target = aux.get("correction_field_target")

    if correction_field_pred is not None and correction_field_target is not None:
        loss_correction_flow = F.mse_loss(correction_field_pred, correction_field_target)
        total = total + lambda_correction_flow * loss_correction_flow
        details["loss_correction_flow"] = float(loss_correction_flow.detach().cpu())

    if return_details:
        details["loss_total"] = float(total.detach().cpu())
        return total, details

    return total


def clear_dynamic_fp_fn_masks(images_path, masks_path):
    """Remove old dynamic FP/FN masks before a fresh run."""

    del images_path

    dataset_roots = set()

    for mask_path in masks_path:
        dataset_roots.add(os.path.dirname(os.path.dirname(mask_path)))

    removed = 0

    for dataset_root in dataset_roots:
        for subdir in ["dynamic_over_seg_masks", "dynamic_under_seg_masks"]:
            folder = os.path.join(dataset_root, "error_masks", subdir)

            if not os.path.isdir(folder):
                continue

            for name in os.listdir(folder):
                path = os.path.join(folder, name)

                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1

    return removed


def _prepare_image_tensor_for_dynamic_update(image_path, size, device):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    return torch.from_numpy(image).to(device, dtype=torch.float32)


def _prepare_mask_tensor_for_dynamic_update(mask_path, size, device):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(mask_path)

    original_h, original_w = mask.shape[:2]

    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)

    tensor = torch.from_numpy(mask).to(device, dtype=torch.float32)

    return tensor, (original_h, original_w)


def update_dynamic_fp_fn_masks(
    model,
    images_path,
    masks_path,
    size,
    device,
    threshold=0.5,
    use_final_prediction=True,
    save_ext=".png",
    soft_error=True
):
    """Regenerate dynamic false-positive and false-negative masks."""

    was_training = model.training
    model.eval()

    total_fp_value = 0.0
    total_fn_value = 0.0
    total_pixels = 0.0

    description = "Updating dynamic soft FP/FN masks" if soft_error else "Updating dynamic hard FP/FN masks"

    with torch.no_grad():
        pairs = list(zip(images_path, masks_path))

        for image_path, mask_path in tqdm(pairs, total=len(images_path), desc=description):
            image_tensor = _prepare_image_tensor_for_dynamic_update(
                image_path=image_path,
                size=size,
                device=device
            )

            gt_tensor, original_size = _prepare_mask_tensor_for_dynamic_update(
                mask_path=mask_path,
                size=size,
                device=device
            )

            mask_pred, _, aux = model(image_tensor)

            if not use_final_prediction and aux.get("direct_mask") is not None:
                pred_prob = aux["direct_mask"]
            else:
                pred_prob = mask_pred

            pred_prob = pred_prob.clamp(0.0, 1.0)
            gt_bin = (gt_tensor > 0.5).float()

            if soft_error:
                fp = pred_prob * (1.0 - gt_bin)
                fn = gt_bin * (1.0 - pred_prob)
            else:
                pred_bin = (pred_prob > threshold).float()
                fp = pred_bin * (1.0 - gt_bin)
                fn = gt_bin * (1.0 - pred_bin)

            total_fp_value += float(fp.sum().detach().cpu())
            total_fn_value += float(fn.sum().detach().cpu())
            total_pixels += float(fp.numel())

            fp_np = fp[0, 0].detach().cpu().numpy()
            fn_np = fn[0, 0].detach().cpu().numpy()

            original_h, original_w = original_size
            resize_mode = cv2.INTER_LINEAR if soft_error else cv2.INTER_NEAREST

            fp_np = cv2.resize(fp_np, (original_w, original_h), interpolation=resize_mode)
            fn_np = cv2.resize(fn_np, (original_w, original_h), interpolation=resize_mode)

            fp_np = np.clip(fp_np, 0.0, 1.0)
            fn_np = np.clip(fn_np, 0.0, 1.0)

            if soft_error:
                fp_save = np.round(fp_np * 255.0).astype(np.uint8)
                fn_save = np.round(fn_np * 255.0).astype(np.uint8)
            else:
                fp_save = (fp_np > 0.5).astype(np.uint8) * 255
                fn_save = (fn_np > 0.5).astype(np.uint8) * 255

            dataset_root = os.path.dirname(os.path.dirname(mask_path))
            stem, _ = os.path.splitext(os.path.basename(mask_path))

            fp_dir = os.path.join(dataset_root, "error_masks", "dynamic_over_seg_masks")
            fn_dir = os.path.join(dataset_root, "error_masks", "dynamic_under_seg_masks")

            os.makedirs(fp_dir, exist_ok=True)
            os.makedirs(fn_dir, exist_ok=True)

            cv2.imwrite(os.path.join(fp_dir, stem + save_ext), fp_save)
            cv2.imwrite(os.path.join(fn_dir, stem + save_ext), fn_save)

    if was_training:
        model.train()

    return {
        "fp_ratio": total_fp_value / max(total_pixels, 1.0),
        "fn_ratio": total_fn_value / max(total_pixels, 1.0),
        "num_samples": len(images_path),
        "soft_error": bool(soft_error)
    }


def _update_metric_sums(y, mask_pred, metric_sums):
    sample_count = 0

    for y_true, y_pred in zip(y, mask_pred):
        score = calculate_metrics(y_true, y_pred)

        metric_sums[0] += float(score[0])
        metric_sums[1] += float(score[1])
        metric_sums[2] += float(score[2])
        metric_sums[3] += float(score[3])
        sample_count += 1

    return sample_count


def train(
    model,
    loader,
    optimizer,
    device,
    lambda_contour=0.3,
    lambda_over=0.1,
    lambda_under=0.1,
    lambda_direct=0.3,
    lambda_correction_flow=0.1
):
    model.train()

    epoch_loss = 0.0
    metric_sums = [0.0, 0.0, 0.0, 0.0]
    total_samples = 0

    for x, (y, over_seg_gt, under_seg_gt, contour_gt) in tqdm(
        loader,
        desc="Training",
        total=len(loader)
    ):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.float32, non_blocking=True)
        over_seg_gt = over_seg_gt.to(device, dtype=torch.float32, non_blocking=True)
        under_seg_gt = under_seg_gt.to(device, dtype=torch.float32, non_blocking=True)
        contour_gt = contour_gt.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        mask_pred, contour_pred, aux = model(
            x,
            gt_mask=y,
            refine_noise_std=0.0
        )

        loss = compute_total_loss(
            mask_pred=mask_pred,
            contour_pred=contour_pred,
            aux=aux,
            y=y,
            over_seg_gt=over_seg_gt,
            under_seg_gt=under_seg_gt,
            contour_gt=contour_gt,
            lambda_contour=lambda_contour,
            lambda_over=lambda_over,
            lambda_under=lambda_under,
            lambda_direct=lambda_direct,
            lambda_correction_flow=lambda_correction_flow
        )

        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.detach().cpu())
        total_samples += _update_metric_sums(y, mask_pred, metric_sums)

    n_batches = len(loader)

    if n_batches == 0:
        raise RuntimeError("Training loader is empty.")

    if total_samples == 0:
        raise RuntimeError("No training samples were processed.")

    metrics = [value / total_samples for value in metric_sums]

    return epoch_loss / n_batches, metrics


def evaluate(
    model,
    loader,
    device,
    lambda_contour=0.3,
    lambda_over=0.1,
    lambda_under=0.1,
    lambda_direct=0.3,
    lambda_correction_flow=0.1
):
    model.eval()

    epoch_loss = 0.0
    metric_sums = [0.0, 0.0, 0.0, 0.0]
    total_samples = 0

    with torch.no_grad():
        for x, (y, over_seg_gt, under_seg_gt, contour_gt) in tqdm(
            loader,
            desc="Evaluation",
            total=len(loader)
        ):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            over_seg_gt = over_seg_gt.to(device, dtype=torch.float32, non_blocking=True)
            under_seg_gt = under_seg_gt.to(device, dtype=torch.float32, non_blocking=True)
            contour_gt = contour_gt.to(device, dtype=torch.float32, non_blocking=True)

            mask_pred, contour_pred, aux = model(
                x,
                gt_mask=y,
                correction_time=0.5,
                refine_noise_std=0.0
            )

            loss = compute_total_loss(
                mask_pred=mask_pred,
                contour_pred=contour_pred,
                aux=aux,
                y=y,
                over_seg_gt=over_seg_gt,
                under_seg_gt=under_seg_gt,
                contour_gt=contour_gt,
                lambda_contour=lambda_contour,
                lambda_over=lambda_over,
                lambda_under=lambda_under,
                lambda_direct=lambda_direct,
                lambda_correction_flow=lambda_correction_flow
            )

            epoch_loss += float(loss.detach().cpu())
            total_samples += _update_metric_sums(y, mask_pred, metric_sums)

    n_batches = len(loader)

    if n_batches == 0:
        raise RuntimeError("Evaluation loader is empty.")

    if total_samples == 0:
        raise RuntimeError("No evaluation samples were processed.")

    metrics = [value / total_samples for value in metric_sums]

    return epoch_loss / n_batches, metrics


DATASET = SegmentationDataset
