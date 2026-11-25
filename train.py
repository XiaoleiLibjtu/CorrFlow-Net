import argparse
from pathlib import Path
import datetime
import os
import random
import time

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from network.model import CorrFlowNet
from network.checkpoint import normalize_state_dict
from utils.run_engine import SegmentationDataset, clear_dynamic_fp_fn_masks, evaluate, load_data, train, update_dynamic_fp_fn_masks
from utils.utils import epoch_time, print_and_save, shuffling


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_weights(model, pretrained_path, prefixes=None):
   
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    pretrained_dict = normalize_state_dict(checkpoint)

    model_dict = model.state_dict()
    matched_dict = {}

    for name, weight in pretrained_dict.items():
        if prefixes is not None and not name.startswith(prefixes):
            continue
        if name not in model_dict:
            continue
        if model_dict[name].shape != weight.shape:
            continue
        matched_dict[name] = weight

    if not matched_dict:
        raise ValueError("No matching pretrained tensors found.")

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CorrFlow-Net")
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--dataset", default="GlaS")
    parser.add_argument("--data-root", type=Path, default=project_root / "datasets")
    parser.add_argument("--output-root", type=Path, default=project_root / "run_files")
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--correction-steps", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset
    val_name = None

    seed = 0
    my_seeding(seed)

    image_size = args.image_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = 1e-4
    early_stopping_patience = 100

    resume_path = args.resume
    pretrained_path = args.pretrained
    pretrained_prefixes = None

    lambda_contour = 0.1
    lambda_over = 0.1
    lambda_under = 0.1
    lambda_direct = 0.3
    lambda_correction_flow = 0.2

    correction_steps = args.correction_steps

    dynamic_fpfn_enabled = True
    dynamic_fpfn_start_epoch = 5
    dynamic_fpfn_end_epoch = 100
    dynamic_fpfn_threshold = 0.5
    dynamic_fpfn_use_final_prediction = True
    dynamic_fpfn_soft_error = True
    dynamic_fpfn_clear_at_start = resume_path is None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{dataset_name}_{val_name}_lr{lr}_{current_time}"

    base_dir = args.data_root
    data_path = os.path.join(base_dir, dataset_name)
    save_dir = os.path.join(args.output_root, dataset_name, folder_name)

    os.makedirs(save_dir, exist_ok=True)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    csv_log_path = os.path.join(save_dir, "train_log.csv")

    with open(train_log_path, "w", encoding="utf-8") as file:
        file.write("\n")

    print_and_save(train_log_path, str(datetime.datetime.now()))

    hyperparameters_str = (
        f"Image Size: {image_size}\n"
        f"Batch Size: {batch_size}\n"
        f"LR: {lr}\n"
        f"Epochs: {num_epochs}\n"
        f"Early Stopping Patience: {early_stopping_patience}\n"
        f"Seed: {seed}\n"
        f"Base Lambda Contour: {lambda_contour}\n"
        f"Base Lambda Over-segmentation: {lambda_over}\n"
        f"Base Lambda Under-segmentation: {lambda_under}\n"
        f"Lambda Direct: {lambda_direct}\n"
        f"Lambda Correction Flow: {lambda_correction_flow}\n"
        "Adaptive Weighting: correction-cue area-adaptive weighting\n"
        "Final Output: refined_mask only\n"
        f"Correction Steps: {correction_steps}\n"
        f"Dynamic FP/FN Enabled: {dynamic_fpfn_enabled}\n"
        f"Dynamic FP/FN Start Epoch: {dynamic_fpfn_start_epoch}\n"
        f"Dynamic FP/FN End Epoch: {dynamic_fpfn_end_epoch}\n"
        f"Dynamic FP/FN Threshold: {dynamic_fpfn_threshold}\n"
        f"Dynamic FP/FN Use Final Prediction: {dynamic_fpfn_use_final_prediction}\n"
        f"Dynamic FP/FN Soft Error: {dynamic_fpfn_soft_error}\n"
        f"Dynamic FP/FN Clear At Start: {dynamic_fpfn_clear_at_start}\n"
    )

    print_and_save(train_log_path, hyperparameters_str)

    transform = A.Compose(
        [
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(
                p=0.3,
                num_holes_range=(1, 10),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32)
            )
        ],
        additional_targets={"over": "mask", "under": "mask", "contour": "mask"}
    )

    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, val_name)
    train_x, train_y = shuffling(train_x, train_y)

    print_and_save(
        train_log_path,
        f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    )

    if dynamic_fpfn_enabled and dynamic_fpfn_clear_at_start:
        removed_dynamic_masks = clear_dynamic_fp_fn_masks(train_x, train_y)
        print_and_save(train_log_path, f"Cleared stale dynamic FP/FN masks: {removed_dynamic_masks}\n")

    train_dataset = SegmentationDataset(train_x, train_y, (image_size, image_size), transform=transform)
    valid_dataset = SegmentationDataset(valid_x, valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CorrFlowNet(
        backbone_weights=args.backbone_weights,
        correction_steps=correction_steps
    )

    if pretrained_path:
        load_pretrained_weights(model=model, pretrained_path=pretrained_path, prefixes=pretrained_prefixes)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        checkpoint = normalize_state_dict(checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        print_and_save(
            train_log_path,
            f"Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}\n"
        )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=30)

    print_and_save(
        train_log_path,
        (
            "Optimizer: Adam\n"
            "Loss: L_refined + lambda_direct*L_direct "
            "+ adaptive cue loss(L_contour, L_over, L_under) "
            "+ lambda_correction_flow*L_flow\n"
            "Modules: DC2M (cue disentanglement + ACGC), CPFC, and best-model-guided dynamic FP/FN refresh enabled.\n"
        )
    )

    num_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print_and_save(train_log_path, f"Number of parameters: {num_params / 1_000_000:.4f}M\n")

    with open(csv_log_path, "w", encoding="utf-8") as file:
        file.write(
            "epoch,train_loss,train_mIoU,train_f1,"
            "train_recall,train_precision,"
            "valid_loss,valid_mIoU,valid_f1,"
            "valid_recall,valid_precision\n"
        )

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        current_epoch = epoch + 1

        train_loss, train_metrics = train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_contour=lambda_contour,
            lambda_over=lambda_over,
            lambda_under=lambda_under,
            lambda_direct=lambda_direct,
            lambda_correction_flow=lambda_correction_flow
        )

        valid_loss, valid_metrics = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            lambda_contour=lambda_contour,
            lambda_over=lambda_over,
            lambda_under=lambda_under,
            lambda_direct=lambda_direct,
            lambda_correction_flow=lambda_correction_flow
        )

        scheduler.step(valid_metrics[0])

        if valid_metrics[0] > best_valid_metrics:
            print_and_save(
                train_log_path,
                (
                    f"Valid mIoU improved from {best_valid_metrics:2.4f} "
                    f"to {valid_metrics[0]:2.4f}. "
                    f"Saving checkpoint: {checkpoint_path}"
                )
            )

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)

            if dynamic_fpfn_enabled and dynamic_fpfn_start_epoch <= current_epoch <= dynamic_fpfn_end_epoch:
                print_and_save(
                    train_log_path,
                    f"Updating dynamic FP/FN masks using current best refined prediction at epoch {current_epoch}...\n"
                )

                dynamic_stats = update_dynamic_fp_fn_masks(
                    model=model,
                    images_path=train_x,
                    masks_path=train_y,
                    size=(image_size, image_size),
                    device=device,
                    threshold=dynamic_fpfn_threshold,
                    use_final_prediction=dynamic_fpfn_use_final_prediction,
                    soft_error=dynamic_fpfn_soft_error
                )

                print_and_save(
                    train_log_path,
                    (
                        f"Dynamic FP/FN updated: samples={dynamic_stats['num_samples']} | "
                        f"fp_ratio={dynamic_stats['fp_ratio']:.6f} | "
                        f"fn_ratio={dynamic_stats['fn_ratio']:.6f} | "
                        f"soft_error={dynamic_stats['soft_error']}\n"
                    )
                )

            elif dynamic_fpfn_enabled and current_epoch < dynamic_fpfn_start_epoch:
                print_and_save(
                    train_log_path,
                    (
                        f"Dynamic FP/FN update skipped before warm-up epoch "
                        f"{dynamic_fpfn_start_epoch}. Current epoch: {current_epoch}.\n"
                    )
                )

            elif dynamic_fpfn_enabled and current_epoch > dynamic_fpfn_end_epoch:
                print_and_save(
                    train_log_path,
                    (
                        f"Dynamic FP/FN update stopped after epoch "
                        f"{dynamic_fpfn_end_epoch}. Current epoch: {current_epoch}.\n"
                    )
                )

            early_stopping_count = 0

        else:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        current_lr = optimizer.param_groups[0]["lr"]

        data_str = (
            f"Epoch: {current_epoch:02} | "
            f"Epoch Time: {epoch_mins}m {epoch_secs}s | "
            f"LR: {current_lr:.6e}\n"
            f"\tTrain Loss: {train_loss:.4f} - "
            f"mIoU: {train_metrics[0]:.4f} - "
            f"F1: {train_metrics[1]:.4f} - "
            f"Recall: {train_metrics[2]:.4f} - "
            f"Precision: {train_metrics[3]:.4f}\n"
            f"\t Val. Loss: {valid_loss:.4f} - "
            f"mIoU: {valid_metrics[0]:.4f} - "
            f"F1: {valid_metrics[1]:.4f} - "
            f"Recall: {valid_metrics[2]:.4f} - "
            f"Precision: {valid_metrics[3]:.4f}\n"
        )

        print_and_save(train_log_path, data_str)

        with open(csv_log_path, "a", encoding="utf-8") as file:
            file.write(
                f"{current_epoch},{train_loss},"
                f"{train_metrics[0]},{train_metrics[1]},"
                f"{train_metrics[2]},{train_metrics[3]},"
                f"{valid_loss},{valid_metrics[0]},"
                f"{valid_metrics[1]},{valid_metrics[2]},"
                f"{valid_metrics[3]}\n"
            )

        if early_stopping_count >= early_stopping_patience:
            print_and_save(
                train_log_path,
                f"Early stopping: validation stops improving for {early_stopping_patience} epochs.\n"
            )
            break
