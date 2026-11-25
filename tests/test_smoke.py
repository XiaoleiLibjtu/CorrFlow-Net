"""Synthetic release checks: python -m unittest discover -s tests -v."""

import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from network import CorrFlowNet
from network.checkpoint import LEGACY_PREFIXES
from tools.prepare_cues import prepare_cues, read_gray
from utils.run_engine import compute_total_loss, get_contour_mask


class ReleaseSmokeTests(unittest.TestCase):
    def test_model_training_and_checkpoint_compatibility(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        model = CorrFlowNet()
        self.assertEqual(model.correction_steps, 4)
        inverse = {new: old for old, new in LEGACY_PREFIXES.items()}
        legacy = OrderedDict()
        for key, value in model.state_dict().items():
            for new, old in inverse.items():
                if key.startswith(new + "."):
                    key = old + key[len(new):]
                    break
            legacy["module." + key] = value
        status = model.load_state_dict({"state_dict": legacy}, strict=True)
        self.assertFalse(status.missing_keys)
        self.assertFalse(status.unexpected_keys)
        image = torch.rand(2, 3, 64, 64)
        gt = (torch.rand(2, 1, 64, 64) > 0.5).float()
        model.eval()
        with torch.no_grad():
            prediction, contour, aux = model(image)
        self.assertEqual(prediction.shape, gt.shape)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(((prediction > 0) & (prediction < 1)).all())
        model.train()
        prediction, contour, aux = model(image, gt_mask=gt, correction_time=0.5)
        loss = compute_total_loss(
            prediction, contour, aux, gt,
            over_seg_gt=gt * 0.2, under_seg_gt=gt * 0.3,
            lambda_contour=0.1, lambda_over=0.1, lambda_under=0.1,
            lambda_direct=0.3, lambda_correction_flow=0.2,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))

    def test_auxiliary_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / "masks"
            refs = root / "references"
            masks.mkdir()
            refs.mkdir()
            gt = np.zeros((16, 16), dtype=np.uint8)
            gt[:8, :8] = 255
            reference = np.full((16, 16), 128, dtype=np.uint8)
            for path, image in ((masks / "sample.png", gt), (refs / "sample.png", reference)):
                ok, data = cv2.imencode(".png", image)
                self.assertTrue(ok)
                path.write_bytes(data.tobytes())
            self.assertEqual(prepare_cues(root, refs), 1)
            cue_root = root / "error_masks"
            over = read_gray(cue_root / "over_seg_masks/sample.png")
            under = read_gray(cue_root / "under_seg_masks/sample.png")
            contour = read_gray(cue_root / "contour_k5/sample.png")
            np.testing.assert_array_equal(over, np.where(gt == 0, 128, 0).astype(np.uint8))
            np.testing.assert_array_equal(under, np.where(gt > 0, 127, 0).astype(np.uint8))
            expected = get_contour_mask(torch.from_numpy((gt > 0).astype(np.float32))[None, None])
            np.testing.assert_array_equal(contour, expected[0, 0].numpy().astype(np.uint8) * 255)


if __name__ == "__main__":
    unittest.main()
