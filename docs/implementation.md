# Implementation notes

## Module layout

| Paper component | Python implementation |
| --- | --- |
| CorrFlow-Net | `network.model.CorrFlowNet` |
| DC2M | `DecoupleThenCalibrateFeatureCorrection`, `model.dc2m` |
| Direction-aware cue disentanglement | `DirectionAwareCueDisentanglement`, `dc2m.cue_disentanglement` |
| ACGC | `AdaptiveCueGuidedCalibration`, `dc2m.acgc1`–`acgc4` |
| Region-level cross-attention | `RegionLevelCrossAttention` |
| CPFC | `ContinuousPredictionSpaceFlowCorrection`, `model.cpfc` |

The PVTv2-B2 backbone produces features with 64, 128, 320, and 512 channels. Multi-scale dilated convolution projects them to 128 channels. The regional pooling grid is 8 × 8, with 128-dimensional attention projections. CPFC uses a 64-dimensional time embedding. The model's default projected Euler schedule is internal to the model and is not exposed by `test.py`.

## Supervision and preprocessing

- Images use OpenCV BGR order and are divided by 255, without additional mean/std normalization.
- Masks are binarized at 0.5 after scaling to [0, 1].
- The total objective includes refined segmentation, direct segmentation, area-adaptive correction-cue supervision, and flow-matching MSE.
- Dynamic soft FP/FN maps are refreshed when validation mIoU improves during epochs 5–100. Initial auxiliary maps must already be available.
- Auxiliary masks use 8-bit storage. Soft error probabilities are quantized when written to disk.
- `SegmentationDataset` returns `(image, (mask, over_mask, under_mask, contour_mask))`. The unused prior-mask input is no longer loaded from disk.

## Implementation details to consider when reproducing the paper

The released implementation preserves the supplied model computation. Shared context is fused after the multi-scale dilated projections. Each cue-disentanglement branch contains two 3 × 3 CBR layers followed by one 1 × 1 CBR layer. Region-level attention upsamples before its bias-free 1 × 1 projection. Empty-mask protection and a cap on inverse-area weights are retained.

These details should be considered when comparing individual equations or architectural descriptions with the executable code. Dataset-specific splits, external reference predictions, and trained weights are not included in this release.

## Checkpoint compatibility

`network/checkpoint.py` maps legacy `ccdm`, `tcm1`–`tcm4`, multi-scale feature prefixes, and `correction_flow_head` names to DC2M/ACGC/CPFC names. Current checkpoints and common checkpoint wrappers are also supported. Historical constructor/forward aliases are retained for older callers.

`--resume` restores model parameters only. It does not continue the complete optimizer/scheduler state of an interrupted run.

## Validation scope

Release checks cover Python syntax, CLI entry points, model forward/backward on synthetic input, legacy checkpoint loading, and synthetic auxiliary-map generation. No new full dataset training run or benchmark evaluation is claimed by this code release.
