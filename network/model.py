import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvtv2 import pvt_v2_b2
from .checkpoint import normalize_state_dict


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(8, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return x * self.sigmoid(y)


class MultiScaleDilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(CBR(in_c, out_c, kernel_size=1, padding=0), ChannelAttention(out_c))
        self.c2 = nn.Sequential(CBR(in_c, out_c, kernel_size=3, padding=6, dilation=6), ChannelAttention(out_c))
        self.c3 = nn.Sequential(CBR(in_c, out_c, kernel_size=3, padding=12, dilation=12), ChannelAttention(out_c))
        self.c4 = nn.Sequential(CBR(in_c, out_c, kernel_size=3, padding=18, dilation=18), ChannelAttention(out_c))

        self.c5 = CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)

        x_multi_scale = torch.cat([x1, x2, x3, x4], dim=1)
        x_multi_scale = self.c5(x_multi_scale)
        x_shortcut = self.c6(x)

        x = self.relu(x_multi_scale + x_shortcut)
        x = self.spatial_attention(x)
        return x


class ContinuousPredictionSpaceFlowCorrection(nn.Module):
    """Continuous Prediction-Space Flow Correction (CPFC), Eqs. (13)-(17)."""

    def __init__(self, in_c, t_dim=64):
        super().__init__()
        self.t_dim = t_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(inplace=True),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(inplace=True)
        )

        self.head = nn.Sequential(
            CBR(in_c + 1 + 1 + 1 + t_dim, 128, kernel_size=3, padding=1),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, output_feat, intermediate_mask, correction_time, direct_mask):
        batch_size, _, height, width = intermediate_mask.shape

        if output_feat.shape[-2:] != (height, width):
            output_feat = F.interpolate(output_feat, size=(height, width), mode="bilinear", align_corners=False)

        if direct_mask.shape[-2:] != (height, width):
            direct_mask = F.interpolate(direct_mask, size=(height, width), mode="bilinear", align_corners=False)

        correction_time = correction_time.view(batch_size, 1, 1, 1).to(
            device=intermediate_mask.device,
            dtype=intermediate_mask.dtype
        )

        time_scalar_map = correction_time.expand(batch_size, 1, height, width)
        time_vec = self.time_embed(correction_time.view(batch_size, 1))
        time_vec_map = time_vec[:, :, None, None].expand(batch_size, self.t_dim, height, width)

        x = torch.cat(
            [output_feat, intermediate_mask, direct_mask, time_scalar_map, time_vec_map],
            dim=1
        )
        return self.head(x)


    def refine(self, output_feat, direct_mask, correction_steps, refine_noise_std=0.0):
        refined_mask = direct_mask.detach().clone()
        batch_size = refined_mask.size(0)

        correction_steps = max(int(correction_steps), 1)
        dt = 1.0 / correction_steps

        for i in range(correction_steps):
            correction_time = torch.full(
                (batch_size, 1, 1, 1),
                float(i) / float(correction_steps),
                device=refined_mask.device,
                dtype=refined_mask.dtype
            )

            correction_field = self(
                output_feat,
                refined_mask,
                correction_time,
                direct_mask
            )

            refined_mask = refined_mask + dt * correction_field

            if refine_noise_std > 0.0:
                refined_mask = refined_mask + torch.randn_like(refined_mask) * refine_noise_std

            refined_mask = refined_mask.clamp(1e-6, 1.0 - 1e-6)

        return refined_mask

class RegionLevelCrossAttention(nn.Module):
    """Region-level interaction between local features and correction cues."""

    def __init__(self, channels, r_h=8, r_w=8, d=None, alpha_init=0.1):
        super().__init__()

        if d is None:
            d = channels

        self.pool = nn.AdaptiveAvgPool2d((r_h, r_w))
        self.q = nn.Conv2d(channels, d, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, d, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, d, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(d, channels, kernel_size=1, bias=False)

        self.scale = d ** -0.5
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, local_feat, correction_cue_feat):
        batch_size, _, height, width = local_feat.shape

        local_pool = self.pool(local_feat)
        cue_pool = self.pool(correction_cue_feat)

        q = self.q(local_pool).flatten(2).transpose(1, 2)
        k = self.k(cue_pool).flatten(2)
        v = self.v(cue_pool).flatten(2).transpose(1, 2)

        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        y = attn @ v

        pooled_h, pooled_w = self.pool.output_size
        y = y.transpose(1, 2).reshape(batch_size, -1, pooled_h, pooled_w)
        y = F.interpolate(y, size=(height, width), mode="bilinear", align_corners=False)
        y = self.proj(y)

        return local_feat + self.alpha * y


class AdaptiveCueGuidedCalibration(nn.Module):
 

    def __init__(self, channels=128, guide_channels=128, r_h=8, r_w=8, ratio=16):
        super().__init__()

        hidden = max(8, channels // ratio)

        self.local_project = nn.Sequential(
            CBR(channels, channels, kernel_size=3, padding=1),
            CBR(channels, channels, kernel_size=3, padding=1)
        )

        self.over_project = CBR(guide_channels, channels, kernel_size=1, padding=0)
        self.under_project = CBR(guide_channels, channels, kernel_size=1, padding=0)
        self.contour_project = CBR(guide_channels, channels, kernel_size=1, padding=0)

        self.tri_cue_router = nn.Sequential(
            CBR(channels * 3, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, 3, kernel_size=1, padding=0)
        )

        self.tri_cue_fuse = nn.Sequential(
            CBR(channels, channels, kernel_size=3, padding=1),
            CBR(channels, channels, kernel_size=3, padding=1)
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.beta_s = nn.Parameter(torch.tensor(0.1))
        self.beta_c = nn.Parameter(torch.tensor(0.1))
        self.beta_cue = nn.Parameter(torch.tensor(0.2))

        self.region_cue_interaction = RegionLevelCrossAttention(
            channels=channels,
            r_h=r_h,
            r_w=r_w,
            d=channels,
            alpha_init=0.1
        )

        self.out_project = CBR(channels, channels, kernel_size=3, padding=1)

    def forward(self, local_feat, f_over, f_under, f_contour):
        x = self.local_project(local_feat)
        target_size = x.shape[-2:]

        f_over = F.interpolate(f_over, size=target_size, mode="bilinear", align_corners=False)
        f_under = F.interpolate(f_under, size=target_size, mode="bilinear", align_corners=False)
        f_contour = F.interpolate(f_contour, size=target_size, mode="bilinear", align_corners=False)

        f_over = self.over_project(f_over)
        f_under = self.under_project(f_under)
        f_contour = self.contour_project(f_contour)

        tri_cue_logits = self.tri_cue_router(torch.cat([f_over, f_under, f_contour], dim=1))
        tri_cue_weight = torch.softmax(tri_cue_logits, dim=1)

        w_over = tri_cue_weight[:, 0:1, :, :]
        w_under = tri_cue_weight[:, 1:2, :, :]
        w_contour = tri_cue_weight[:, 2:3, :, :]

        correction_cue_feat = w_over * f_over + w_under * f_under + w_contour * f_contour
        correction_cue_feat = self.tri_cue_fuse(correction_cue_feat)

        spatial_gate = self.spatial_gate(correction_cue_feat)
        channel_gate = self.channel_gate(correction_cue_feat)

        x = x * (1.0 + self.beta_s * spatial_gate)
        x = x * (1.0 + self.beta_c * channel_gate)
        x = x + self.beta_cue * correction_cue_feat

        x = self.region_cue_interaction(x, correction_cue_feat)
        x = self.out_project(x)

        aux = {
            "correction_spatial_gate": spatial_gate,
            "correction_channel_gate": channel_gate,
            "tri_cue_weight": tri_cue_weight,
            "correction_cue_feat": correction_cue_feat
        }

        return x, aux


class DirectionAwareCueDisentanglement(nn.Module):
    """Direction-aware cue disentanglement in DC2M, Eq. (1)."""

    def __init__(self, in_c=128, out_c=128):
        super().__init__()

        def make_branch():
            return nn.Sequential(
                CBR(in_c, 512, kernel_size=3, padding=1),
                CBR(512, out_c, kernel_size=3, padding=1),
                CBR(out_c, out_c, kernel_size=1, padding=0)
            )

        self.over_segmentation_branch = make_branch()
        self.under_segmentation_branch = make_branch()
        self.contour_refinement_branch = make_branch()

    def forward(self, x):
        f_over = self.over_segmentation_branch(x)
        f_under = self.under_segmentation_branch(x)
        f_contour = self.contour_refinement_branch(x)
        return f_over, f_under, f_contour


class CorrectionCueAuxiliaryHead(nn.Module):
   

    def __init__(self, in_c):
        super().__init__()

        def make_branch():
            return nn.Sequential(
                CBR(in_c, 256, kernel_size=3, padding=1),
                CBR(256, 128, kernel_size=3, padding=1),
                CBR(128, 64, kernel_size=3, padding=1),
                nn.Conv2d(64, 1, kernel_size=1, padding=0)
            )

        self.over_segmentation_head = make_branch()
        self.under_segmentation_head = make_branch()
        self.contour_refinement_head = make_branch()

    def forward(self, f_over, f_under, f_contour, out_size=None):
        over_seg_pred = torch.sigmoid(self.over_segmentation_head(f_over))
        under_seg_pred = torch.sigmoid(self.under_segmentation_head(f_under))
        contour_pred = torch.sigmoid(self.contour_refinement_head(f_contour))

        if out_size is not None:
            over_seg_pred = F.interpolate(over_seg_pred, size=out_size, mode="bilinear", align_corners=False)
            under_seg_pred = F.interpolate(under_seg_pred, size=out_size, mode="bilinear", align_corners=False)
            contour_pred = F.interpolate(contour_pred, size=out_size, mode="bilinear", align_corners=False)

        return over_seg_pred, under_seg_pred, contour_pred


class MultiScaleFusion(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.fuse = nn.Sequential(
            CBR(channels * 4, channels, kernel_size=3, padding=1),
            CBR(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, d1, d2, d3, d4):
        target_size = d1.shape[-2:]

        x = torch.cat(
            [
                d1,
                F.interpolate(d2, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(d3, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(d4, size=target_size, mode="bilinear", align_corners=False)
            ],
            dim=1
        )

        return self.fuse(x)


class CorrectionAwareDecoder(nn.Module):
    def __init__(self, channels=128, out_c=1):
        super().__init__()

        self.decode = nn.Sequential(
            CBR(channels, channels, kernel_size=3, padding=1),
            CBR(channels, channels, kernel_size=3, padding=1),
            CBR(channels, channels, kernel_size=3, padding=1)
        )

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

        self.head = nn.Sequential(
            CBR(channels, 128, kernel_size=3, padding=1),
            CBR(128, 64, kernel_size=1, padding=0),
            nn.Conv2d(64, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, out_size):
        residual = x

        x = self.decode(x)
        x = x + residual

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        output_feat = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        direct_mask = self.head(output_feat)

        return direct_mask, output_feat


class DecoupleThenCalibrateFeatureCorrection(nn.Module):


    def __init__(self):
        super().__init__()

        self.dconv1 = MultiScaleDilatedConv(64, 128)
        self.dconv2 = MultiScaleDilatedConv(128, 128)
        self.dconv3 = MultiScaleDilatedConv(320, 128)
        self.dconv4 = MultiScaleDilatedConv(512, 128)

        self.ms_fuse = MultiScaleFusion(128)
        self.ms_fuse_refined = MultiScaleFusion(128)

        self.cue_disentanglement = DirectionAwareCueDisentanglement(128, 128)
        self.cue_aux_head = CorrectionCueAuxiliaryHead(128)

        self.acgc4 = AdaptiveCueGuidedCalibration(128, guide_channels=128)
        self.acgc3 = AdaptiveCueGuidedCalibration(128, guide_channels=128)
        self.acgc2 = AdaptiveCueGuidedCalibration(128, guide_channels=128)
        self.acgc1 = AdaptiveCueGuidedCalibration(128, guide_channels=128)

    def forward(self, x1, x2, x3, x4, out_size):
        d1 = self.dconv1(x1)
        d2 = self.dconv2(x2)
        d3 = self.dconv3(x3)
        d4 = self.dconv4(x4)

        fused = self.ms_fuse(d1, d2, d3, d4)
        f_over, f_under, f_contour = self.cue_disentanglement(fused)

        over_seg_pred, under_seg_pred, contour_pred = self.cue_aux_head(
            f_over,
            f_under,
            f_contour,
            out_size=out_size
        )

        f4, acgc_aux4 = self.acgc4(d4, f_over, f_under, f_contour)
        f3, acgc_aux3 = self.acgc3(d3, f_over, f_under, f_contour)
        f2, acgc_aux2 = self.acgc2(d2, f_over, f_under, f_contour)
        f1, acgc_aux1 = self.acgc1(d1, f_over, f_under, f_contour)

        fused_refined = self.ms_fuse_refined(f1, f2, f3, f4)

        return fused_refined, over_seg_pred, under_seg_pred, contour_pred, acgc_aux1


class CorrFlowNet(nn.Module):

    def __init__(
        self,
        correction_alpha=1.0,
        correction_steps=4,
        flow_alpha=None,
        fm_steps=None,
        use_mamba=False,
        backbone_weights=None
    ):
        super().__init__()

        if flow_alpha is not None:
            correction_alpha = flow_alpha

        if fm_steps is not None:
            correction_steps = fm_steps

        self.use_mamba = use_mamba
        self.correction_alpha = correction_alpha
        self.correction_steps = correction_steps

        self.backbone = pvt_v2_b2()

        if backbone_weights is not None:
            checkpoint = torch.load(backbone_weights, map_location="cpu")
            weights = normalize_state_dict(checkpoint)
            model_dict = self.backbone.state_dict()
            matched = {
                k.removeprefix("backbone."): v for k, v in weights.items()
                if k.removeprefix("backbone.") in model_dict
                and model_dict[k.removeprefix("backbone.")].shape == v.shape
            }
            if not matched:
                raise ValueError("No matching PVTv2-B2 tensors in backbone_weights.")
            self.backbone.load_state_dict(matched, strict=False)

        self.dc2m = DecoupleThenCalibrateFeatureCorrection()

        self.correction_aware_decoder = CorrectionAwareDecoder(128, 1)
        self.cpfc = ContinuousPredictionSpaceFlowCorrection(128)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Accept current names and legacy CorrFlow checkpoints."""
        return super().load_state_dict(normalize_state_dict(state_dict), strict=strict, **kwargs)

    def predict_correction_field(self, output_feat, intermediate_mask, correction_time, direct_mask):
        return self.cpfc(output_feat, intermediate_mask, correction_time, direct_mask)

    def progressive_refine(self, output_feat, direct_mask, correction_steps=None, refine_noise_std=0.0):
        if correction_steps is None:
            correction_steps = self.correction_steps
        return self.cpfc.refine(output_feat, direct_mask, correction_steps, refine_noise_std)

    @staticmethod
    def _make_correction_time(batch_size, device, dtype, correction_time=None, training=True):
        if correction_time is None:
            if training:
                return torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype)

            return torch.full(
                (batch_size, 1, 1, 1),
                0.5,
                device=device,
                dtype=dtype
            )

        if isinstance(correction_time, (float, int)):
            return torch.full(
                (batch_size, 1, 1, 1),
                float(correction_time),
                device=device,
                dtype=dtype
            )

        correction_time = correction_time.to(device=device, dtype=dtype)
        return correction_time.view(batch_size, 1, 1, 1)

    def forward(
        self,
        image,
        prior_mask=None,
        gt_mask=None,
        correction_steps=None,
        correction_time=None,
        fm_steps=None,
        flow_t=None,
        refine_noise_std=0.0
    ):
        if correction_steps is None and fm_steps is not None:
            correction_steps = fm_steps

        if correction_time is None and flow_t is not None:
            correction_time = flow_t

        # prior_mask is intentionally ignored.
        out_size = image.shape[-2:]

        # PVT encoder
        pvt = self.backbone(image)
        x1, x2, x3, x4 = pvt[0], pvt[1], pvt[2], pvt[3]

        fused_refined, over_seg_pred, under_seg_pred, contour_pred, acgc_aux1 = self.dc2m(
            x1, x2, x3, x4, out_size
        )

        direct_mask, output_feat = self.correction_aware_decoder(
            fused_refined,
            out_size=out_size
        )

        correction_field_pred = None
        correction_field_target = None
        intermediate_mask = None
        correction_time_used = None

        if gt_mask is not None:
            gt_mask = gt_mask.float()

            if gt_mask.shape[-2:] != direct_mask.shape[-2:]:
                gt_mask = F.interpolate(gt_mask, size=direct_mask.shape[-2:], mode="nearest")

            source_mask = direct_mask.detach()
            target_mask = gt_mask.clamp(0.0, 1.0)

            batch_size = target_mask.size(0)

            correction_time_used = self._make_correction_time(
                batch_size=batch_size,
                device=target_mask.device,
                dtype=target_mask.dtype,
                correction_time=correction_time,
                training=self.training
            )

            intermediate_mask = (1.0 - correction_time_used) * source_mask + correction_time_used * target_mask
            correction_field_target = target_mask - source_mask

            correction_field_pred = self.cpfc(
                output_feat,
                intermediate_mask,
                correction_time_used,
                direct_mask
            )

        refined_mask = self.progressive_refine(
            output_feat=output_feat,
            direct_mask=direct_mask,
            correction_steps=correction_steps,
            refine_noise_std=refine_noise_std
        )

        final_mask = refined_mask.clamp(1e-6, 1.0 - 1e-6)

        correction_spatial_gate = F.interpolate(
            acgc_aux1["correction_spatial_gate"],
            size=final_mask.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        tri_cue_weight = F.interpolate(
            acgc_aux1["tri_cue_weight"],
            size=final_mask.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        aux = {
            "correction_spatial_gate": correction_spatial_gate,
            "tri_cue_weight": tri_cue_weight,
            "tri_cue_over": tri_cue_weight[:, 0:1, :, :],
            "tri_cue_under": tri_cue_weight[:, 1:2, :, :],
            "tri_cue_contour": tri_cue_weight[:, 2:3, :, :],
            "contour_guide": correction_spatial_gate,
            "direct_mask": direct_mask,
            "refined_mask": refined_mask,
            "correction_field_pred": correction_field_pred if correction_field_pred is not None else refined_mask - direct_mask,
            "correction_field_target": correction_field_target,
            "intermediate_mask": intermediate_mask,
            "correction_time": correction_time_used,
            "over_seg_pred": over_seg_pred,
            "under_seg_pred": under_seg_pred
        }

        return final_mask, contour_pred, aux



CoReFlowNet = CorrFlowNet
