from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LatentAlign(nn.Module):
    def __init__(self, in_ch: int, feat_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 1),
            nn.GroupNorm(8, feat_ch),
            nn.SiLU(),
            ConvBlock(feat_ch, feat_ch),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FeatureDecomposer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.base_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.detail_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self.base_branch(x)
        detail = self.detail_refine(x - base)
        return base, detail


class LatentReconstructor(nn.Module):
    def __init__(self, feat_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(feat_ch * 2, feat_ch), nn.Conv2d(feat_ch, out_ch, 1))

    def forward(self, base: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([base, detail], dim=1))


class FusionNetwork(nn.Module):
    # def __init__(self, feat_ch: int, latent_ch: int):
    #     super().__init__()
    #     self.base_gate = nn.Sequential(nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1), nn.SiLU(), nn.Conv2d(feat_ch, feat_ch, 1), nn.Sigmoid())
    #     self.detail_gate = nn.Sequential(nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1), nn.SiLU(), nn.Conv2d(feat_ch, feat_ch, 1), nn.Sigmoid())
    #     self.out_proj = nn.Sequential(ConvBlock(feat_ch * 2, feat_ch), nn.Conv2d(feat_ch, latent_ch, 1))

    # def forward(self, ir_base, ir_detail, vis_base, vis_detail) -> torch.Tensor:
    #     wb = self.base_gate(torch.cat([ir_base, vis_base], dim=1))
    #     wd = self.detail_gate(torch.cat([ir_detail, vis_detail], dim=1))
    #     base_fused = wb * ir_base + (1.0 - wb) * vis_base
    #     detail_fused = wd * ir_detail + (1.0 - wd) * vis_detail
    #     return self.out_proj(torch.cat([base_fused, detail_fused], dim=1))
    """
    保持原类名不变。
    实现一种低成本的 Transformer 风格融合方案：
    将红外模态通过小型 Cross-Attention 聚合到可见光模态中。
    """
    def __init__(self, feat_ch: int, latent_ch: int):
        super().__init__()

        def make_adapter(dim):
            reduction = 4
            return nn.Sequential(
                nn.Conv2d(dim * 2, dim // reduction, 1),
                nn.SiLU(),
                nn.Conv2d(dim // reduction, dim, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(dim, dim, 1)
            )

        # 保持这些新模块的名字与旧 checkpoint 不同
        self.base_adapter = make_adapter(feat_ch)
        self.detail_adapter = make_adapter(feat_ch)
        
        # Channel-wise gate is stronger than a single-channel gate for detail control.
        self.base_gate_new = nn.Sequential(nn.Conv2d(feat_ch * 2, feat_ch, 1), nn.Sigmoid())
        self.detail_gate_new = nn.Sequential(nn.Conv2d(feat_ch * 2, feat_ch, 1), nn.Sigmoid())

        def make_ffn(dim):
            return nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.SiLU(),
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.SiLU(),
                nn.Conv2d(dim, dim, 1),
            )

        self.base_ffn = make_ffn(feat_ch)
        self.detail_ffn = make_ffn(feat_ch)

        self.out_proj = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feat_ch, latent_ch, 1)
        )
        
        nn.init.zeros_(self.base_adapter[-1].weight)
        nn.init.zeros_(self.detail_adapter[-1].weight)

    def forward(self, phi_ib, phi_id, phi_vb, phi_vd):
        cat_b = torch.cat([phi_vb, phi_ib], dim=1)
        cat_d = torch.cat([phi_vd, phi_id], dim=1)
        
        # Use channel-wise modulation and a light FFN refinement for sharper details.
        base_fused = phi_vb + self.base_gate_new(cat_b) * self.base_adapter(cat_b)
        detail_fused = phi_vd + self.detail_gate_new(cat_d) * self.detail_adapter(cat_d)
        base_fused = base_fused + 0.2 * self.base_ffn(base_fused)
        detail_fused = detail_fused + 0.2 * self.detail_ffn(detail_fused)

        decoder_cond = torch.cat([base_fused, detail_fused], dim=1)
        z_fused = self.out_proj(decoder_cond)
        return z_fused, decoder_cond


class MultiScaleDecoderAdapter(nn.Module):
    """Inject fusion guidance into multiple VAE decoder layers.

    The adapter predicts spatial modulation maps per decoder stage.
    This keeps the module lightweight and shape-agnostic across decoder channels.
    """

    def __init__(self, cond_ch: int, num_layers: int = 4):
        super().__init__()
        hidden_ch = max(cond_ch // 2, 16)
        self.attn_pool_size = 16
        self.attn_heads = 4
        self.attn_head_dim = 8
        self.attn_dim = self.attn_heads * self.attn_head_dim
        self.pre = nn.Sequential(
            nn.Conv2d(cond_ch, hidden_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.SiLU(),
        )
        self.alpha_heads = nn.ModuleList([nn.Conv2d(hidden_ch, 1, 1) for _ in range(num_layers)])
        self.beta_heads = nn.ModuleList([nn.Conv2d(hidden_ch, 1, 1) for _ in range(num_layers)])
        self.edge_heads = nn.ModuleList([nn.Conv2d(hidden_ch, 1, 1) for _ in range(num_layers)])
        self.attn_q_heads = nn.ModuleList([nn.Conv2d(2, self.attn_dim, 1) for _ in range(num_layers)])
        self.attn_k_heads = nn.ModuleList([nn.Conv2d(hidden_ch, self.attn_dim, 1) for _ in range(num_layers)])
        self.attn_v_heads = nn.ModuleList([nn.Conv2d(hidden_ch, self.attn_dim, 1) for _ in range(num_layers)])
        self.attn_out_heads = nn.ModuleList([nn.Conv2d(self.attn_dim, 1, 1) for _ in range(num_layers)])
        self.attn_gamma = nn.Parameter(torch.full((num_layers,), 0.05, dtype=torch.float32))
        self._prev_layer_fuse = None

    def _cross_attention(self, layer_idx: int, hidden: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = hidden.shape
        hidden_mean = hidden.mean(dim=1, keepdim=True)
        hidden_std = hidden.std(dim=1, keepdim=True, unbiased=False)
        hidden_stats = torch.cat([hidden_mean, hidden_std], dim=1)

        pooled_h = min(h, self.attn_pool_size)
        pooled_w = min(w, self.attn_pool_size)
        cond_small = F.adaptive_avg_pool2d(cond_feat, (pooled_h, pooled_w))
        hidden_small = F.adaptive_avg_pool2d(hidden_stats, (pooled_h, pooled_w))

        q = self.attn_q_heads[layer_idx](hidden_small)
        k = self.attn_k_heads[layer_idx](cond_small)
        v = self.attn_v_heads[layer_idx](cond_small)

        q = q.view(b, self.attn_heads, self.attn_head_dim, pooled_h * pooled_w).permute(0, 1, 3, 2)  # [B,H,N,D]
        k = k.view(b, self.attn_heads, self.attn_head_dim, pooled_h * pooled_w)  # [B,H,D,M]
        v = v.view(b, self.attn_heads, self.attn_head_dim, pooled_h * pooled_w).permute(0, 1, 3, 2)  # [B,H,M,D]

        attn_logits = torch.matmul(q, k) / (float(self.attn_head_dim) ** 0.5 + 1e-6)  # [B,H,N,M]
        attn = torch.softmax(attn_logits, dim=-1)
        ctx = torch.matmul(attn, v)  # [B,H,N,D]
        ctx = ctx.permute(0, 1, 3, 2).reshape(b, self.attn_dim, pooled_h, pooled_w)
        ctx = F.interpolate(ctx, size=(h, w), mode="bilinear", align_corners=False)
        ctx = self.attn_out_heads[layer_idx](ctx)
        return ctx

    def _build_layer_cond(self, layer_idx: int, hidden: torch.Tensor, cond: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if layer_idx == 0:
            self._prev_layer_fuse = None

        if torch.is_tensor(cond):
            cond_resized = F.interpolate(cond, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
            layer_cond = self.pre(cond_resized)
        else:
            fused_src = cond.get("fused")
            vis_src = cond.get("vis")
            ir_src = cond.get("ir")
            if fused_src is None or vis_src is None or ir_src is None:
                raise RuntimeError("decoder_adapter_cond dict must contain keys: fused, vis, ir")

            vis_feat = self.pre(F.interpolate(vis_src, size=hidden.shape[-2:], mode="bilinear", align_corners=False))
            ir_feat = self.pre(F.interpolate(ir_src, size=hidden.shape[-2:], mode="bilinear", align_corners=False))

            # Progressive cross-layer fusion:
            # Layer 0: use FusionNetwork output as base, combine with vis/ir.
            # Layer 1+: use previous layer's fused output as base, re-inject vis/ir.
            if self._prev_layer_fuse is not None:
                prev_up = F.interpolate(self._prev_layer_fuse, size=hidden.shape[-2:], mode="bilinear", align_corners=False)
                layer_cond = prev_up + 0.6 * vis_feat + 0.2 * ir_feat
            else:
                fused_feat = self.pre(F.interpolate(fused_src, size=hidden.shape[-2:], mode="bilinear", align_corners=False))
                layer_cond = fused_feat + 0.6 * vis_feat + 0.2 * ir_feat

        self._prev_layer_fuse = layer_cond
        return layer_cond

    def apply(self, layer_idx: int, hidden: torch.Tensor, cond: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if layer_idx >= len(self.alpha_heads):
            return hidden
        cond_feat = self._build_layer_cond(layer_idx, hidden, cond)
        alpha = 0.35 * torch.tanh(self.alpha_heads[layer_idx](cond_feat))
        beta = 0.10 * torch.tanh(self.beta_heads[layer_idx](cond_feat))
        edge_w = torch.sigmoid(self.edge_heads[layer_idx](cond_feat))

        # Lightweight spatial cross-attention: decoder features query multimodal condition.
        attn_ctx = self._cross_attention(layer_idx, hidden, cond_feat)
        hidden = hidden + self.attn_gamma[layer_idx] * attn_ctx

        modulated = hidden * (1.0 + alpha) + beta
        hidden_mean = modulated.mean(dim=1, keepdim=True)
        high_pass = hidden_mean - F.avg_pool2d(hidden_mean, kernel_size=3, stride=1, padding=1)
        return modulated + edge_w * high_pass


class DualVAEFusionModel(nn.Module):
    def __init__(self, latent_ch: int, feat_ch: int):
        super().__init__()
        self.align_ir = LatentAlign(latent_ch, feat_ch)
        self.align_vis = LatentAlign(latent_ch, feat_ch)
        self.decomp_ir = FeatureDecomposer(feat_ch)
        self.decomp_vis = FeatureDecomposer(feat_ch)
        self.rec_ir = LatentReconstructor(feat_ch, latent_ch)
        self.rec_vis = LatentReconstructor(feat_ch, latent_ch)
        self.fusion = FusionNetwork(feat_ch, latent_ch)
        self.decoder_adapter = MultiScaleDecoderAdapter(cond_ch=feat_ch * 2, num_layers=4)

    def forward_stage1(self, z_ir, z_vis) -> Dict[str, torch.Tensor]:
        h_ir, h_vis = self.align_ir(z_ir), self.align_vis(z_vis)
        phi_ib, phi_id = self.decomp_ir(h_ir)
        phi_vb, phi_vd = self.decomp_vis(h_vis)
        return {
            "phi_ib": phi_ib,
            "phi_id": phi_id,
            "phi_vb": phi_vb,
            "phi_vd": phi_vd,
            "z_ir_rec": self.rec_ir(phi_ib, phi_id),
            "z_vis_rec": self.rec_vis(phi_vb, phi_vd),
        }

    def forward_stage2(self, z_ir, z_vis) -> Dict[str, torch.Tensor]:
        out = self.forward_stage1(z_ir, z_vis)
        z_fused, fused_decoder_cond = self.fusion(out["phi_ib"], out["phi_id"], out["phi_vb"], out["phi_vd"])
        out["z_fused"] = z_fused
        out["decoder_adapter_cond"] = {
            "fused": fused_decoder_cond,
            "vis": torch.cat([out["phi_vb"], out["phi_vd"]], dim=1),
            "ir": torch.cat([out["phi_ib"], out["phi_id"]], dim=1),
        }
        return out
