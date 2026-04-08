"""
Latent-space denoising network for diffusion-based IR super-resolution.

Takes noisy latent z_t concatenated with encoded LQ z_y as input,
conditioned on timestep t, and predicts x_0 (or noise epsilon).

Architecture: lightweight UNet with timestep embedding and gradient guidance,
designed to operate in the 4-channel latent space of AutoencoderKL.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embeddings [N] -> [N, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepResBlock(nn.Module):
    """Residual block with FiLM-style timestep modulation."""

    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # scale + shift from timestep embedding
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, channels * 2),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # timestep modulation: scale-shift
        emb_out = self.emb_proj(emb)[:, :, None, None]
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class GradientGuidanceModule(nn.Module):
    """Inject LQ image gradient information into latent features."""

    def __init__(self, feat_channels: int, grad_channels: int = 2):
        super().__init__()
        self.grad_proj = nn.Conv2d(grad_channels, feat_channels, 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, 1),
            nn.SiLU(),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor, lr_grad: torch.Tensor) -> torch.Tensor:
        grad_feat = self.grad_proj(
            F.interpolate(lr_grad, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        )
        return feat + self.fuse(torch.cat([feat, grad_feat], dim=1))


# ---------------------------------------------------------------------------
# Latent Denoising UNet
# ---------------------------------------------------------------------------

class LatentDenoisingUNet(nn.Module):
    """
    Lightweight UNet for denoising in VAE latent space.

    Input:  z_t concatenated with z_y  (in_channels = latent_ch * 2)
    Output: predicted x_0 (or noise)  (out_channels = latent_ch)

    Supports:
      - Timestep conditioning via sinusoidal embedding + FiLM
      - LQ gradient guidance injection
      - Multi-scale encoder-decoder with skip connections
    """

    def __init__(
        self,
        latent_channels: int = 4,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        cond_lq: bool = True,
        use_gradient_guidance: bool = True,
    ):
        super().__init__()
        self.cond_lq = cond_lq
        in_ch = latent_channels * 2 if cond_lq else latent_channels
        out_ch = latent_channels
        emb_ch = model_channels * 4

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        # Gradient guidance (operates on original LQ image gradients)
        self.use_gradient_guidance = use_gradient_guidance
        if use_gradient_guidance:
            self.grad_guide = GradientGuidanceModule(model_channels * channel_mult[0])

        # --- Encoder ---
        self.input_conv = nn.Conv2d(in_ch, model_channels, 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = model_channels
        enc_channels = [ch]
        for level, mult in enumerate(channel_mult):
            out = model_channels * mult
            for _ in range(num_res_blocks):
                self.enc_blocks.append(TimestepResBlock(ch, emb_ch, dropout))
                if ch != out:
                    # channel projection within block
                    self.enc_blocks.append(None)  # placeholder
                ch = out
                enc_channels.append(ch)
            if level < len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                enc_channels.append(ch)

        # Rebuild encoder properly: list of (resblock, optional_proj) per block
        # Let me redo this cleanly:
        self.enc_blocks = nn.ModuleList()
        self.enc_projs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = model_channels
        enc_channels = [ch]

        for level, mult in enumerate(channel_mult):
            out = model_channels * mult
            for _ in range(num_res_blocks):
                self.enc_blocks.append(TimestepResBlock(ch, emb_ch, dropout))
                if ch != out:
                    self.enc_projs.append(nn.Conv2d(ch, out, 1))
                else:
                    self.enc_projs.append(nn.Identity())
                ch = out
                enc_channels.append(ch)
            if level < len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                enc_channels.append(ch)

        # --- Middle ---
        self.mid_block1 = TimestepResBlock(ch, emb_ch, dropout)
        self.mid_block2 = TimestepResBlock(ch, emb_ch, dropout)

        # --- Decoder ---
        self.dec_blocks = nn.ModuleList()
        self.dec_projs = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out = model_channels * mult
            for i in range(num_res_blocks + 1):
                # skip connection doubles channels
                skip_ch = enc_channels.pop()
                block_in = ch + skip_ch
                self.dec_blocks.append(TimestepResBlock(block_in, emb_ch, dropout))
                if block_in != out:
                    self.dec_projs.append(nn.Conv2d(block_in, out, 1))
                else:
                    self.dec_projs.append(nn.Identity())
                ch = out
            if level > 0:
                self.upsamples.append(Upsample(ch))

        # --- Output ---
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)

    @staticmethod
    def _image_gradient(x: torch.Tensor) -> torch.Tensor:
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return torch.cat([gx, gy], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, lq: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: noisy latent z_t (already normalized by _scale_input), [B, C_lat, H, W]
            t: timestep indices [B]
            lq: low-quality image in original pixel space [B, C_img, H_lq, W_lq]
                Used for: (1) encoding to z_y for concatenation, (2) gradient guidance.
                Note: z_y concatenation is handled OUTSIDE this network by the pipeline;
                      x already contains [z_t, z_y] if cond_lq=True.
                      lq here is only for gradient guidance.
        Returns:
            prediction [B, C_lat, H, W]
        """
        # Timestep embedding
        emb = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))

        # Input
        h = self.input_conv(x)

        # Gradient guidance injection at the first level
        if self.use_gradient_guidance and lq is not None:
            lr_grad = self._image_gradient(lq)
            h = self.grad_guide(h, lr_grad)

        # Encoder
        hs = [h]
        block_idx = 0
        ds_idx = 0
        for level, mult in enumerate(self._channel_mult()):
            for _ in range(self._num_res_blocks()):
                h = self.enc_blocks[block_idx](h, emb)
                h = self.enc_projs[block_idx](h)
                block_idx += 1
                hs.append(h)
            if level < len(self._channel_mult()) - 1:
                h = self.downsamples[ds_idx](h)
                ds_idx += 1
                hs.append(h)

        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)

        # Decoder
        block_idx = 0
        us_idx = 0
        for level, mult in reversed(list(enumerate(self._channel_mult()))):
            for i in range(self._num_res_blocks() + 1):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.dec_blocks[block_idx](h, emb)
                h = self.dec_projs[block_idx](h)
                block_idx += 1
            if level > 0:
                h = self.upsamples[us_idx](h)
                us_idx += 1

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)

    def _channel_mult(self):
        # Reconstruct from module counts
        # Store as attribute during init instead
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Cleaner implementation: LatentDenoiser
# ---------------------------------------------------------------------------

class LatentDenoiser(nn.Module):
    """
    Clean latent-space denoiser for IR diffusion SR.

    Follows DifIISR's UNet pattern:
      - Input: [z_t; z_y] concatenated along channel dim (if cond_lq=True)
      - Timestep conditioning via FiLM
      - Encoder -> Middle -> Decoder with skip connections
      - Output: predicted x_0 in latent space
    """

    def __init__(
        self,
        latent_channels: int = 4,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        cond_lq: bool = True,
        use_gradient_guidance: bool = True,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.cond_lq = cond_lq
        self.channel_mult = channel_mult
        self.num_res_blocks_per_level = num_res_blocks

        in_ch = latent_channels * 2 if cond_lq else latent_channels
        out_ch = latent_channels
        emb_ch = model_channels * 4

        # --- Timestep embedding ---
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        # --- Gradient guidance ---
        self.use_gradient_guidance = use_gradient_guidance
        if use_gradient_guidance:
            self.grad_guide = GradientGuidanceModule(model_channels)

        # --- Build encoder ---
        self.input_conv = nn.Conv2d(in_ch, model_channels, 3, padding=1)

        enc_blocks = []
        enc_ch_list = [model_channels]  # track channels for skip connections
        ch = model_channels

        for level, mult in enumerate(channel_mult):
            target_ch = model_channels * mult
            for _ in range(num_res_blocks):
                enc_blocks.append(self._make_res_block(ch, target_ch, emb_ch, dropout))
                ch = target_ch
                enc_ch_list.append(ch)
            if level < len(channel_mult) - 1:
                enc_blocks.append(Downsample(ch))
                enc_ch_list.append(ch)

        self.enc_blocks = nn.ModuleList(enc_blocks)

        # --- Middle ---
        self.mid1 = self._make_res_block(ch, ch, emb_ch, dropout)
        self.mid2 = self._make_res_block(ch, ch, emb_ch, dropout)

        # --- Build decoder ---
        dec_blocks = []
        for level in reversed(range(len(channel_mult))):
            target_ch = model_channels * channel_mult[level]
            for i in range(num_res_blocks + 1):
                skip_ch = enc_ch_list.pop()
                dec_blocks.append(self._make_res_block(ch + skip_ch, target_ch, emb_ch, dropout))
                ch = target_ch
            if level > 0:
                dec_blocks.append(Upsample(ch))

        self.dec_blocks = nn.ModuleList(dec_blocks)

        # --- Output ---
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)

    @staticmethod
    def _make_res_block(in_ch, out_ch, emb_ch, dropout):
        """Create a residual block, optionally with channel projection."""
        block = TimestepResBlock(in_ch, emb_ch, dropout)
        if in_ch != out_ch:
            return ResBlockWithProj(block, nn.Conv2d(in_ch, out_ch, 1))
        return block

    @staticmethod
    def _image_gradient(x: torch.Tensor) -> torch.Tensor:
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return torch.cat([gx, gy], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, lq: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [z_t] or [z_t; z_y] if cond_lq, shape [B, C, H, W]
               Already normalized by GaussianDiffusion._scale_input.
            t: timestep indices [B]
            lq: original LQ image [B, C_img, H_lq, W_lq] for gradient guidance only.
        """
        emb = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))

        h = self.input_conv(x)

        if self.use_gradient_guidance and lq is not None:
            lr_grad = self._image_gradient(lq)
            h = self.grad_guide(h, lr_grad)

        # Encoder
        hs = [h]
        for block in self.enc_blocks:
            if isinstance(block, Downsample):
                h = block(h)
                hs.append(h)
            else:
                h = block(h, emb)
                hs.append(h)

        # Middle
        h = self.mid1(h, emb)
        h = self.mid2(h, emb)

        # Decoder
        for block in self.dec_blocks:
            if isinstance(block, Upsample):
                h = block(h)
            else:
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, emb)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)


class ResBlockWithProj(nn.Module):
    """Wrapper: residual block followed by 1x1 channel projection."""

    def __init__(self, block: TimestepResBlock, proj: nn.Conv2d):
        super().__init__()
        self.block = block
        self.proj = proj

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.proj(self.block(x, emb))
