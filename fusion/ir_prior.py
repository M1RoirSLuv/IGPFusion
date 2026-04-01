"""IR-aware UNet feature extractor for semantic prior constraint.

Extracts multi-scale encoder features from a Stable Diffusion UNet that was
fine-tuned on infrared images.  During fusion training the prior loss encourages
the fused latent to preserve IR semantic structure as understood by this UNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel


class IRPriorExtractor(nn.Module):
    """Frozen SD UNet encoder used as an IR feature prior.

    Only the encoder half (conv_in → down_blocks → mid_block) is executed,
    saving ~50 % compute compared to a full UNet forward pass.
    """

    def __init__(self, unet_path: str, feature_layers: tuple = (1, 2)):
        """
        Args:
            unet_path: Path to the UNet directory (contains config.json + weights).
            feature_layers: Indices of down_blocks whose outputs are collected.
                            Default (1, 2) → 640-ch @ 32×32 and 1280-ch @ 16×16.
        """
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            unet_path, torch_dtype=torch.float32, local_files_only=True,
        )
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False
        self.feature_layers = feature_layers
        # SD1.5 text encoder output shape: [B, 77, 768].
        # A zero embedding acts as unconditional (null-text) guidance.
        self.register_buffer("null_text_emb", torch.zeros(1, 77, 768))

    # ------------------------------------------------------------------
    # Encoder-only forward
    # ------------------------------------------------------------------
    def _encode(self, z: torch.Tensor) -> list:
        """Run the UNet encoder and return intermediate feature maps.

        Args:
            z: Latent tensor **already scaled** by the VAE scaling factor,
               shape [B, 4, H, W].  This matches the convention used during
               diffusion training (latents are stored in scaled form).

        Returns:
            List of feature tensors from the selected ``down_blocks``.
        """
        b = z.shape[0]
        timestep = torch.zeros(b, dtype=torch.long, device=z.device)
        enc_hid = self.null_text_emb.expand(b, -1, -1).to(dtype=z.dtype)

        # Time embedding
        t_emb = self.unet.time_proj(timestep).to(dtype=z.dtype)
        emb = self.unet.time_embedding(t_emb)

        # Spatial entry
        sample = self.unet.conv_in(z)

        # Down blocks (encoder path)
        features = []
        for i, block in enumerate(self.unet.down_blocks):
            has_xattn = getattr(block, "has_cross_attention", False)
            if has_xattn:
                sample, _res = block(
                    hidden_states=sample, temb=emb,
                    encoder_hidden_states=enc_hid,
                )
            else:
                sample, _res = block(hidden_states=sample, temb=emb)
            if i in self.feature_layers:
                features.append(sample)

        return features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_reference(self, z: torch.Tensor) -> list:
        """Extract features for the **reference** IR latent (no gradient)."""
        return self._encode(z)

    def extract_fused(self, z: torch.Tensor) -> list:
        """Extract features for the **fused** latent (gradient flows back to *z*)."""
        return self._encode(z)


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def ir_prior_feature_loss(fused_feats: list, ir_feats: list) -> torch.Tensor:
    """Compute L2 loss on L2-normalised encoder features.

    Normalising first makes the loss scale-invariant: we penalise
    *directional* differences in the feature space rather than raw
    magnitude, so a mild weight (0.05–0.2) is usually sufficient.
    """
    loss = torch.tensor(0.0, device=fused_feats[0].device, dtype=fused_feats[0].dtype)
    for ff, irf in zip(fused_feats, ir_feats):
        ff_n = F.normalize(ff.flatten(2), dim=-1)
        irf_n = F.normalize(irf.flatten(2), dim=-1)
        loss = loss + F.mse_loss(ff_n, irf_n)
    return loss / max(len(fused_feats), 1)
