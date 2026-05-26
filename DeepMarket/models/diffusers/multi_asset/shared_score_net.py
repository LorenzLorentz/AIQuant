"""SharedScoreNet — wraps a single TRADES backbone so it can be called on a
batch of N assets with shared parameters plus an additive asset embedding.

The wrapper is intentionally thin: it adds an ``nn.Embedding(N, F_in)``
contribution to the first ``cond_orders`` token of each asset, flattens
``(B, N, ...)`` into ``(B*N, ...)`` for the TRADES call, then restores the
leading asset axis. P2 optionally adds a spread-conditioning embedding to the
same first conditioning token. The TRADES backbone itself is unchanged.

Tensors fed in here are already post-augmentation — same dtype/shape that
the existing single-asset ``GaussianDiffusion.ddpm_single_step`` hands to
``self.NN``.
"""

import torch
from torch import nn

import constants as cst
from models.diffusers.TRADES.TRADES import TRADES


class SharedScoreNet(nn.Module):
    """Multi-asset wrapper around a shared TRADES backbone.

    Parameters mirror ``GaussianDiffusion``'s construction of ``self.NN``.
    """

    def __init__(
        self,
        num_assets: int,
        input_size: int,
        cond_seq_len: int,
        num_diffusionsteps: int,
        depth: int,
        num_heads: int,
        gen_sequence_size: int,
        cond_dropout_prob: float,
        is_augmented: bool,
        dropout: float,
        cond_type: str,
        cond_method: str,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.input_size = input_size

        self.backbone = TRADES(
            input_size=input_size,
            cond_seq_len=cond_seq_len,
            num_diffusionsteps=num_diffusionsteps,
            depth=depth,
            num_heads=num_heads,
            gen_sequence_size=gen_sequence_size,
            cond_dropout_prob=cond_dropout_prob,
            is_augmented=is_augmented,
            dropout=dropout,
            cond_type=cond_type,
            cond_method=cond_method,
        )

        # Additive asset embedding, applied to the first cond_orders token.
        # Initialized small so P0 behaves like vanilla TRADES at step 0.
        self.asset_embedding = nn.Embedding(num_assets, input_size)
        nn.init.normal_(self.asset_embedding.weight, mean=0.0, std=0.02)

        self.spread_embedding = nn.Sequential(
            nn.Linear(1, input_size),
            nn.SiLU(),
            nn.Linear(input_size, input_size, bias=False),
        )
        nn.init.normal_(self.spread_embedding[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.spread_embedding[0].bias)
        nn.init.normal_(self.spread_embedding[2].weight, mean=0.0, std=0.02)

    def forward(self, x_t, cond_orders, t, cond_lob, asset_ids, spread_cond=None):
        """
        Parameters
        ----------
        x_t           : (B, N, K_gen, F_in)
        cond_orders   : (B, N, K_cond, F_in)
        t             : (B,) int64 -- shared diffusion step across assets
        cond_lob      : (B, N, K_cond+1, 40) or None
        asset_ids     : (N,) int64
        spread_cond   : optional (B, N) scalar spread signal per asset

        Returns
        -------
        noise, var    : both (B, N, K_gen, F_out)
        """
        if x_t.dim() != 4:
            raise ValueError(f"x_t must be (B, N, K_gen, F), got {tuple(x_t.shape)}")
        B, N, K_gen, F_in = x_t.shape
        if N != self.num_assets:
            raise ValueError(
                f"asset axis mismatch: expected N={self.num_assets}, got {N}"
            )
        K_cond = cond_orders.shape[2]

        # (N, F_in) asset embeddings, broadcast onto the first cond token.
        asset_vec = self.asset_embedding(asset_ids.to(x_t.device))  # (N, F_in)
        # Build a (B, N, K_cond, F_in) additive mask with the embedding only
        # on the first token position. (Adding to position 0 mirrors how
        # TRADES injects positional / timestep embeddings.)
        emb_mask = torch.zeros((B, N, K_cond, F_in), device=x_t.device, dtype=x_t.dtype)
        emb_mask[:, :, 0, :] = asset_vec.to(dtype=x_t.dtype).unsqueeze(0)
        if spread_cond is not None:
            if spread_cond.shape != (B, N):
                raise ValueError(
                    "spread_cond must be (B, N) when provided, "
                    f"got {tuple(spread_cond.shape)} for {(B, N)}"
                )
            spread_in = spread_cond.to(device=x_t.device, dtype=x_t.dtype).reshape(B * N, 1)
            spread_vec = self.spread_embedding(spread_in).reshape(B, N, F_in)
            emb_mask[:, :, 0, :] = emb_mask[:, :, 0, :] + spread_vec
        cond_orders = cond_orders + emb_mask

        # Flatten the asset axis into the batch axis so TRADES sees plain
        # (B*N, K, F) tensors.
        x_t_flat = x_t.reshape(B * N, K_gen, F_in)
        cond_flat = cond_orders.reshape(B * N, K_cond, F_in)
        # Broadcast diffusion timesteps across assets.
        t_flat = t.repeat_interleave(N) if t.shape[0] == B else t
        cond_lob_flat = None
        if cond_lob is not None:
            K_lob = cond_lob.shape[2]
            F_lob = cond_lob.shape[3]
            cond_lob_flat = cond_lob.reshape(B * N, K_lob, F_lob)

        noise_flat, var_flat = self.backbone(x_t_flat, cond_flat, t_flat, cond_lob_flat)
        # backbone output: (B*N, K_gen, F_out); unflatten asset axis.
        F_out = noise_flat.shape[-1]
        noise = noise_flat.reshape(B, N, K_gen, F_out)
        var = var_flat.reshape(B, N, K_gen, F_out)
        return noise, var
