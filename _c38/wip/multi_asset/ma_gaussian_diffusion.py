"""MultiAssetGaussianDiffusion — N-asset variant of
:class:`models.diffusers.gaussian_diffusion.GaussianDiffusion`.

All tensors carry a leading ``(B, N, ...)`` shape. P0 keeps the per-asset
denoising fully independent: ``fuse`` is identity and the two hooks
``pre_fusion_hook`` / ``post_fusion_hook`` are no-ops. P1 replaces ``fuse``
with a graph stack; P2 / P3 plug into the hooks.

Loss policy: each asset contributes a hybrid (L_simple + lambda * L_vlb)
loss; the per-asset scalars are *summed* across assets so the batch-level
loss has shape ``(B,)`` and slots straight into the existing schedule
sampler. This matches §2.5 of `workflow.md`.
"""

import os
import numpy as np
import torch
from einops import repeat
from torch import nn

import constants as cst
from constants import LearningHyperParameter
from models.diffusers.multi_asset.ablation_flags import AblationFlags
from models.diffusers.multi_asset.graph import (
    AttentionAggregator,
    EdgeWeightNet,
    MessageFunction,
    NoiseFusion,
    RelationEmbedding,
    compute_rolling_stats,
)
from models.diffusers.multi_asset.shared_score_net import SharedScoreNet


class MultiAssetGaussianDiffusion(nn.Module):
    def __init__(self, config, asset_universe, feature_augmenter=None):
        super().__init__()
        self.config = config
        self.asset_universe = asset_universe
        self.num_assets = asset_universe.num_assets

        self.dropout = config.HYPER_PARAMETERS[LearningHyperParameter.DROPOUT]
        self.batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE]
        self.num_diffusionsteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        self.lambda_ = config.HYPER_PARAMETERS[LearningHyperParameter.LAMBDA]
        self.gen_seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]
        self.seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE]
        self.cond_seq_size = self.seq_size - self.gen_seq_size
        self.depth = config.HYPER_PARAMETERS[LearningHyperParameter.CDT_DEPTH]
        self.num_heads = config.HYPER_PARAMETERS[LearningHyperParameter.CDT_NUM_HEADS]
        self.mlp_ratio = config.HYPER_PARAMETERS[LearningHyperParameter.CDT_MLP_RATIO]
        self.cond_dropout_prob = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.sampling_type = config.SAMPLING_TYPE
        self.IS_AUGMENTATION = config.IS_AUGMENTATION
        self.cond_method = config.COND_METHOD
        self.cond_type = config.COND_TYPE

        if self.IS_AUGMENTATION:
            self.input_size = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
            self.feature_augmenter = feature_augmenter
        else:
            self.input_size = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB]
            self.feature_augmenter = None
        self.order_feature_dim = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB]

        self.NN = SharedScoreNet(
            num_assets=self.num_assets,
            input_size=self.input_size,
            cond_seq_len=self.cond_seq_size,
            num_diffusionsteps=self.num_diffusionsteps,
            depth=self.depth,
            num_heads=self.num_heads,
            gen_sequence_size=self.gen_seq_size,
            cond_dropout_prob=self.cond_dropout_prob,
            is_augmented=self.IS_AUGMENTATION,
            dropout=self.dropout,
            cond_type=self.cond_type,
            cond_method=self.cond_method,
        )

        # Diffusion schedule (mirrors single-asset GaussianDiffusion).
        # Registered as buffers so Lightning / .to(device) relocates them to
        # the correct per-process device under DDP. Plain attributes would
        # stay on the construction device (cuda:0) and cause a device mismatch
        # when indexed by t on another rank's tensor (cuda:1).
        betas = config.BETAS
        if not torch.is_tensor(betas):
            betas = torch.tensor(betas, dtype=torch.float32)
        betas = betas.detach().to(dtype=torch.float32, device="cpu")
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0, dtype=torch.float32)
        alphas_cumprod_prev = torch.cat([alphas_cumprod[:1], alphas_cumprod[:-1]])
        posterior_var = (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("posterior_var", posterior_var)
        self.register_buffer("posterior_log_var_clipped", torch.log(posterior_var))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        if self.sampling_type == "DDIM":
            self.ddim_eta = config.HYPER_PARAMETERS[LearningHyperParameter.DDIM_ETA]
            self.ddim_nsteps = config.HYPER_PARAMETERS[LearningHyperParameter.DDIM_NSTEPS]
            tmp = self.num_diffusionsteps / self.ddim_nsteps
            self.t = torch.arange(0, self.num_diffusionsteps, tmp).long() + 1
            self.ddim_alpha = self.alphas_cumprod[self.t].clone()
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat(
                [self.alphas_cumprod[:1], self.alphas_cumprod[self.t[:-1]]]
            )
            self.ddim_sqrt_one_minus_alpha = (1.0 - self.ddim_alpha) ** 0.5
            self.ddim_sigma = (
                self.ddim_eta
                * (
                    (1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha)
                    * (1 - self.ddim_alpha / self.ddim_alpha_prev)
                ) ** 0.5
            )

        # Fixed buffer of asset ids -- shape (N,), shared across batch.
        self.register_buffer("asset_ids", torch.arange(self.num_assets, dtype=torch.long))

        # P1 graph-coupling modules. The residual fusion gamma starts at 0,
        # so graph-enabled runs begin exactly at P0 behavior.
        self.ablation_flags = AblationFlags.from_config(config)
        relation_dim = getattr(config, "GRAPH_RELATION_DIM", 8)
        graph_hidden_dim = getattr(config, "GRAPH_HIDDEN_DIM", 32)
        edge_src, edge_dst, edge_relation_ids = self._build_graph_edges()
        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)
        self.register_buffer("edge_relation_ids", edge_relation_ids)
        self.relation_embedding = RelationEmbedding(
            num_relation_types=max(1, self.asset_universe.num_relation_types),
            relation_dim=relation_dim,
        )
        self.edge_weight_net = EdgeWeightNet(
            stats_dim=4,
            relation_dim=relation_dim,
            hidden_dim=graph_hidden_dim,
        )
        self.message_fn = MessageFunction(
            feature_dim=self.order_feature_dim,
            hidden_dim=graph_hidden_dim,
        )
        self.aggregator = AttentionAggregator(
            feature_dim=self.order_feature_dim,
            hidden_dim=graph_hidden_dim,
        )
        self.noise_fusion = NoiseFusion(
            feature_dim=self.order_feature_dim,
            hidden_dim=graph_hidden_dim,
            init_gamma=0.0,
        )

        # Extension points for P2 / P3. Plain attributes (not nn.Module) so
        # that ablation toggles can swap them out without re-registering
        # parameters. Signature: hook(eps, x_t=..., t=..., **ctx) -> eps.
        self.pre_fusion_hook = _identity_hook
        self.post_fusion_hook = _identity_hook

        self.init_losses()

    # ---- P1 hook ---------------------------------------------------------

    def fuse(self, eps_local, **ctx):
        """Per-step graph fusion across assets.

        With ``disable_graph=True`` this returns ``eps_local`` directly,
        giving the P0 shared-backbone behavior bit-for-bit.
        """
        if self.ablation_flags.disable_graph or self.edge_src.numel() == 0:
            return eps_local

        x_t = ctx["x_t"]
        if x_t.shape[-1] != eps_local.shape[-1]:
            raise ValueError(
                "graph fusion expects x_t and eps_local to share the feature "
                f"dimension, got {x_t.shape[-1]} and {eps_local.shape[-1]}"
            )

        raw_cond_orders = ctx.get("raw_cond_orders", None)
        if raw_cond_orders is None:
            raw_cond_orders = ctx.get("cond_orders", None)
        if raw_cond_orders is None:
            return eps_local
        raw_cond_lob = ctx.get("raw_cond_lob", None)
        if raw_cond_lob is None:
            raw_cond_lob = ctx.get("cond_lob", None)

        stats = compute_rolling_stats(raw_cond_orders, raw_cond_lob).to(
            device=eps_local.device,
            dtype=eps_local.dtype,
        )
        src = self.edge_src.to(eps_local.device)
        dst = self.edge_dst.to(eps_local.device)
        rel_ids = self.edge_relation_ids.to(eps_local.device)

        stats_src = stats[:, src, :]
        stats_dst = stats[:, dst, :]
        relation_emb = self.relation_embedding(rel_ids).to(dtype=eps_local.dtype)
        relation_emb = relation_emb.unsqueeze(0).expand(eps_local.shape[0], -1, -1)

        edge_weights = self.edge_weight_net(stats_src, stats_dst, relation_emb)
        if self.ablation_flags.freeze_edge_weights:
            edge_weights = edge_weights.detach()

        messages = self.message_fn(x_t[:, src, :, :], eps_local[:, src, :, :], edge_weights)
        aggregated = self.aggregator(messages, x_t, src, dst, num_nodes=self.num_assets)
        return self.noise_fusion(eps_local, aggregated)

    @property
    def graph_gamma(self):
        return self.noise_fusion.gamma

    def _build_graph_edges(self):
        edges = list(self.asset_universe.directed_edges())
        if not edges and self.num_assets > 1:
            edges = [
                (j, i)
                for j in range(self.num_assets)
                for i in range(self.num_assets)
                if i != j
            ]

        src = []
        dst = []
        rel_ids = []
        for j, i in edges:
            src.append(j)
            dst.append(i)
            if (j, i) in self.asset_universe.relation_types:
                rel_ids.append(self.asset_universe.relation_id(j, i))
            else:
                rel_ids.append(0)

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(rel_ids, dtype=torch.long),
        )

    # ---- forward (noising) process --------------------------------------

    def forward_reparametrized(self, x_0, t):
        """x_0: (B, N, K_gen, F). t: (B,). Returns x_t and noise same shape."""
        noise = torch.distributions.normal.Normal(0, 1).sample(x_0.shape).to(
            x_0.device, non_blocking=True
        )
        sqrt_acp = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus = torch.sqrt(1 - self.alphas_cumprod[t])
        first_term = torch.einsum("bnld,b->bnld", x_0, sqrt_acp)
        second_term = torch.einsum("bnld,b->bnld", noise, sqrt_one_minus)
        x_t = first_term + second_term
        return x_t, noise

    # ---- augmentation (per-asset, via flatten) --------------------------

    def augment(self, x_t, cond_orders, cond_lob):
        if not self.IS_AUGMENTATION:
            return x_t, cond_orders, cond_lob
        B, N, K_gen, F_in = x_t.shape
        K_cond = cond_orders.shape[2]
        full = torch.cat([cond_orders, x_t], dim=2)  # (B, N, K_cond+K_gen, F_in)
        full_flat = full.reshape(B * N, K_cond + K_gen, F_in)
        cond_lob_flat = None
        if cond_lob is not None:
            cond_lob_flat = cond_lob.reshape(B * N, cond_lob.shape[2], cond_lob.shape[3])
        full_aug, cond_lob_aug_flat = self.feature_augmenter.augment(full_flat, cond_lob_flat)
        F_aug = full_aug.shape[-1]
        full_aug = full_aug.reshape(B, N, K_cond + K_gen, F_aug)
        cond_orders_aug = full_aug[:, :, :K_cond, :]
        x_t_aug = full_aug[:, :, K_cond:, :]
        if cond_lob_aug_flat is None:
            cond_lob_aug = cond_lob
        else:
            cond_lob_aug = cond_lob_aug_flat.reshape(B, N, *cond_lob_aug_flat.shape[1:])
        return x_t_aug, cond_orders_aug, cond_lob_aug

    def deaugment(self, noise, v):
        """noise / v: (B, N, K_gen, F_aug) -> (B, N, K_gen, F_orig)."""
        B, N, K, F_aug = noise.shape
        noise_flat = noise.reshape(B * N, K, F_aug)
        v_flat = v.reshape(B * N, K, F_aug)
        noise_out, v_out = self.feature_augmenter.deaugment(noise_flat, v_flat)
        F_out = noise_out.shape[-1]
        return noise_out.reshape(B, N, K, F_out), v_out.reshape(B, N, K, F_out)

    # ---- sampling -------------------------------------------------------

    def sample(
        self,
        x_0,
        real_cond_orders,
        real_cond_lob,
        weights,
        graph_cond_orders=None,
        graph_cond_lob=None,
    ):
        if self.sampling_type == "DDIM":
            return self.ddim_sample(
                x_0,
                real_cond_orders,
                real_cond_lob,
                graph_cond_orders=graph_cond_orders,
                graph_cond_lob=graph_cond_lob,
            )
        return self.ddpm_sample(
            x_0,
            real_cond_orders,
            real_cond_lob,
            weights,
            graph_cond_orders=graph_cond_orders,
            graph_cond_lob=graph_cond_lob,
        )

    def ddpm_sample(self, x_0, cond_orders, cond_lob, weights,
                    graph_cond_orders=None, graph_cond_lob=None):
        orig_cond_orders = cond_orders.detach().clone()
        orig_cond_lob = cond_lob.detach().clone() if cond_lob is not None else None
        raw_cond_orders = (
            graph_cond_orders.detach().clone()
            if graph_cond_orders is not None else orig_cond_orders
        )
        raw_cond_lob = (
            graph_cond_lob.detach().clone()
            if graph_cond_lob is not None else orig_cond_lob
        )
        t = torch.full(size=(x_0.shape[0],), fill_value=self.num_diffusionsteps - 1,
                       device=x_0.device, dtype=torch.int64)
        x_t, noise = self.forward_reparametrized(x_0, t)
        for _ in range(self.num_diffusionsteps - 1, -1, -1):
            x_t_aug, cond_orders_aug, cond_lob_aug = self.augment(x_t, orig_cond_orders, orig_cond_lob)
            x_t = self.ddpm_single_step(
                x_0,
                x_t_aug,
                x_t,
                t,
                cond_orders_aug,
                noise,
                weights,
                cond_lob_aug,
                raw_cond_orders=raw_cond_orders,
                raw_cond_lob=raw_cond_lob,
            )
            t = t - 1
        return x_t

    def ddim_sample(self, x_0, cond_orders, cond_lob,
                    graph_cond_orders=None, graph_cond_lob=None):
        orig_cond_orders = cond_orders.detach().clone()
        orig_cond_lob = cond_lob.detach().clone() if cond_lob is not None else None
        raw_cond_orders = (
            graph_cond_orders.detach().clone()
            if graph_cond_orders is not None else orig_cond_orders
        )
        raw_cond_lob = (
            graph_cond_lob.detach().clone()
            if graph_cond_lob is not None else orig_cond_lob
        )
        tmp = torch.full(size=(x_0.shape[0],), fill_value=self.num_diffusionsteps - 1,
                         device=x_0.device, dtype=torch.int64)
        x_t, _ = self.forward_reparametrized(x_0, tmp)
        time_steps = torch.flip(self.t, dims=(0,))
        for i, step in enumerate(time_steps):
            x_t_aug, cond_orders_aug, cond_lob_aug = self.augment(x_t, orig_cond_orders, orig_cond_lob)
            index = len(time_steps) - i - 1
            ts = x_t.new_full((x_0.shape[0],), step, dtype=torch.long)
            x_t = self.ddim_single_step(
                x_t_aug,
                cond_lob_aug,
                cond_orders_aug,
                ts,
                index,
                x_t,
                raw_cond_orders=raw_cond_orders,
                raw_cond_lob=raw_cond_lob,
            )
        return x_t

    def ddim_single_step(self, x_t_aug, cond_lob, cond_orders, ts, index, x_t,
                         raw_cond_orders=None, raw_cond_lob=None):
        eps_local, v = self.NN(x_t_aug, cond_orders, ts, cond_lob, self.asset_ids)
        if self.IS_AUGMENTATION:
            eps_local, v = self.deaugment(eps_local, v)
        eps_local = self.pre_fusion_hook(eps_local, x_t=x_t, t=ts,
                                         cond_orders=cond_orders, cond_lob=cond_lob,
                                         raw_cond_orders=raw_cond_orders,
                                         raw_cond_lob=raw_cond_lob)
        eps_fused = self.fuse(eps_local, x_t=x_t, t=ts,
                              cond_orders=cond_orders, cond_lob=cond_lob,
                              raw_cond_orders=raw_cond_orders,
                              raw_cond_lob=raw_cond_lob)
        eps_fused = self.post_fusion_hook(eps_fused, x_t=x_t, t=ts,
                                          cond_orders=cond_orders, cond_lob=cond_lob,
                                          raw_cond_orders=raw_cond_orders,
                                          raw_cond_lob=raw_cond_lob)
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        pred_x0 = (x_t - sqrt_one_minus_alpha * eps_fused) / (alpha ** 0.5)
        dir_xt = (1.0 - alpha_prev - sigma ** 2).sqrt() * eps_fused
        if sigma == 0.0:
            noise = 0.0
        else:
            noise = torch.randn(x_t.shape, device=x_t.device)
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev

    # ---- one reverse DDPM step (training + eval) ------------------------

    def ddpm_single_step(self, x_0, x_t_aug, x_t, t, cond_orders, noise_true,
                         weights, cond_lob, batch_idx=None,
                         raw_cond_orders=None, raw_cond_lob=None):
        B, N, K, F = x_0.shape

        beta_t = self.betas[t]
        alpha_t = 1 - beta_t
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = repeat(beta_t, "b -> b n l d", n=N, l=K, d=F)
        alpha_t = repeat(alpha_t, "b -> b n l d", n=N, l=K, d=F)
        alpha_cumprod_t = repeat(alpha_cumprod_t, "b -> b n l d", n=N, l=K, d=F)

        eps_local, v = self.NN(x_t_aug, cond_orders, t, cond_lob, self.asset_ids)
        if self.IS_AUGMENTATION:
            eps_local, v = self.deaugment(eps_local, v)
        if torch.isnan(eps_local).any():
            print("eps_local:", eps_local.max())

        # P2/P3 extension points + P1 fusion.
        eps_local = self.pre_fusion_hook(eps_local, x_t=x_t, t=t,
                                         cond_orders=cond_orders, cond_lob=cond_lob,
                                         raw_cond_orders=raw_cond_orders,
                                         raw_cond_lob=raw_cond_lob)
        eps_fused = self.fuse(eps_local, x_t=x_t, t=t,
                              cond_orders=cond_orders, cond_lob=cond_lob,
                              raw_cond_orders=raw_cond_orders,
                              raw_cond_lob=raw_cond_lob)
        eps_fused = self.post_fusion_hook(eps_fused, x_t=x_t, t=t,
                                          cond_orders=cond_orders, cond_lob=cond_lob,
                                          raw_cond_orders=raw_cond_orders,
                                          raw_cond_lob=raw_cond_lob)

        # Variance head (same parametrization as single-asset).
        frac = (v + 1) / 2
        max_log = torch.log(beta_t)
        min_log = repeat(self.posterior_log_var_clipped[t], "b -> b n l d", n=N, l=K, d=F)
        log_var_t = frac * max_log + (1 - frac) * min_log
        var_t = torch.exp(log_var_t)
        std_t = torch.sqrt(var_t)

        z = torch.distributions.normal.Normal(0, 1).sample(x_t.shape).to(
            x_t.device, non_blocking=True
        )
        # zero the noise where t == 0 (batch dim only)
        indexes = torch.where(t == 0)[0]
        if len(indexes) > 0:
            z[indexes] = 0.0

        x_recon = 1 / torch.sqrt(alpha_t) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t) * eps_fused)
        ) + (std_t * z)

        # ---- losses (per-asset, then summed across N) -------------------
        L_mse = self._mse_loss_per_asset(eps_fused, noise_true)  # (B, N)
        L_mse_total = L_mse.sum(dim=1)  # (B,)
        self.mse_losses.append(L_mse_total)

        L_vlb = self._vlb_loss_per_asset(
            noise_t=eps_fused.detach(),
            pred_log_var=log_var_t,
            x_0=x_0,
            x_t=x_t,
            t=t,
            beta_t=beta_t,
            alpha_t=alpha_t,
            alpha_cumprod_t=alpha_cumprod_t,
            weights=weights,
        )  # (B, N)
        if torch.isnan(L_vlb).any():
            print("L_vlb:", L_vlb.max())
        L_vlb_total = L_vlb.sum(dim=1)  # (B,)
        self.vlb_losses.append(L_vlb_total)

        return x_recon

    # ---- loss helpers (per-asset shapes) --------------------------------

    def _mse_loss_per_asset(self, noise_t, noise_true):
        """L2 norm of noise residual per (batch, asset). Returns (B, N).

        TIME_LOSS_W>1 upweights the time channel (idx 0) of the epsilon error
        to attack the inter_arrival weakness (NEXT_WORK_ZH.md task 2.2).
        """
        diff = noise_t - noise_true  # (B, N, K, F)
        tw = float(os.environ.get("TIME_LOSS_W", "1.0"))
        if tw != 1.0:
            w = torch.ones(diff.shape[-1], device=diff.device, dtype=diff.dtype)
            w[0] = tw ** 0.5  # squared inside L2 norm -> effective weight tw
            diff = diff * w
        # MSE_REDUCE='mean' (task 3.2): mean-of-squares per asset instead of the
        # L2 norm. Norm scale ~sqrt(K*F) makes the early epsilon gradient hot ->
        # seed-sensitivity; mean keeps it O(1). Default 'norm' = baseline.
        if os.environ.get("MSE_REDUCE", "norm") == "mean":
            return (diff ** 2).mean(dim=[2, 3])
        return torch.norm(diff, p=2, dim=[2, 3])

    def _vlb_loss_per_asset(self, noise_t, pred_log_var, x_0, x_t, t,
                            beta_t, alpha_t, alpha_cumprod_t, weights):
        true_mean, true_log_variance_clipped = self._q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)
        pred_mean = self._p_mean(noise_t, x_t, beta_t, alpha_t, alpha_cumprod_t)
        kl = self._normal_kl(true_mean, true_log_variance_clipped, pred_mean, pred_log_var)
        kl = self._mean_per_asset(kl) / np.log(2.0)
        decoder_nll = -self._gaussian_log_likelihood(
            x_0, means=pred_mean, log_scales=pred_log_var * 0.5
        )
        decoder_nll = self._mean_per_asset(decoder_nll) / np.log(2.0)
        # At t==0 use decoder NLL, otherwise KL.
        t_b = t.view(-1, 1)  # (B, 1) broadcast over N
        output = torch.where((t_b == 0), decoder_nll, kl)
        w = torch.from_numpy(weights).to(t.device)[t].view(-1, 1)
        return output / w

    def _p_mean(self, noise_t, x_t, beta_t, alpha_t, alpha_cumprod_t):
        return 1 / torch.sqrt(alpha_t) * (x_t - (beta_t * noise_t / torch.sqrt(1 - alpha_cumprod_t)))

    def _q_posterior_mean_var(self, x_0, x_t, t):
        F = x_0.shape[-1]
        N = x_0.shape[1]
        c1 = repeat(self.posterior_mean_coef1[t], "b -> b n 1 d", n=N, d=F)
        c2 = repeat(self.posterior_mean_coef2[t], "b -> b n 1 d", n=N, d=F)
        true_mean = c1 * x_0 + c2 * x_t
        true_log_var = repeat(self.posterior_log_var_clipped[t], "b -> b n 1 d", n=N, d=F)
        return true_mean, true_log_var

    def _normal_kl(self, mean1, logvar1, mean2, logvar2):
        return 0.5 * (
            -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )

    def _gaussian_log_likelihood(self, x, means, log_scales):
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(log_scales)
        plus_in = inv_stdv * (centered_x + 1.0)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-6))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-6))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-6))),
        )
        return log_probs

    @staticmethod
    def _approx_standard_normal_cdf(x):
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    @staticmethod
    def _mean_per_asset(tensor):
        """Mean over all dims except (batch, asset). Input (B, N, ...), out (B, N)."""
        if tensor.dim() <= 2:
            return tensor
        return tensor.mean(dim=list(range(2, tensor.dim())))

    # ---- public loss accumulators (parity with single-asset) ------------

    def loss(self):
        L_simple = torch.stack(self.mse_losses)
        L_vlb = torch.stack(self.vlb_losses)
        L_hybrid = L_simple + self.lambda_ * L_vlb
        return L_hybrid, L_simple, L_vlb

    def init_losses(self):
        self.mse_losses = []
        self.vlb_losses = []


def _identity_hook(x, **_ctx):
    """Default pre/post fusion hook: returns its input unchanged.

    Real hooks (P2 spread conditioning, P3 energy guidance) follow the
    same ``(tensor, **ctx) -> tensor`` contract.
    """
    return x
