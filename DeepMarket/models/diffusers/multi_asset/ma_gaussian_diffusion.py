"""MultiAssetGaussianDiffusion — N-asset variant of
:class:`models.diffusers.gaussian_diffusion.GaussianDiffusion`.

All tensors carry a leading ``(B, N, ...)`` shape. P0 keeps the per-asset
denoising fully independent: ``fuse`` is identity. P1 replaces ``fuse`` with
a graph stack. P2 computes spread conditioning in ``pre_fusion_hook`` during
sampling; P3 applies energy guidance in ``post_fusion_hook``.

Loss policy: each asset contributes a hybrid (L_simple + lambda * L_vlb)
loss; the per-asset scalars are *summed* across assets so the batch-level
loss has shape ``(B,)`` and slots straight into the existing schedule
sampler. This matches §2.5 of `workflow.md`.
"""

import numpy as np
import torch
from einops import repeat
from torch import nn

import constants as cst
from constants import LearningHyperParameter
from models.diffusers.multi_asset.ablation_flags import AblationFlags
from models.diffusers.multi_asset.arbitrage import (
    ArbitrageEnergyGuidance,
    ReverseLoopState,
    StressHead,
)
from models.diffusers.multi_asset.graph import GraphCoupler
from models.diffusers.multi_asset.spread import SpreadConditioner
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
        self.betas = config.BETAS
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.cat(
            [torch.Tensor([self.alphas_cumprod[0]]).to(cst.DEVICE), self.alphas_cumprod[:-1]]
        )
        self.posterior_var = (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * self.betas
        self.posterior_log_var_clipped = torch.log(self.posterior_var)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        if self.sampling_type == "DDIM":
            self.ddim_eta = config.HYPER_PARAMETERS[LearningHyperParameter.DDIM_ETA]
            self.ddim_nsteps = config.HYPER_PARAMETERS[LearningHyperParameter.DDIM_NSTEPS]
            tmp = self.num_diffusionsteps / self.ddim_nsteps
            self.t = torch.arange(0, self.num_diffusionsteps, tmp).long() + 1
            self.ddim_alpha = self.alphas_cumprod[self.t].clone()
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat(
                [torch.Tensor([self.alphas_cumprod[0]]).to(cst.DEVICE), self.alphas_cumprod[self.t[:-1]]]
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
        self.graph_coupler = GraphCoupler(
            asset_universe=self.asset_universe,
            feature_dim=self.order_feature_dim,
            relation_dim=relation_dim,
            hidden_dim=graph_hidden_dim,
            flags=self.ablation_flags,
        )

        # P2 spread-conditioning controls. ``K_SPREAD_STEPS=None`` maps to
        # the default final quarter of the reverse schedule.
        self.k_spread_steps = getattr(config, "K_SPREAD_STEPS", None)
        self.spread_price_feature_index = getattr(config, "SPREAD_PRICE_FEATURE_INDEX", None)
        self.spread_price_is_delta = bool(getattr(config, "SPREAD_PRICE_IS_DELTA", True))
        self.spread_conditioner = SpreadConditioner(
            asset_universe=self.asset_universe,
            alphas_cumprod=self.alphas_cumprod,
            num_diffusionsteps=self.num_diffusionsteps,
            num_assets=self.num_assets,
            k_spread_steps=self.k_spread_steps,
            price_feature_index=self.spread_price_feature_index,
            price_is_delta=self.spread_price_is_delta,
            flags=self.ablation_flags,
        )

        # P3 annealed energy-guidance controls. ``arb_delta_base`` is a
        # checkpointed non-trainable tolerance buffer; callers can overwrite it
        # with the training-set median |spread| via ``set_arb_delta_base``.
        delta_base = getattr(config, "ARB_DELTA_BASE", 1.0)
        self.register_buffer("arb_delta_base", torch.as_tensor(delta_base, dtype=torch.float32))
        self.arb_kappa = float(getattr(config, "ARB_GUIDANCE_KAPPA", 1.0))
        self.arb_lambda_max = float(getattr(config, "ARB_GUIDANCE_LAMBDA_MAX", 1.0))
        self.arb_lambda_power = float(getattr(config, "ARB_GUIDANCE_POWER", 2.0))
        self.arb_stress_head = StressHead(stats_dim=4)
        self.arbitrage_guidance = ArbitrageEnergyGuidance(
            asset_universe=self.asset_universe,
            alphas_cumprod=self.alphas_cumprod,
            num_diffusionsteps=self.num_diffusionsteps,
            k_spread_steps=self.k_spread_steps,
            price_feature_index=self.spread_price_feature_index,
            price_is_delta=self.spread_price_is_delta,
            kappa=self.arb_kappa,
            lambda_max=self.arb_lambda_max,
            lambda_power=self.arb_lambda_power,
            flags=self.ablation_flags,
        )
        self._register_p3_state_compat_hook()

        # Extension points for P2 / P3. ``pre_fusion_hook`` computes the spread
        # condition before the score-net call; ``post_fusion_hook`` applies P3
        # guidance unless the ablation flag makes it an exact no-op.
        self.pre_fusion_hook = self._spread_pre_fusion_hook
        self.post_fusion_hook = self._arb_post_fusion_hook

        self.init_losses()

    # ---- P1 hook ---------------------------------------------------------

    def fuse(self, eps_local, **ctx):
        """Per-step graph fusion across assets.

        With ``disable_graph=True`` this returns ``eps_local`` directly,
        giving the P0 shared-backbone behavior bit-for-bit.
        """
        raw_cond_orders = ctx.get("raw_cond_orders", None)
        if raw_cond_orders is None:
            raw_cond_orders = ctx.get("cond_orders", None)
        raw_cond_lob = ctx.get("raw_cond_lob", None)
        if raw_cond_lob is None:
            raw_cond_lob = ctx.get("cond_lob", None)

        return self.graph_coupler(
            eps_local,
            x_t=ctx["x_t"],
            cond_orders=raw_cond_orders,
            cond_lob=raw_cond_lob,
        )

    @property
    def graph_gamma(self):
        return self.graph_coupler.gamma

    @property
    def noise_fusion(self):
        """Backward-compatible access to the P1 residual fusion module."""
        return self.graph_coupler.noise_fusion

    def _register_p3_state_compat_hook(self) -> None:
        """Register checkpoint compatibility hook across PyTorch versions."""
        if hasattr(self, "register_load_state_dict_pre_hook"):
            self.register_load_state_dict_pre_hook(self._fill_missing_p3_state)
        else:
            self._register_load_state_dict_pre_hook(self._fill_missing_p3_state, with_module=True)

    def _fill_missing_p3_state(self, _module, state_dict, prefix, *_args):
        """Allow strict loading of pre-refactor P0/P1/P2/P3 checkpoints."""
        legacy_graph_prefixes = (
            "edge_src",
            "edge_dst",
            "edge_relation_ids",
            "relation_embedding.",
            "edge_weight_net.",
            "message_fn.",
            "aggregator.",
            "noise_fusion.",
        )
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix):]
            if any(local_key == item or local_key.startswith(item) for item in legacy_graph_prefixes):
                state_dict.setdefault(prefix + "graph_coupler." + local_key, state_dict[key])
                state_dict.pop(key)

        defaults = {
            "arb_delta_base": self.arb_delta_base,
            "arb_stress_head.linear.weight": self.arb_stress_head.linear.weight,
            "arb_stress_head.linear.bias": self.arb_stress_head.linear.bias,
        }
        for name, value in defaults.items():
            state_dict.setdefault(prefix + name, value.detach().clone())

    # ---- P2 spread conditioning ----------------------------------------

    def _spread_pre_fusion_hook(self, x_t, state=None, t=None, cond_lob=None, **_ctx):
        """Compute the spread condition consumed by the next score-net call."""
        self.spread_conditioner.k_spread_steps = self.k_spread_steps
        self.spread_conditioner.price_feature_index = self.spread_price_feature_index
        self.spread_conditioner.price_is_delta = self.spread_price_is_delta
        return self.spread_conditioner(x_t=x_t, state=state, t=t, cond_lob=cond_lob)

    def _predict_x0_from_eps(self, x_t, eps_fused, t):
        """One-shot clean estimate used by the spread decoder."""
        return self.spread_conditioner._predict_x0_from_eps(x_t, eps_fused, t)

    def set_arb_delta_base(self, delta_base) -> None:
        """Set the P3 base spread tolerance buffer.

        Pass a scalar for one tolerance shared by all spread groups, or a
        length-G tensor/list containing the training-set median ``|spread|`` per
        group.
        """
        delta = torch.as_tensor(delta_base, device=self.arb_delta_base.device, dtype=self.arb_delta_base.dtype)
        self.arb_delta_base = delta.detach().clone()

    def calibrate_arb_delta_base(self, spread: torch.Tensor) -> torch.Tensor:
        """Calibrate ``delta_base`` from a ``(B, G)`` historical spread tensor."""
        if spread.dim() != 2:
            raise ValueError(f"spread must be (B, G), got {tuple(spread.shape)}")
        delta = spread.detach().abs().median(dim=0).values
        self.set_arb_delta_base(delta)
        return self.arb_delta_base

    def _arb_post_fusion_hook(self, eps_fused, x_t, t=None, state=None,
                              cond_orders=None, cond_lob=None,
                              raw_cond_orders=None, raw_cond_lob=None, **_ctx):
        """Apply P3 energy-gradient guidance in eps space."""
        self.arbitrage_guidance.k_spread_steps = self.k_spread_steps
        self.arbitrage_guidance.price_feature_index = self.spread_price_feature_index
        self.arbitrage_guidance.price_is_delta = self.spread_price_is_delta
        self.arbitrage_guidance.kappa = self.arb_kappa
        self.arbitrage_guidance.lambda_max = self.arb_lambda_max
        self.arbitrage_guidance.lambda_power = self.arb_lambda_power
        return self.arbitrage_guidance(
            eps_fused,
            x_t=x_t,
            t=t,
            state=state,
            cond_orders=cond_orders,
            cond_lob=cond_lob,
            raw_cond_orders=raw_cond_orders,
            raw_cond_lob=raw_cond_lob,
            delta_base=self.arb_delta_base,
            stress_head=self.arb_stress_head,
        )

    def training_spread_cond(self, x_0, cond_lob, dropout_prob: float = 0.0):
        """Optional P2.7 ground-truth spread condition for fresh P2 training."""
        return self.spread_conditioner.training_condition(
            x_0,
            cond_lob,
            dropout_prob=dropout_prob,
        )

    # ---- forward (noising) process --------------------------------------

    def forward_reparametrized(self, x_0, t):
        """x_0: (B, N, K_gen, F). t: (B,). Returns x_t and noise same shape."""
        noise = torch.distributions.normal.Normal(0, 1).sample(x_0.shape).to(
            cst.DEVICE, non_blocking=True
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
                       device=cst.DEVICE, dtype=torch.int64)
        x_t, noise = self.forward_reparametrized(x_0, t)
        state = ReverseLoopState()
        for _ in range(self.num_diffusionsteps - 1, -1, -1):
            spread_cond = self.pre_fusion_hook(
                x_t=x_t,
                state=state,
                t=t,
                cond_lob=raw_cond_lob,
            )
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
                spread_cond=spread_cond,
                reverse_state=state,
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
                         device=cst.DEVICE, dtype=torch.int64)
        x_t, _ = self.forward_reparametrized(x_0, tmp)
        state = ReverseLoopState()
        time_steps = torch.flip(self.t, dims=(0,))
        for i, step in enumerate(time_steps):
            index = len(time_steps) - i - 1
            ts = x_t.new_full((x_0.shape[0],), step, dtype=torch.long)
            spread_cond = self.pre_fusion_hook(
                x_t=x_t,
                state=state,
                t=ts,
                cond_lob=raw_cond_lob,
            )
            x_t_aug, cond_orders_aug, cond_lob_aug = self.augment(x_t, orig_cond_orders, orig_cond_lob)
            x_t = self.ddim_single_step(
                x_t_aug,
                cond_lob_aug,
                cond_orders_aug,
                ts,
                index,
                x_t,
                raw_cond_orders=raw_cond_orders,
                raw_cond_lob=raw_cond_lob,
                spread_cond=spread_cond,
                reverse_state=state,
            )
        return x_t

    def ddim_single_step(self, x_t_aug, cond_lob, cond_orders, ts, index, x_t,
                         raw_cond_orders=None, raw_cond_lob=None,
                         spread_cond=None, reverse_state=None):
        eps_local, v = self.NN(
            x_t_aug,
            cond_orders,
            ts,
            cond_lob,
            self.asset_ids,
            spread_cond=spread_cond,
        )
        if self.IS_AUGMENTATION:
            eps_local, v = self.deaugment(eps_local, v)
        eps_fused = self.fuse(eps_local, x_t=x_t, t=ts,
                              cond_orders=cond_orders, cond_lob=cond_lob,
                              raw_cond_orders=raw_cond_orders,
                              raw_cond_lob=raw_cond_lob)
        eps_fused = self.post_fusion_hook(eps_fused, x_t=x_t, t=ts,
                                          cond_orders=cond_orders, cond_lob=cond_lob,
                                          raw_cond_orders=raw_cond_orders,
                                          raw_cond_lob=raw_cond_lob,
                                          state=reverse_state)
        if reverse_state is not None:
            reverse_state.update_eps(eps_fused)
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
                         raw_cond_orders=None, raw_cond_lob=None,
                         spread_cond=None, reverse_state=None):
        B, N, K, F = x_0.shape

        beta_t = self.betas[t]
        alpha_t = 1 - beta_t
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = repeat(beta_t, "b -> b n l d", n=N, l=K, d=F)
        alpha_t = repeat(alpha_t, "b -> b n l d", n=N, l=K, d=F)
        alpha_cumprod_t = repeat(alpha_cumprod_t, "b -> b n l d", n=N, l=K, d=F)

        eps_local, v = self.NN(
            x_t_aug,
            cond_orders,
            t,
            cond_lob,
            self.asset_ids,
            spread_cond=spread_cond,
        )
        if self.IS_AUGMENTATION:
            eps_local, v = self.deaugment(eps_local, v)
        if torch.isnan(eps_local).any():
            print("eps_local:", eps_local.max())

        # P2 spread conditioning is injected into the score-net call above;
        # P1 graph fusion and the P3 post-fusion hook operate on eps here.
        eps_fused = self.fuse(eps_local, x_t=x_t, t=t,
                              cond_orders=cond_orders, cond_lob=cond_lob,
                              raw_cond_orders=raw_cond_orders,
                              raw_cond_lob=raw_cond_lob)
        eps_fused = self.post_fusion_hook(eps_fused, x_t=x_t, t=t,
                                          cond_orders=cond_orders, cond_lob=cond_lob,
                                          raw_cond_orders=raw_cond_orders,
                                          raw_cond_lob=raw_cond_lob,
                                          state=reverse_state)
        if reverse_state is not None:
            reverse_state.update_eps(eps_fused)

        # Variance head (same parametrization as single-asset).
        frac = (v + 1) / 2
        max_log = torch.log(beta_t)
        min_log = repeat(self.posterior_log_var_clipped[t], "b -> b n l d", n=N, l=K, d=F)
        log_var_t = frac * max_log + (1 - frac) * min_log
        var_t = torch.exp(log_var_t)
        std_t = torch.sqrt(var_t)

        z = torch.distributions.normal.Normal(0, 1).sample(x_t.shape).to(
            cst.DEVICE, non_blocking=True
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
        """L2 norm of noise residual per (batch, asset). Returns (B, N)."""
        diff = noise_t - noise_true  # (B, N, K, F)
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
        w = torch.from_numpy(weights).to(cst.DEVICE)[t].view(-1, 1)
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
