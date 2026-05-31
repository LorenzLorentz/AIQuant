# AIQuant Workflow — Multi-Asset Graph-Coupled Diffusion (P0 + P1 + P2 + P3)

This document covers the implementation scope from the multi-asset baseline
through quasi-no-arbitrage guidance. It describes:

1. The problem we are solving in this scope.
2. The model architecture and how it deviates from single-asset TRADES.
3. The codebase layout — what is reused, what is added.
4. A stepwise build plan: P0 (multi-asset shared backbone) → P1 (graph
   coupling) → P2 (spread-aware conditioning) → P3 (annealed energy guidance).

P3 adds optional soft guidance on ETF-basket price-relation energy. It does
not impose hard constraints, replace P2 conditioning, or change the base
DDPM/DDIM update equations.

---

## 1. Problem Definition (scoped to P0 + P1 + P2 + P3)

### 1.1 Setting

We have **N = 2** assets with a known economic relation (initial target:
one ETF and one of its most heavily-weighted constituents from LOBSTER-style
Level-3 data). Both assets are observed over a common time interval.

For each asset `i ∈ {1, 2}` and each time step `t`:

- **Order event** `e_t^(i) ∈ ℝ^6` — six-dim vector as in DeepMarket
  (`constants.LEN_ORDER = 6`): inter-arrival Δt, event type, side, relative
  price, size, and one extra DeepMarket-specific field.
- **Top-L book state** `B_t^(i) ∈ ℝ^{40}` — top-10 levels × 4 (bid/ask price
  + bid/ask depth) = 40, as in DeepMarket (`N_LOB_LEVELS=10`, `LEN_LEVEL=4`).

### 1.2 Generation task

Given a per-asset conditioning history of length `K_cond`, generate the next
`K_gen` order events **jointly** for the two assets:

```
input  per asset:  cond_orders^(i) ∈ ℝ^{K_cond × 6}
                    cond_lob^(i)    ∈ ℝ^{(K_cond+1) × 40}
output per asset:  x_0^(i)         ∈ ℝ^{K_gen × 6}
```

Joint distribution being modeled:

```
p_θ( x_0^(1), x_0^(2) | cond_orders^(1), cond_orders^(2),
                        cond_lob^(1),    cond_lob^(2) )
```

The single-asset version factorizes as `p(x^(1)|cond^(1)) · p(x^(2)|cond^(2))`.
Our model breaks this independence by making the per-step denoising of asset
`i` depend on the current latent state and predicted noise of asset `j`,
through a learned **state-dependent coupling**.

### 1.3 What we are *not* doing in P0 + P1 + P2 + P3

- **No** hard cross-asset constraints. P3 is soft annealed guidance only.
- **No** sparsification of the graph. With N = 2 the graph is fully connected
  (2 directed edges).
- **No** multivariate-Hawkes / copula / ABIDES baselines yet. Those are P5.

### 1.4 Success criteria for this scope

| Stage | What "done" means                                                                              |
| ----- | ---------------------------------------------------------------------------------------------- |
| P0    | Multi-asset shared-backbone diffusion trains end-to-end, per-asset loss curves look like TRADES. |
| P1    | Adding graph coupling: (a) loss does not regress; (b) ablation flag `disable_graph=True` recovers P0 exactly; (c) generated trajectories show non-trivial cross-asset correlation that the P0 baseline lacks. |
| P2    | Spread-aware hook defaults to P1 behavior; when enabled, it can learn from ETF-constituent spread / NAV-gap context without loss regression, and generated samples improve the cross-asset price-relation sanity metric over P1. |
| P3    | Energy guidance defaults to P2 behavior; when enabled it only acts in late denoising, keeps guidance energy/loss finite, and generated samples improve the NAV-gap sanity metric over P2 baseline or at least do not regress. |

Detailed metrics (lead–lag, cross-corr, etc.) live in P4 evaluation but we
already need a coarse cross-correlation sanity check at the end of P1.

---

## 2. Model Architecture

### 2.1 Reused: TRADES score net (unchanged interface)

`DeepMarket/models/diffusers/TRADES/TRADES.py`:

```
TRADES.forward(x, cond_orders, t, cond_lob) -> (noise, var)

  x           : (B, K_gen, F_in)            -- noisy event tokens
  cond_orders : (B, K_cond, F_in)           -- history events
  t           : (B,) int64                  -- diffusion step
  cond_lob    : (B, K_cond+1, 40) or None   -- LOB conditioning

  noise       : (B, K_gen, F_in)
  var         : (B, K_gen, F_in)
```

We do **not** modify TRADES itself. We wrap it.

### 2.2 P0: Shared score net with asset embedding

We treat the N assets as an extra leading dimension and run the shared TRADES
once per asset using the same parameters. The only new information injected
is an asset embedding:

```
shared_score_net(x_t, cond_orders, t, cond_lob, asset_id) -> (eps_local, var)

  x_t         : (B, N, K_gen, F_in)
  cond_orders : (B, N, K_cond, F_in)
  cond_lob    : (B, N, K_cond+1, 40)
  asset_id    : (N,) int64                  -- e.g. [0, 1]

  eps_local   : (B, N, K_gen, F_in)
  var         : (B, N, K_gen, F_in)
```

Implementation: a lightweight `nn.Embedding(N, F_in)` whose vector is added
into the first cond_orders token (or as an extra prepended token — TBD at
implementation; the cleanest is an added token similar to TRADES' positional
embedding pattern). Reshape `(B, N, ...) → (B*N, ...)` before calling TRADES
so the backbone code is unchanged, then reshape back.

P0 reverse step (asset-independent):

```
eps_fused^(i)  = eps_local^(i)
x_{t-1}^(i)    = ddpm_update(x_t^(i), eps_fused^(i), t)
```

### 2.3 P1: Graph coupling inside each reverse step

For each reverse step `t`:

```
  Step 1:  eps_local[i] = shared_score_net(x_t[i], ...)            for i ∈ {1,2}

  Step 2:  for each ordered pair (j → i):
               w_{ji}(t) = EdgeWeightNet( rolling_stats(j),
                                          rolling_stats(i),
                                          r_{ji} )
               m_{ji}    = MessageFn( x_t[j], eps_local[j], w_{ji}(t) )
           m[i] = Aggregate({ m_{ji} : j ∈ N(i) })

  Step 3:  eps_fused[i] = NoiseFusion( eps_local[i], m[i] )

  Step 4:  x_{t-1}[i]   = ddpm_update( x_t[i], eps_fused[i], t )
```

For N = 2 the in-neighborhood of node `i` is just `{j}` where `j ≠ i`, so the
aggregation collapses to identity. We still wire `Aggregate(...)` as a real
operator so the code generalizes to N > 2 later.

#### 2.3.1 Edge weight `w_{ji}(t)`

Following §4.1.2 of the report:

```
w_{ji}(t) = σ( MLP( concat[ stats_j_window,
                            stats_i_window,
                            r_{ji} ] ) )
```

where `stats_*_window` is a fixed-length feature vector over the last `K_stat`
steps. For P1 we use the cheapest meaningful set:

- rolling mid-price log-return std (volatility)
- rolling order-flow imbalance (signed size sum / total size)
- rolling cancellation ratio (count of cancels / total events)
- rolling mean inter-arrival Δt

So `stats ∈ ℝ^4` per asset, `concat[stats_j, stats_i, r_{ji}] ∈ ℝ^{8 + d_r}`.
Default `d_r = 8`.

These stats are computed from `cond_orders` + `cond_lob` once per batch
(they do not change across the reverse diffusion steps within one generation
window), so this is **not** a per-step compute hit.

#### 2.3.2 Relation embedding `r_{ji}`

A small `nn.Embedding(num_relation_types, d_r)`. For the initial ETF +
constituent setup we have two relation types: `ETF→constituent` and
`constituent→ETF`. The directed edge type fully identifies `r_{ji}`.

#### 2.3.3 Message function `f_φ`

```
f_φ( x_t[j], eps_local[j], w_{ji} ) -> m_{ji} ∈ ℝ^{K_gen × F_in}

  implemented as:
     h = MLP_in( concat[ x_t[j], eps_local[j] ] )   # token-wise
     m_{ji} = w_{ji} · MLP_out( h )                 # scalar weight × tensor
```

#### 2.3.4 Aggregator

```
Aggregate({m_{ji}}) = Σ_j  α_{ji} · m_{ji}            # attention-weighted

  with α_{ji} = softmax_j( score(h_i, h_j) )
```

For N = 2 we keep the attention machinery but it reduces to `α_{ji} = 1`.

#### 2.3.5 Noise fusion `g_ψ`

```
g_ψ( eps_local[i], m[i] ) = eps_local[i]
                          + γ · MLP_fuse( concat[eps_local[i], m[i]] )
```

with a **learnable scalar `γ`** initialized at 0, so that at training start
fusion is a no-op and the model recovers P0 behavior. This is the same trick
used in many residual / gating designs to stabilize the introduction of new
modules. Crucially this also means the **ablation `disable_graph=True`**
simply forces `γ = 0` (and skips the message-passing forward pass).

### 2.4 P2: Spread-Aware Conditioning

P2 adds a learnable **pre-fusion conditioning residual** before graph fusion.
It provides the model with current ETF-constituent price-relation context but
does not act as a hard arbitrage constraint.

From the raw conditioning LOB state, use the last available top-of-book
snapshot per asset:

```
mid_i    = (best_ask_i + best_bid_i) / 2
spread_i =  best_ask_i - best_bid_i
```

For configured ETF basket metadata:

```
nav_gap_etf = mid_etf - Σ_j weight_j · mid_j
```

The per-asset spread context is:

```
c_i = [mid_i, spread_i, signed_nav_gap_i, |signed_nav_gap_i|]
```

where the ETF receives `+nav_gap_etf` and each constituent receives
`-weight_j · nav_gap_etf`. If `AssetUniverse.etf_basket_weights` is empty,
P2 is a no-op even when the hook module exists.

The pre-fusion hook is:

```
eps_spread[i] = eps_local[i]
              + γ_spread · MLP_spread( concat[eps_local[i], c_i] )
```

`γ_spread` is initialized to `0`, so enabling P2 starts exactly from P1
behavior. P2 does not reconstruct future LOB state from generated orders.
`x_hat_0`, annealed energy guidance, persistence tracking, and gradient
correction belong to P3.

### 2.5 P3: Annealed Energy Guidance

P3 adds an optional **post-fusion** quasi-no-arbitrage guidance hook. It uses
ETF-basket metadata to reduce persistent generated NAV-gap dislocations during
late denoising, without adding a training loss or hard projection.

The post-fusion hook receives the fused noise estimate:

```
eps_guided = post_fusion_hook(eps_fused, x_t, t, cond_lob, ...)
```

When enabled, it reconstructs a clean-event proxy:

```
x_hat_0 = (x_t - sqrt(1 - alpha_bar_t) * eps_fused) / sqrt(alpha_bar_t)
```

P3 uses `x_hat_0[..., ENERGY_PRICE_PROXY_INDEX]` as a generated relative price
proxy, averaged over the generation horizon and added to the latest
conditioning mid price:

```
mid_hat_i = last_mid_i + mean_k(x_hat_0[i, k, price_proxy_index])
gap_hat   = mid_hat_etf - Σ_j weight_j · mid_hat_constituent_j
energy    = huber(gap_hat, delta=ENERGY_GUIDANCE_DELTA)
```

The guided noise is:

```
eps_guided = eps_fused - scale(t) · clipped_grad(energy, eps_fused)
```

where `scale(t)` is zero in high-noise timesteps and linearly ramps in the
last `ENERGY_GUIDANCE_START_STEP` reverse steps up to
`ENERGY_GUIDANCE_MAX_SCALE`. Defaults keep P3 off:

```
ENERGY_GUIDANCE_ENABLED       = False
ENERGY_GUIDANCE_START_STEP    = 10
ENERGY_GUIDANCE_MAX_SCALE     = 0.01
ENERGY_GUIDANCE_DELTA         = 1.0
ENERGY_GUIDANCE_MAX_GRAD_NORM = 1.0
ENERGY_PRICE_PROXY_INDEX      = 5
ENERGY_PERSISTENCE_WEIGHT     = 0.0
```

No-op rules:

- no ETF basket metadata ⇒ no-op;
- `GraphAblationFlags.disable_arb_guidance=True` ⇒ no-op;
- disabled config or `scale(t)=0` ⇒ no-op;
- invalid shapes, non-finite energy, or unavailable gradient ⇒ no-op.

P3 v1 includes a persistence-tracker interface with zero default weight. This
keeps the sampling-time state boundary explicit while avoiding extra behavior
until persistence weighting is deliberately enabled.

### 2.6 What is reused unchanged

- `GaussianDiffusion.forward_reparametrized` — forward (noising) process.
- The β schedule, `α_t`, `ᾱ_t`, posterior coefficients, EMA, type embedder.
- The hybrid loss (`L_simple + λ·L_vlb`) — applied per asset and summed.
- `DDPM` and `DDIM` update equations. P1/P2/P3 only change the noise estimate
  before the existing reverse update is applied.

### 2.7 Training objective

Per-asset hybrid loss, summed:

```
L_total = Σ_{i=1}^{N}  L_hybrid^(i)
        = Σ_{i=1}^{N}  ( L_simple^(i) + λ · L_vlb^(i) )
```

No auxiliary cross-asset loss in P0/P1/P2/P3. The graph and
spread-conditioning parameters are trained purely through the per-asset
reconstruction gradient. P3 is inference/reverse-step guidance, not a training
objective. If `γ` stays near zero through training, it is a signal that the
graph is unhelpful and should be investigated before relying on guidance.

---

## 3. Codebase Architecture

### 3.1 What we touch / add

```
DeepMarket/
├── constants.py                            [PATCH] add N_ASSETS, asset-id constants
├── configuration.py                        [PATCH] AssetUniverse hook in config
├── run.py                                  [PATCH] dispatch to MA pipeline when configured
│
├── preprocessing/
│   ├── LOBSTERDataBuilder.py               [unchanged]
│   ├── LOBDataset.py                       [unchanged]   ← used by baseline (A)
│   ├── AssetUniverse.py                    [NEW]
│   └── MultiAssetLOBDataset.py             [NEW]
│
├── models/diffusers/
│   ├── TRADES/                             [unchanged]
│   ├── gaussian_diffusion.py               [unchanged]
│   ├── diffusion_engine.py                 [unchanged]   ← baseline (A) entry
│   │
│   └── multi_asset/                        [NEW]
│       ├── __init__.py
│       ├── ma_diffusion_engine.py          [NEW] MultiAssetDiffusionEngine (Lightning)
│       ├── ma_gaussian_diffusion.py        [NEW] reverse loop with graph hook
│       ├── shared_score_net.py             [NEW] TRADES + asset embedding wrapper
│       ├── ablation_flags.py               [NEW] disable_graph, freeze_edge_weights, disable_spread_conditioning, disable_arb_guidance
│       ├── graph/
│           ├── __init__.py
│           ├── rolling_stats.py            [NEW] per-asset window statistics
│           ├── relation_embedding.py       [NEW] r_{ji}
│           ├── edge_weight_net.py          [NEW] w_{ji}(t)
│           ├── message_passing.py          [NEW] f_φ
│           ├── aggregator.py               [NEW] attention aggregation
│           └── noise_fusion.py             [NEW] g_ψ with learnable γ
│       ├── spread/
│           ├── __init__.py
│           ├── spread_context.py           [NEW] top-of-book mid/spread + NAV gap
│           └── spread_conditioner.py       [NEW] pre-fusion residual conditioning
│       └── arbitrage/
│           ├── __init__.py
│           ├── energy.py                   [NEW] x_hat_0 reconstruction + NAV-gap energy
│           ├── guidance_schedule.py        [NEW] late-step annealed scale
│           ├── persistence_tracker.py      [NEW] optional persistence multiplier
│           └── consistency_check.py        [NEW] shape and basket metadata checks
```

### 3.2 Files we deliberately do **not** touch in P0/P1/P2/P3

- `models/diffusers/TRADES/TRADES.py` — the score net itself.
- `models/diffusers/gaussian_diffusion.py` — the single-asset diffuser remains
  intact so baseline (A) keeps working.
- `models/diffusers/diffusion_engine.py` — the single-asset Lightning module.
- `models/gan/`, `models/feature_augmenters/`.

The multi-asset stack is parallel to the existing single-asset stack, not a
modification of it.

### 3.3 P3 hook boundary

P3 hooks into `ma_gaussian_diffusion.py` at the post-fusion extension point:

- `post_fusion_hook(eps_fused, x_t, ...)` — for energy guidance (P3).

This hook is a no-op unless energy guidance is enabled, ETF basket metadata is
present, and the current timestep is inside the late-denoising guidance window.

---

## 4. Stepwise Build Plan

Each step lists: the file(s) it touches, what it produces, and how we verify
it before moving on. Steps are listed in execution order.

### Phase P0 — Multi-asset shared-backbone diffusion (no graph)

**Step P0.1 — Asset universe config**
- File: `preprocessing/AssetUniverse.py`
- Content: dataclass `AssetUniverse` with fields:
  - `assets: list[str]` (e.g. `["SPY", "AAPL"]`)
  - `relation_types: dict[(i,j) -> int]` (e.g. `{(0,1): 0, (1,0): 1}`)
  - `etf_basket_weights: dict[int -> dict[int, float]]` *(reserved for P2,
    safe to leave empty in P0/P1)*
- Verify: instantiate with the two target tickers, print, no runtime use yet.

**Step P0.2 — Multi-asset dataset**
- File: `preprocessing/MultiAssetLOBDataset.py`
- Behavior: takes N `.npy` paths (one per asset, in the same format DeepMarket
  already produces). Each `__getitem__(idx)` returns a tuple of N triples
  `(cond_orders[i], x_0[i], cond_lob[i])` aligned to the **same wall-clock
  time bucket**.
- Alignment policy for P0: take the intersection of timestamps available in
  both files; if exact match is unavailable, last-observation-carry-forward
  to the coarser asset's grid. (Document the choice in code.)
- Verify: load, check shapes `(N, K_cond, 6)`, `(N, K_gen, 6)`,
  `(N, K_cond+1, 40)`; spot-check that timestamps line up.

**Step P0.3 — Shared score net**
- File: `models/diffusers/multi_asset/shared_score_net.py`
- Wraps TRADES with an `nn.Embedding(N, F_in)` injected into `cond_orders`
  (as an additional first token, with a corresponding extra slot in the
  positional embedding lookup — or, simpler, added element-wise to the first
  cond token).
- Reshape `(B, N, ...) → (B*N, ...)` for the TRADES call, reshape back.
- Verify: forward a dummy `(B=2, N=2, K_gen=8, F_in)` tensor; assert output
  shape, assert that swapping asset 0/1 produces different outputs.

**Step P0.4 — Multi-asset gaussian diffusion**
- File: `models/diffusers/multi_asset/ma_gaussian_diffusion.py`
- Mirrors `GaussianDiffusion` but:
  - Takes tensors with leading `(B, N, ...)`.
  - Calls `shared_score_net` once per reverse step (handling all N together).
  - **Exposes** `pre_fusion_hook` and `post_fusion_hook` as no-op callables
    (these are where P2/P3 will attach).
  - **Calls** `fuse(eps_local) → eps_fused`. In P0 `fuse` is identity. In
    P1 `fuse` becomes the graph stack.
- Loss: per-asset hybrid loss summed across N (and averaged across batch).
- Verify: train one step on dummy data, assert loss is finite, assert
  gradient flows to the shared TRADES weights.

**Step P0.5 — Multi-asset Lightning engine**
- File: `models/diffusers/multi_asset/ma_diffusion_engine.py`
- Mirrors `DiffusionEngine`: owns optimizer, EMA, type embedder, sampler
  scheduling. Forward calls into `ma_gaussian_diffusion`.
- Verify: launch a 1-epoch dry run on a tiny subset (`IS_DEBUG=True` style)
  end-to-end via a modified `run.py`. Confirm checkpoints save, val loss
  computes.

**Step P0.6 — Entry-point wiring**
- File: `run.py` (patch) and `configuration.py` (patch)
- Add an option `MULTI_ASSET=True` (or a new `cst.Models.MA_TRADES` enum
  value) that swaps in `MultiAssetLOBDataset` + `MultiAssetDiffusionEngine`.
- Single-asset code paths must continue to run unchanged.
- Verify: both `MULTI_ASSET=False` (existing) and `MULTI_ASSET=True` (new)
  start training without error.

> **P0 exit criterion**: training runs for at least a few hundred steps on
> the real two-asset data, val loss decreases monotonically over several
> validation cycles, and the architecture is equivalent to "two shared-weight
> TRADES trained jointly with no cross-talk."

### Phase P1 — Graph coupling

**Step P1.1 — Rolling stats**
- File: `models/diffusers/multi_asset/graph/rolling_stats.py`
- Function: given `cond_orders` and `cond_lob` of one asset, compute the
  4-dim stats vector defined in §2.3.1.
- Verify: shapes, no NaNs on real data.

**Step P1.2 — Relation embedding**
- File: `models/diffusers/multi_asset/graph/relation_embedding.py`
- `RelationEmbedding(num_relation_types, d_r)`, looks up `r_{ji}` from the
  AssetUniverse.
- Verify: trivial.

**Step P1.3 — Edge weight net**
- File: `models/diffusers/multi_asset/graph/edge_weight_net.py`
- MLP `(stats_j ⊕ stats_i ⊕ r_{ji}) → σ(·) ∈ (0,1)`.
- Verify: forward, check output range and gradient.

**Step P1.4 — Message function**
- File: `models/diffusers/multi_asset/graph/message_passing.py`
- Takes `x_t[j]`, `eps_local[j]`, scalar `w_{ji}`. Returns
  `m_{ji} ∈ ℝ^{K_gen × F_in}`. See §2.3.3.
- Verify: shape and that gradient flows to `w_{ji}`.

**Step P1.5 — Aggregator**
- File: `models/diffusers/multi_asset/graph/aggregator.py`
- Attention-weighted sum of incoming messages. For N=2 this is a single
  message, but write the general code.
- Verify: numerical agreement with hand-computed example for N=2,3.

**Step P1.6 — Noise fusion**
- File: `models/diffusers/multi_asset/graph/noise_fusion.py`
- `g_ψ(eps_local, m) = eps_local + γ · MLP_fuse(concat[eps_local, m])`,
  with `γ` a learnable scalar initialized at 0.
- Verify: at init, `g_ψ(eps_local, m) == eps_local` exactly.

**Step P1.7 — Ablation flags**
- File: `models/diffusers/multi_asset/ablation_flags.py`
- Flags: `disable_graph` (forces `γ=0` and skips message-passing forward),
  `freeze_edge_weights` (detaches the edge MLP), `disable_arb_guidance`
  (reserved for P3).
- Verify: with `disable_graph=True`, model output is bit-identical to P0.

**Step P1.8 — Wire graph into reverse step**
- File: `models/diffusers/multi_asset/ma_gaussian_diffusion.py` (extend)
- Replace the identity `fuse(...)` from P0 with the full graph stack.
- Verify: single reverse step on dummy data, then a short training run.
  Loss should not regress vs. P0. Print `γ` periodically — it should drift
  away from zero if the graph is useful.

**Step P1.9 — Cross-asset sanity metric**
- File: `lob_bench/cross_asset/cross_corr.py` (new directory)
- Compute realized correlation of mid-price returns between the two
  generated assets and compare against the same metric on real data, and
  against P0 generations.
- Verify: P1 samples have measurably higher cross-asset correlation than P0
  samples, and closer to the real-data correlation.

> **P1 exit criterion**: same training stability as P0, `γ` is non-zero at
> convergence, ablation toggle reproduces P0, and the cross-correlation
> sanity metric prefers P1 over P0 on held-out windows.

### Phase P2 — Spread-Aware Conditioning

**Step P2.1 — Spread context**
- File: `models/diffusers/multi_asset/spread/spread_context.py`
- Function: given raw `cond_lob` and `AssetUniverse`, compute per-asset
  context `[mid, spread, signed_nav_gap, abs_signed_nav_gap]`.
- Behavior: if `etf_basket_weights` is empty, return no context so P2 falls
  back to P1.
- Verify: shape `(B, N, 4)`, no NaNs, deterministic output on synthetic LOB.

**Step P2.2 — Spread conditioner**
- File: `models/diffusers/multi_asset/spread/spread_conditioner.py`
- Module: residual pre-fusion hook
  `eps + γ_spread · MLP(concat[eps, spread_context])`.
- Defaults: `γ_spread = 0`; disabled if config says spread conditioning is
  off or ablation flag disables it.
- Verify: `γ_spread = 0` yields exact P1 behavior; with enabled ETF metadata,
  gradients reach `γ_spread`.

**Step P2.3 — Configuration and ablation wiring**
- Files: `configuration.py`, `ablation_flags.py`,
  `ma_gaussian_diffusion.py`.
- Add `SPREAD_CONDITIONING_ENABLED=False`, `SPREAD_CONTEXT_DIM=4`,
  `SPREAD_HIDDEN_DIM=None`, and `disable_spread_conditioning`.
- Wire the conditioner into `pre_fusion_hook`; disabled path remains identity.
- Verify: P0/P1 smoke tests continue to pass unchanged.

**Step P2.4 — Price-relation sanity metric**
- Extend or reuse `lob_bench/cross_asset/` to report generated vs. real
  ETF-constituent NAV-gap / price-relation error alongside cross-correlation.
- Verify: synthetic arrays return deterministic finite errors.

**Step P2.5 — P2 smoke**
- File: `DeepMarket/tests/p2_spread_smoke.py`
- Checks:
  - raw `cond_lob` produces finite mid/spread/NAV-gap context;
  - no basket weights produces no-op;
  - `γ_spread = 0` is equivalent to P1;
  - one `ddpm_single_step` has finite loss and `γ_spread` receives gradient.

> **P2 exit criterion**: P0/P1/P2 smoke tests pass, P2 is off by default,
> enabling P2 with ETF basket metadata is trainable, and no-basket configs
> safely recover P1 behavior.

### Phase P3 — Annealed Energy Guidance

**Step P3.1 — Guidance schedule**
- File: `models/diffusers/multi_asset/arbitrage/guidance_schedule.py`
- Function: return zero guidance for high-noise steps and linearly ramp in
  the last `ENERGY_GUIDANCE_START_STEP` reverse steps.
- Verify: high `t` gives `0`, `t=0` gives `ENERGY_GUIDANCE_MAX_SCALE`.

**Step P3.2 — NAV-gap energy**
- File: `models/diffusers/multi_asset/arbitrage/energy.py`
- Behavior:
  - reconstruct `x_hat_0` from `(x_t, eps_fused, t)`;
  - read generated price proxy from `ENERGY_PRICE_PROXY_INDEX`;
  - combine with latest conditioning mid price;
  - compute ETF-basket Huber NAV-gap energy.
- Verify: reconstructed tensor shape matches `eps_fused`, energy/gaps are
  finite, and gradients with respect to `eps_fused` are finite.

**Step P3.3 — Guidance correction**
- File: `models/diffusers/multi_asset/arbitrage/energy.py`
- Behavior: compute a detached eps-space gradient correction and apply
  `eps_fused - scale(t) * clipped_grad`. Clip per-batch gradient norm by
  `ENERGY_GUIDANCE_MAX_GRAD_NORM`.
- Verify: disabled/no-basket/high-`t` paths are exact no-ops; enabled late-`t`
  path changes the noise estimate by a finite bounded amount.

**Step P3.4 — Persistence and validation interfaces**
- Files:
  - `models/diffusers/multi_asset/arbitrage/persistence_tracker.py`
  - `models/diffusers/multi_asset/arbitrage/consistency_check.py`
- Behavior: provide default-zero persistence multiplier and centralized
  basket/shape validation. Persistence state resets at the start of each
  DDPM/DDIM sample loop.
- Verify: default persistence weight leaves behavior unchanged.

**Step P3.5 — Wire post-fusion hook**
- Files: `configuration.py`, `ma_gaussian_diffusion.py`.
- Add defaults:
  - `ENERGY_GUIDANCE_ENABLED=False`
  - `ENERGY_GUIDANCE_START_STEP=10`
  - `ENERGY_GUIDANCE_MAX_SCALE=0.01`
  - `ENERGY_GUIDANCE_DELTA=1.0`
  - `ENERGY_GUIDANCE_MAX_GRAD_NORM=1.0`
  - `ENERGY_PRICE_PROXY_INDEX=5`
  - `ENERGY_PERSISTENCE_WEIGHT=0.0`
- Wire `ArbitrageEnergyGuidance` into `post_fusion_hook`; default configs and
  `disable_arb_guidance=True` recover P2.
- Verify: P0/P1/P2 smoke tests continue to pass.

**Step P3.6 — P3 smoke**
- File: `DeepMarket/tests/p3_energy_smoke.py`
- Checks:
  - `x_hat_0` reconstruction shape and finite values;
  - NAV-gap energy and gradient finite;
  - no-basket, disabled, and high-`t` no-op paths;
  - late-step guidance changes output and respects grad clipping;
  - one `ddpm_single_step` has finite loss.

> **P3 exit criterion**: P0/P1/P2/P3 smoke tests pass, P3 is off by default,
> no-basket configs safely recover P2 behavior, and enabled late-step guidance
> produces finite bounded corrections.

---

## 5. Open Items / Deferred

- **Score-net sharing strategy** — currently fully shared backbone +
  `asset_emb`. If marginal realism on the ETF vs. the constituent diverges
  noticeably in P0 evaluation, revisit by adding per-asset-class output
  heads.
- **N > 2 sparsification** — economic-prior + top-k. Not needed at N = 2.
- **Baselines (B/C/D)** — copula post-hoc, Hawkes, ABIDES. Decide after P1
  results.
- **P4 evaluation suite** — formalize cross-correlation, lead-lag, and NAV-gap
  metrics on held-out windows after P3 smoke stability is locked.
