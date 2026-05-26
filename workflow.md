# AIQuant Workflow — Multi-Asset Graph-Coupled Diffusion with Quasi-No-Arbitrage

This document covers the full method-side implementation plan across five
phases:

1. The problem we are solving.
2. The model architecture and how it deviates from single-asset TRADES.
3. The codebase layout — what is reused, what is added.
4. A stepwise build plan across phases:
   - **P0** — multi-asset shared backbone (no cross-asset signal).
   - **P1** — graph coupling inside the reverse diffusion step.
   - **P2** — spread-aware conditioning (Layer 1 of quasi-no-arbitrage).
   - **P3** — annealed energy guidance (Layer 2 of quasi-no-arbitrage).
   - **P4** — Layer 3 posterior check + cross-asset evaluation metrics.

P0 and P1 form a self-contained multi-asset stack; P2/P3 plug in through two
explicit hooks (`pre_fusion_hook`, `post_fusion_hook`) that we install in
P0 and leave as no-ops until P2/P3.

---

## 1. Problem Definition (scoped to P0 + P1)

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

### 1.3 What we are *not* doing in this document

- **No** sparsification of the graph. With N = 2 the graph is fully connected
  (2 directed edges); we keep the dense code path and defer top-k / economic
  sparsification to N > 2 work.
- **No** multivariate-Hawkes / copula / ABIDES baselines yet. Those are P5.
- **No** training-time arbitrage loss. The quasi-no-arbitrage mechanism in
  P2/P3 is **inference-time** (conditioning + classifier-free-style guidance);
  no extra loss term is added to the training objective.

### 1.4 Success criteria per phase

| Stage | What "done" means                                                                              |
| ----- | ---------------------------------------------------------------------------------------------- |
| P0    | Multi-asset shared-backbone diffusion trains end-to-end, per-asset loss curves look like TRADES. |
| P1    | Adding graph coupling: (a) loss does not regress; (b) ablation flag `disable_graph=True` recovers P0 exactly; (c) generated trajectories show non-trivial cross-asset correlation that the P0 baseline lacks. |
| P2    | Spread conditioning runs only in the last K reverse steps; (a) `disable_spread_cond=True` recovers P1; (b) the generated spread distribution is closer to the historical one than P1's. |
| P3    | Energy guidance is applied via `post_fusion_hook`; (a) `disable_arb_guidance=True` recovers P2; (b) the four-corner ablation grid runs; (c) persistent (long-duration) spread excursions are reduced vs. P2 without flattening the spread distribution to zero. |
| P4    | Cross-asset metrics (corr matrix, CCF, lead–lag error, spillover) and Layer 3 posterior consistency check produce a JSON report comparing P0/P1/P2/P3 on held-out data. |

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

### 2.4 What is reused unchanged

- `GaussianDiffusion.forward_reparametrized` — forward (noising) process.
- The β schedule, `α_t`, `ᾱ_t`, posterior coefficients, EMA, type embedder.
- The hybrid loss (`L_simple + λ·L_vlb`) — applied per asset and summed.
- `DDPM` sampler (we stay with DDPM in P0/P1 for simplicity; DDIM can be
  added later by porting the same fusion hook into `ddim_single_step`).

### 2.5 Training objective

Per-asset hybrid loss, summed:

```
L_total = Σ_{i=1}^{N}  L_hybrid^(i)
        = Σ_{i=1}^{N}  ( L_simple^(i) + λ · L_vlb^(i) )
```

No auxiliary cross-asset loss in P0/P1. The graph parameters are trained
purely through the per-asset reconstruction gradient. If `γ` stays near zero
through training, it is a signal that the graph is unhelpful and we should
investigate before adding P2/P3.

### 2.6 P2 — Spread-aware conditioning (Layer 1 of quasi-no-arbitrage)

The cross-asset spread implied by the current noisy state is fed back into the
denoiser as an auxiliary conditioning signal at the **next** reverse step,
so the model is aware of its own evolving dislocation. Inference-time only —
training-time inputs are unchanged.

#### 2.6.1 Recovering a usable mid-price from `x_t`

The mid-price is a property of the (clean) LOB state, not directly of the
noisy latent. We use the standard one-shot estimate:

```
x̂_0^(i)(x_t, eps_fused) = ( x_t - sqrt(1-ᾱ_t) · eps_fused ) / sqrt(ᾱ_t)
```

From `x̂_0` (in the same space as `x_0`, i.e. order events with relative
price `Δp`) and the conditioning LOB state `cond_lob`, we reconstruct a
mid-price estimate `m̂_t^(i)`. Concretely: take the last LOB snapshot in
`cond_lob`, apply the predicted `Δp` from `x̂_0` to it (using the same
reverse mapping LOBSTER uses), and read off the new mid.

Two caveats:
- At high `t`, `eps_fused` is dominated by noise and `x̂_0` is unreliable.
- The mid-price decode depends on the cond LOB state, which is fixed across
  the reverse loop and so contributes no noise.

Both are addressed by the activation schedule below.

#### 2.6.2 Implied cross-asset spread

For the ETF group `g = (E, {S_k})` with weights `ω_k`:

```
spread_t^(g) = m̂_t^(E) - Σ_k ω_k · m̂_t^(S_k)
```

For N = 2 this collapses to `m̂_t^(E) - ω · m̂_t^(S)` with one scalar `ω`.
The weight is read from `AssetUniverse.etf_basket_weights`, which we declared
in P0.1 with empty content and populate now.

#### 2.6.3 Activation schedule

```
spread_inject_active(t) := (t < K_spread_steps)
```

Default `K_spread_steps = num_diffusionsteps // 4`. Above the threshold the
spread feature is zero and the network sees a "no spread information" cond.
This keeps both `x̂_0` and the energy gradient (P3) well-behaved at high noise.

#### 2.6.4 Spread injection into the score net

`shared_score_net` is extended to accept an optional scalar `spread_cond ∈ ℝ`
(per group, broadcast to assets in the group):

```
shared_score_net(x_t, cond_orders, t, cond_lob, asset_id, spread_cond)
                                                          ^^^^^^^^^^^
                                                          new in P2
```

Internally, `spread_cond` is mapped through a small MLP (`ℝ → ℝ^{F_in}`) and
**added** to the asset-embedding token introduced in P0.3. The TRADES
backbone is untouched.

#### 2.6.5 The pre_fusion_hook becomes non-trivial

In `ma_gaussian_diffusion.py`:

```
def pre_fusion_hook(x_t, state):
    if not spread_inject_active(t):
        return None
    x̂_0    = (x_t - sqrt(1-ᾱ_t) · state.last_eps_fused) / sqrt(ᾱ_t)
    m̂      = decode_mid_price(x̂_0, cond_lob)
    spread = compute_spread(m̂, asset_universe)
    return spread
```

The returned spread is stored in `state` and used by the next step's
`shared_score_net` call. On the very first reverse step there is no
`last_eps_fused`, so spread is treated as zero.

### 2.7 P3 — Annealed energy guidance (Layer 2 of quasi-no-arbitrage)

A classifier-free-style guidance term whose strength is annealed across the
reverse schedule and modulated by the prevailing market regime.

#### 2.7.1 The energy function

```
E_arb(x_t) = Σ_g  ρ( |spread^(g)_t|, δ_dyn(t) )  ·  φ( τ^(g)_t )

  ρ(u, δ) = max(0, u - δ)²                      — hinge-quadratic
  φ(τ)    = τ                                    — linear, default
  δ_dyn(t)= δ_base · (1 + κ · stress(x_t))      — state-dependent tolerance
```

`stress(x_t)` is a scalar derived from the P1 rolling-stats vector (high
volatility / high cancellation rate ⇒ large stress ⇒ wider tolerance band).
For the first cut: a small linear head over the same `rolling_stats` we
already compute in P1, so there is no new feature-engineering cost.

`δ_base` is calibrated **once** from the historical training set as the
median `|spread|`. It is a non-trainable buffer.

`κ` and `p` (in `λ(t)` below) are hyperparameters. Default `κ = 1.0`.

#### 2.7.2 Persistence tracker `τ`

`τ^(g)_t` counts consecutive reverse steps (down through `t`) for which
group `g` has been outside its tolerance band. It is **state** carried across
reverse steps:

```
class ReverseLoopState:
    last_eps_fused: Tensor          # P2 already needs this
    tau: Dict[group_id, int]        # P3 adds this
```

When `spread_inject_active(t)` is False (high noise), `tau` is held at 0;
counting only begins when spread becomes meaningful.

#### 2.7.3 Guidance schedule

```
λ(t) = λ_max · (1 - ᾱ_t)^p
```

Small at high noise, growing as `t → 0`. Default `λ_max = 1.0`, `p = 2`.

#### 2.7.4 The post_fusion_hook becomes non-trivial

```
def post_fusion_hook(eps_fused, x_t, state):
    if not arb_guidance_active(t):
        return eps_fused
    spread = state.last_spread                      # set by pre_fusion_hook
    update_tau(state.tau, spread, δ_dyn(t))
    E      = energy(spread, δ_dyn(t), state.tau)
    grad   = autograd.grad(E, x_t, retain_graph=False)
    return eps_fused + λ(t) · sqrt(1-ᾱ_t) · grad
```

Because `spread` is computed from `x_t` via `x̂_0` (which depends on
`last_eps_fused`, treated as a constant for this gradient), `∇_{x_t} E_arb`
flows only through the mid-price decoder — cheap, no backprop through the
score net.

For simplicity `arb_guidance_active(t) := spread_inject_active(t)` — guidance
turns on at the same threshold as spread conditioning.

#### 2.7.5 Ablation grid

`disable_arb_guidance=True` ⇒ `post_fusion_hook` returns `eps_fused`
unchanged. Combined with P1's `disable_graph` and P2's `disable_spread_cond`:

| `disable_graph` | `disable_spread_cond` | `disable_arb_guidance` | Recovers           |
| --------------- | --------------------- | ---------------------- | ------------------ |
| T               | T                     | T                      | P0 (shared-only)   |
| F               | T                     | T                      | P1 (graph)         |
| F               | F                     | T                      | P2 (graph+cond)    |
| F               | F                     | F                      | P3 (full)          |
| T               | F                     | F                      | "arb without graph" — diagnostic |

### 2.8 P4 — Layer 3 posterior check + cross-asset evaluation

Layer 3 is a **diagnostic** on full generated trajectories; it does not
modify generation. Cross-asset metrics live alongside it in
`lob_bench/cross_asset/`.

#### 2.8.1 Layer 3 posterior consistency check

For each asset group `g` and each generated trajectory of length T:

- Compute `spread^(g)_τ` for `τ = 1..T` in **wall-clock time** (not
  diffusion steps).
- Summary stats:
  - mean and std of `|spread|`
  - mean and tail (95%/99%) duration of excursions outside `δ_base`
  - tail (top 1%, top 0.1%) of `|spread|`
  - correlation of `|spread|` with rolling realized volatility
- Compare against the same statistics on the historical training set.
- Output: a JSON report + a boolean pass/fail per statistic at a configured
  tolerance band.

#### 2.8.2 Cross-asset metrics

| Metric                | What it measures                                                  |
| --------------------- | ----------------------------------------------------------------- |
| Realized corr matrix  | Pearson corr of mid-price log-returns at a fixed sampling rate.   |
| Cross-correlation fn  | `corr(r_t^(1), r_{t+ℓ}^(2))` over `ℓ ∈ [-L, L]`.                 |
| Lead–lag error        | L2 distance between peak-lag location in real vs. generated CCF.  |
| Conditional spillover | `E[ |r^(2)_{t+1}|  |  |r^(1)_t| > q ]` for high quantiles `q`.   |

The cross-correlation sanity check from P1.9 graduates here to a proper
metric.

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
│   ├── AssetUniverse.py                    [NEW] P0
│   └── MultiAssetLOBDataset.py             [NEW] P0
│
├── models/diffusers/
│   ├── TRADES/                             [unchanged]
│   ├── gaussian_diffusion.py               [unchanged]
│   ├── diffusion_engine.py                 [unchanged]   ← baseline (A) entry
│   │
│   └── multi_asset/                        [NEW]
│       ├── __init__.py
│       ├── ma_diffusion_engine.py          [NEW] P0  MultiAssetDiffusionEngine (Lightning)
│       ├── ma_gaussian_diffusion.py        [NEW] P0  reverse loop + pre/post fusion hooks
│       ├── shared_score_net.py             [NEW] P0  TRADES + asset emb; P2 adds spread cond
│       ├── ablation_flags.py               [NEW] P1  disable_graph/disable_spread_cond/disable_arb_guidance
│       │
│       ├── graph/                          [NEW in P1]
│       │   ├── __init__.py
│       │   ├── rolling_stats.py            [NEW] per-asset window statistics
│       │   ├── relation_embedding.py       [NEW] r_{ji}
│       │   ├── edge_weight_net.py          [NEW] w_{ji}(t)
│       │   ├── message_passing.py          [NEW] f_φ
│       │   ├── aggregator.py               [NEW] attention aggregation
│       │   └── noise_fusion.py             [NEW] g_ψ with learnable γ
│       │
│       └── arbitrage/                      [NEW in P2/P3/P4]
│           ├── __init__.py
│           ├── reverse_loop_state.py       [NEW] P2  last_eps_fused, last_spread, tau
│           ├── spread_computer.py          [NEW] P2  x̂_0 → mid-price → spread
│           ├── activation_schedule.py      [NEW] P2  spread/guidance on-off threshold
│           ├── persistence_tracker.py      [NEW] P3  τ update
│           ├── energy.py                   [NEW] P3  ρ, φ, δ_dyn, stress(x_t)
│           ├── guidance_schedule.py        [NEW] P3  λ(t)
│           └── consistency_check.py        [NEW] P4  Layer 3 diagnostic
│
└── evaluation/quantitative_eval/
    └── posterior_report.py                 [NEW] P4  driver that runs consistency_check + emits JSON

lob_bench/cross_asset/                      [NEW in P4]
├── __init__.py
├── cross_corr.py                           [NEW] P1.9 sanity → P4 metric
├── lead_lag.py                             [NEW] P4
├── arbitrage_metrics.py                    [NEW] P4
└── spillover.py                            [NEW] P4 (optional)
```

### 3.2 Files we deliberately do **not** touch in P0/P1

- `models/diffusers/TRADES/TRADES.py` — the score net itself.
- `models/diffusers/gaussian_diffusion.py` — the single-asset diffuser remains
  intact so baseline (A) keeps working.
- `models/diffusers/diffusion_engine.py` — the single-asset Lightning module.
- `models/gan/`, `models/feature_augmenters/`.

The multi-asset stack is parallel to the existing single-asset stack, not a
modification of it.

### 3.3 Extension points

The multi-asset diffuser exposes two hooks that are no-ops in P0/P1 and gain
behavior in P2/P3:

- `pre_fusion_hook(x_t, state) → spread` — wired in P2 to compute the
  implied cross-asset spread from `x̂_0` and inject it into the next
  `shared_score_net` call.
- `post_fusion_hook(eps_fused, x_t, state) → eps_guided` — wired in P3 to
  add the energy-gradient guidance term.

A small `ReverseLoopState` dataclass (`arbitrage/reverse_loop_state.py`)
carries `last_eps_fused`, `last_spread`, and `tau` between reverse steps;
P0/P1 only use the `last_eps_fused` field (and only because it costs
nothing).

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

### Phase P2 — Spread-aware conditioning

P2 is **inference-time only**. Training does not change; we only modify the
sampling path. The model trained in P1 can be re-used for P2 evaluation
without retraining, though we will also train a fresh P2 model where
`spread_cond` is dropped out (cond_dropout-style) so the network learns to
use it.

**Step P2.1 — Reverse loop state**
- File: `models/diffusers/multi_asset/arbitrage/reverse_loop_state.py`
- Dataclass `ReverseLoopState` with fields `last_eps_fused`, `last_spread`,
  `tau` (dict). Built once per generation batch and updated in-place across
  reverse steps.
- Verify: round-trip construction + update, no allocation in inner loop.

**Step P2.2 — Spread computer**
- File: `models/diffusers/multi_asset/arbitrage/spread_computer.py`
- Two functions:
  - `decode_mid_price(x̂_0, cond_lob) -> (B, N)` — apply the predicted
    `Δp` from `x̂_0` to the last LOB snapshot to estimate the new mid.
    Reuses the inverse normalization that `LOBSTERDataBuilder` uses
    forward.
  - `compute_spread(mid, asset_universe) -> (B, num_groups)` — for each
    group, `mid_E - Σ ω_k · mid_{S_k}`.
- Verify: at `x̂_0 = x_0` ground truth, recovered mid matches the LOB
  snapshot's mid up to LOBSTER tick rounding; spread on real data has the
  same magnitude as a manual reference calculation.

**Step P2.3 — Activation schedule**
- File: `models/diffusers/multi_asset/arbitrage/activation_schedule.py`
- `spread_inject_active(t, num_diffusionsteps, K_spread_steps)`
- Default `K_spread_steps = num_diffusionsteps // 4`.
- Verify: returns False above threshold, True below.

**Step P2.4 — Extend `shared_score_net` with `spread_cond` input**
- File: `models/diffusers/multi_asset/shared_score_net.py` (extend)
- Add optional `spread_cond: (B, N) or None`. When `None`, behavior is
  bit-identical to P1. When provided, route through a small MLP
  (`ℝ → ℝ^{F_in}`) and add to the asset-embedding token.
- Verify: `spread_cond=None` path matches P1 outputs exactly; non-`None`
  path produces a different output. Gradient flows back to the new MLP.

**Step P2.5 — Wire `pre_fusion_hook`**
- File: `models/diffusers/multi_asset/ma_gaussian_diffusion.py` (extend)
- Replace the no-op `pre_fusion_hook` with the body from §2.6.5.
- On step T (first reverse step), spread is treated as 0 (no
  `last_eps_fused` yet).
- Verify: with `disable_spread_cond=True`, the path is bit-identical to
  P1; with it False, the path differs only after step `num_diffusionsteps - K_spread_steps`.

**Step P2.6 — Ablation flag**
- File: `models/diffusers/multi_asset/ablation_flags.py` (extend)
- Add `disable_spread_cond: bool`. When True, `pre_fusion_hook` returns
  `None` regardless of `t`.
- Verify: with flag set, P1 outputs are reproduced exactly.

**Step P2.7 — Optional: retrain with spread-cond dropout**
- Change in `ma_diffusion_engine.py`: during training, occasionally feed
  `spread_cond` computed from the *training* ground truth (with prob
  `1 - cond_dropout_prob`) and zero otherwise.
- Without this step, P2 still works at inference but the network has not
  learned to use `spread_cond` informatively — useful as an ablation
  ("P2 without retraining") but not as the main P2 result.
- Verify: per-asset loss at convergence is within noise of P1's.

**Step P2.8 — Sanity check: generated spread distribution**
- Quick script in `evaluation/quantitative_eval/` (no new metric file
  needed yet — that comes in P4).
- Compare KDE of generated `|spread|` from P1 vs. P2 vs. real, on a
  held-out window.
- Verify: P2's KDE is visually closer to real than P1's; means and tails
  in the same order.

> **P2 exit criterion**: P2.8 prefers P2 over P1, ablation toggle exactly
> reproduces P1, and the spread-conditioning code path costs less than 10%
> additional wall-time per reverse step (since it activates only in the
> last K).

### Phase P3 — Annealed energy guidance

Also **inference-time only**. Same trained weights as P2 are used.

**Step P3.1 — Persistence tracker**
- File: `models/diffusers/multi_asset/arbitrage/persistence_tracker.py`
- `update_tau(state.tau, spread, delta_dyn)`: for each group, if
  `|spread^(g)| > δ_dyn^(g)`, increment `tau[g]`; else reset to 0.
- Resets `tau` to 0 when `spread_inject_active(t)` is False.
- Verify: a synthetic spread sequence produces the expected `τ` trajectory
  (e.g., spread spike of length 5 ⇒ `τ` hits 5 and resets).

**Step P3.2 — Energy function**
- File: `models/diffusers/multi_asset/arbitrage/energy.py`
- `rho(u, delta) = relu(u - delta) ** 2`
- `phi(tau) = tau` (default, linear; signature kept so we can swap in a
  monotone schedule later)
- `delta_dyn(t, rolling_stats, delta_base, kappa) = delta_base * (1 + kappa * stress)`,
  with `stress` a small linear head over `rolling_stats`.
- `energy(spread, delta_dyn, tau) = Σ_g rho(|spread^(g)|, delta_dyn^(g)) * phi(tau^(g))`
- `delta_base` is a calibrated buffer — populate from training-set median
  `|spread|` on first instantiation and persist with the checkpoint.
- Verify: gradient w.r.t. spread is zero inside the tolerance band; nonzero
  and quadratic outside.

**Step P3.3 — Guidance schedule**
- File: `models/diffusers/multi_asset/arbitrage/guidance_schedule.py`
- `lambda(t, alpha_bar, lambda_max, p) = lambda_max * (1 - alpha_bar_t) ** p`
- Default `lambda_max = 1.0`, `p = 2`.
- Verify: monotone decreasing in `ᾱ_t` (so increasing in noise level
  removed — i.e. smaller at high `t`).

**Step P3.4 — Wire `post_fusion_hook`**
- File: `models/diffusers/multi_asset/ma_gaussian_diffusion.py` (extend)
- Replace the no-op `post_fusion_hook` with the body from §2.7.4.
- Important: `x_t.requires_grad_(True)` for the duration of the spread
  computation and energy gradient, then detach for the standard DDPM update.
- `state.last_spread` must be set by `pre_fusion_hook` before this hook fires.
- Verify: at the four ablation corners (see §2.7.5), behavior matches
  expectations bit-by-bit for the three "disable" combinations.

**Step P3.5 — Ablation flag**
- File: `models/diffusers/multi_asset/ablation_flags.py` (extend)
- Add `disable_arb_guidance: bool`. When True, `post_fusion_hook` returns
  `eps_fused` unchanged.
- Verify: with flag True, P2 outputs are reproduced exactly.

**Step P3.6 — Sanity check: spread persistence distribution**
- Compare the empirical distribution of excursion durations
  (`|spread| > δ_base`) across P1, P2, P3, and real.
- Verify: P3 reduces long-tail excursions relative to P2, while the bulk of
  the spread distribution (small/short excursions) is preserved.

> **P3 exit criterion**: ablation grid runs cleanly, tail of excursion
> durations is reduced vs. P2 without collapsing the central spread
> distribution, and the four-corner table from §2.7.5 is produced as a
> diagnostic.

### Phase P4 — Layer 3 posterior check + cross-asset evaluation

**Step P4.1 — Layer 3 consistency check**
- File: `models/diffusers/multi_asset/arbitrage/consistency_check.py`
- Input: a generated trajectory (`(T, N, ...)` of orders+LOB),
  AssetUniverse, and a reference statistic set computed once from training
  data.
- Output: dict of summary stats (see §2.8.1) and pass/fail booleans.
- Verify: on real data, all stats pass; on a degenerate trajectory
  (`spread ≡ 0`), at least the variance and tail stats fail.

**Step P4.2 — Cross-correlation metric**
- File: `lob_bench/cross_asset/cross_corr.py`
- Functions:
  - `realized_corr(returns_1, returns_2, dt) -> float`
  - `ccf(returns_1, returns_2, max_lag) -> ndarray`
- Mid-price returns are sampled at a configurable bar size.
- Verify: white-noise inputs produce ~0 corr and a flat CCF; a lagged
  copy produces the expected peak.

**Step P4.3 — Lead-lag error**
- File: `lob_bench/cross_asset/lead_lag.py`
- `lead_lag_error(real_ccf, gen_ccf) -> float` — L2 (or peak-position
  L1) between peak-lag locations.
- Verify: on identical inputs returns 0; on inputs with shifted peaks
  returns the shift.

**Step P4.4 — Arbitrage metrics**
- File: `lob_bench/cross_asset/arbitrage_metrics.py`
- Aggregates: empirical distribution of `spread` (KS distance vs. real),
  excursion-duration tail, conditional-on-stress spread mean.
- Verify: on real-vs-real (different windows of the same data) the KS
  statistic is small; on real-vs-shuffled the statistic is large.

**Step P4.5 — Spillover (optional)**
- File: `lob_bench/cross_asset/spillover.py`
- `E[|r^(2)_{t+1}| | |r^(1)_t| > q]` at a few high quantiles `q`.
- Verify: returns a finite number on real data; spillover on shuffled
  data is approximately the unconditional mean.

**Step P4.6 — Posterior report driver**
- File: `DeepMarket/evaluation/quantitative_eval/posterior_report.py`
- Wires P4.1 + P4.2 + P4.3 + P4.4 (+ optionally P4.5) into a single
  evaluation pass. Runs once per ablation-corner checkpoint (P0/P1/P2/P3
  + the diagnostic "arb without graph"). Emits a JSON report.
- Verify: report compiles for at least one full corner end-to-end on a
  small held-out window.

> **P4 exit criterion**: the JSON report compares all four ablation
> corners (plus real-data reference values) across all four metric
> groups, and the trends match the per-phase exit criteria (P1 > P0 in
> corr, P2 > P1 in spread distribution closeness, P3 > P2 in excursion
> tails).

---

## 5. Open Items / Deferred

- **Score-net sharing strategy** — currently fully shared backbone +
  `asset_emb`. If marginal realism on the ETF vs. the constituent diverges
  noticeably in P0 evaluation, revisit by adding per-asset-class output
  heads.
- **`φ(τ)` shape** — defaulting to linear in P3. If the excursion-tail
  reduction in P3.6 is too aggressive (collapses the central spread
  distribution), swap to a soft-plus or capped variant.
- **`K_spread_steps`, `λ_max`, `p`, `κ` tuning** — set at first-cut defaults
  in §2.6.3 / §2.7. Sweep only if P2/P3 exit criteria are missed.
- **Calibration of `δ_base`** — currently the training-set median `|spread|`.
  Per-regime `δ_base` (e.g. higher in stressed regimes) is a possible
  extension if `δ_dyn(t)`'s state modulation is insufficient.
- **N > 2 sparsification** — economic-prior + top-k. Not needed at N = 2.
- **Baselines (B/C/D)** — copula post-hoc, Hawkes, ABIDES. Decide after P3
  results.
- **DDIM port of the pre/post-fusion hooks** — P0–P3 currently target DDPM
  sampling. Porting the same hooks to `ddim_single_step` is straightforward
  but deferred until DDPM results are in.
