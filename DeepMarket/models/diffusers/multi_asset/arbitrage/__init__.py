"""P2/P3/P4 quasi-no-arbitrage helpers for multi-asset diffusion.

The P4 posterior diagnostics are numpy-only and should remain importable in
lightweight evaluation environments that do not have torch installed. P2/P3
helpers are exported when torch is available.
"""

from models.diffusers.multi_asset.arbitrage.consistency_check import (
    ConsistencyReference,
    check_stats,
    consistency_check,
    normalize_spread,
    reference_from_lob,
    reference_from_spread,
    spread_from_lob,
    spread_from_mid,
    summarize_spread_consistency,
)

__all__ = [
    "ConsistencyReference",
    "check_stats",
    "consistency_check",
    "normalize_spread",
    "reference_from_lob",
    "reference_from_spread",
    "spread_from_lob",
    "spread_from_mid",
    "summarize_spread_consistency",
]

try:
    import torch as _torch  # noqa: F401
except ModuleNotFoundError:
    _torch = None

if _torch is not None:
    from models.diffusers.multi_asset.arbitrage.activation_schedule import (
        default_k_spread_steps,
        spread_inject_active,
    )
    from models.diffusers.multi_asset.arbitrage.energy import (
        StressHead,
        calibrate_delta_base,
        dynamic_delta,
        energy,
        group_rolling_stats,
        phi,
        rho,
    )
    from models.diffusers.multi_asset.arbitrage.guidance_schedule import guidance_lambda
    from models.diffusers.multi_asset.arbitrage.persistence_tracker import (
        tau_to_tensor,
        update_tau,
    )
    from models.diffusers.multi_asset.arbitrage.reverse_loop_state import ReverseLoopState
    from models.diffusers.multi_asset.arbitrage.spread_computer import (
        broadcast_spread_to_assets,
        compute_spread,
        decode_mid_price,
        spread_groups,
    )

    __all__ += [
        "ReverseLoopState",
        "StressHead",
        "broadcast_spread_to_assets",
        "calibrate_delta_base",
        "compute_spread",
        "decode_mid_price",
        "default_k_spread_steps",
        "dynamic_delta",
        "energy",
        "group_rolling_stats",
        "guidance_lambda",
        "phi",
        "rho",
        "spread_groups",
        "spread_inject_active",
        "tau_to_tensor",
        "update_tau",
    ]
