"""Ablation switches for the multi-asset diffusion stack.

The flags live in a small dataclass so tests, scripts, and configs can pass
them around without coupling to Lightning or argparse.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AblationFlags:
    """Runtime switches used by P1+ graph coupling.

    disable_graph:
        Skip graph message passing and return the local score estimate exactly.
    freeze_edge_weights:
        Stop gradients from flowing through relation embeddings and edge MLP.
    disable_spread_cond:
        Skip P2 spread conditioning and call the score net exactly as P1 did.
    disable_arb_guidance:
        Reserved for P3. Present now so config files do not need another
        migration when arbitrage guidance is added.
    """

    disable_graph: bool = False
    freeze_edge_weights: bool = False
    disable_spread_cond: bool = False
    disable_arb_guidance: bool = False

    @classmethod
    def from_config(cls, config) -> "AblationFlags":
        """Build flags from either a structured or legacy config object."""
        structured = getattr(config, "ABLATION_FLAGS", None)
        if isinstance(structured, cls):
            return structured
        if isinstance(structured, dict):
            return cls(
                disable_graph=bool(structured.get("disable_graph", False)),
                freeze_edge_weights=bool(structured.get("freeze_edge_weights", False)),
                disable_spread_cond=bool(structured.get("disable_spread_cond", False)),
                disable_arb_guidance=bool(structured.get("disable_arb_guidance", False)),
            )

        legacy = getattr(config, "GRAPH_ABLATIONS", None)
        if isinstance(legacy, cls):
            return legacy
        if isinstance(legacy, dict):
            return cls(
                disable_graph=bool(legacy.get("disable_graph", False)),
                freeze_edge_weights=bool(legacy.get("freeze_edge_weights", False)),
                disable_spread_cond=bool(legacy.get("disable_spread_cond", False)),
                disable_arb_guidance=bool(legacy.get("disable_arb_guidance", False)),
            )

        return cls(
            disable_graph=bool(getattr(config, "DISABLE_GRAPH", False)),
            freeze_edge_weights=bool(getattr(config, "FREEZE_EDGE_WEIGHTS", False)),
            disable_spread_cond=bool(getattr(config, "DISABLE_SPREAD_COND", False)),
            disable_arb_guidance=bool(getattr(config, "DISABLE_ARB_GUIDANCE", False)),
        )
