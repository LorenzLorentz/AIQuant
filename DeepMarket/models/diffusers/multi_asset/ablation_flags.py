"""Ablation switches for the multi-asset graph stack."""

from dataclasses import dataclass


@dataclass
class GraphAblationFlags:
    """Runtime switches used by P1/P3 experiments.

    ``disable_arb_guidance`` is reserved for P3 so experiment configs can
    stabilize around one flag container before arbitrage guidance exists.
    """

    disable_graph: bool = False
    freeze_edge_weights: bool = False
    disable_arb_guidance: bool = True
