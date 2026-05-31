"""Ablation switches for the multi-asset graph stack."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphAblationFlags:
    """Runtime switches used by P1/P2/P3 experiments."""

    disable_graph: bool = False
    freeze_edge_weights: bool = False
    disable_spread_conditioning: Optional[bool] = None
    disable_arb_guidance: bool = True
