"""AssetUniverse — declarative description of the set of assets participating
in a multi-asset run, together with their pairwise economic relations.

Used by the multi-asset stack (P0/P1) but kept intentionally light so that
P2 (spread / ETF-basket NAV) and P3 (energy guidance) can attach extra
metadata without breaking older configs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


RelationKey = Tuple[int, int]  # (j -> i) directed edge between asset indices


@dataclass
class AssetUniverse:
    """Container describing the N assets used in a multi-asset training run.

    Attributes
    ----------
    assets:
        Ordered list of stock ticker names. Their index in this list is the
        canonical asset id used everywhere downstream (asset embedding lookup,
        per-asset stat indexing, etc.).
    relation_types:
        Map from a directed pair ``(j, i)`` of asset indices to an integer
        relation-type id. Used as the lookup key for ``RelationEmbedding`` in
        P1. For an ETF + constituent pair we use::

            {(0, 1): 0,   # ETF -> constituent
             (1, 0): 1}   # constituent -> ETF
    etf_basket_weights:
        Reserved for P2 (spread / NAV computation). Map from ETF asset index
        to ``{constituent_index: weight}``. Safe to leave empty in P0/P1.
    """

    assets: List[str]
    relation_types: Dict[RelationKey, int] = field(default_factory=dict)
    etf_basket_weights: Dict[int, Dict[int, float]] = field(default_factory=dict)

    @property
    def num_assets(self) -> int:
        return len(self.assets)

    @property
    def num_relation_types(self) -> int:
        if not self.relation_types:
            return 0
        return max(self.relation_types.values()) + 1

    def directed_edges(self) -> List[RelationKey]:
        """Return all directed edges (j -> i) registered in relation_types,
        in deterministic order. For N=2 this is ``[(0,1), (1,0)]``."""
        return sorted(self.relation_types.keys())

    def relation_id(self, j: int, i: int) -> int:
        return self.relation_types[(j, i)]

    @classmethod
    def etf_pair(cls, etf: str, constituent: str) -> "AssetUniverse":
        """Convenience constructor for the initial ETF + constituent setup."""
        return cls(
            assets=[etf, constituent],
            relation_types={(0, 1): 0, (1, 0): 1},
            etf_basket_weights={0: {1: 1.0}},
        )
