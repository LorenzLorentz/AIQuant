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

    @classmethod
    def basis_pair(cls, reference: str, leg: str) -> "AssetUniverse":
        """Two economically tied legs whose mid-prices should track (basis ~ 0).

        Use for bucket-4 scheme A (spot vs perpetual) and scheme B (same coin on
        two exchanges). ``reference`` is index 0, ``leg`` is index 1. The single
        spread group ``reference - 1.0 * leg`` is exactly the basis, which feeds
        P3 ``compute_spread`` / the no-arb energy term.
        """
        return cls(
            assets=[reference, leg],
            relation_types={(0, 1): 0, (1, 0): 1},
            etf_basket_weights={0: {1: 1.0}},
        )

    @classmethod
    def cross_venue(cls, venues: List[str], *, reference_index: int = 0) -> "AssetUniverse":
        """Same instrument on N venues, all pinned to one another by arbitrage.

        ``venues`` are the per-venue asset names (index = order in the list).
        Every venue is related to every other (fully connected directed graph),
        and each non-reference venue forms a basis spread group against the
        reference venue (``venue_k - reference``), so P3 sees ``N-1`` groups.
        Enables the N>2 graph that BTC+ETH never exercised.
        """
        if len(venues) < 2:
            raise ValueError(f"cross_venue needs >=2 venues, got {venues!r}")
        n = len(venues)
        if not 0 <= reference_index < n:
            raise ValueError(f"reference_index {reference_index} outside 0..{n - 1}")
        relation_types: Dict[RelationKey, int] = {}
        rid = 0
        for j in range(n):
            for i in range(n):
                if i != j:
                    relation_types[(j, i)] = rid
                    rid += 1
        etf_basket_weights = {
            k: {reference_index: 1.0}
            for k in range(n)
            if k != reference_index
        }
        return cls(
            assets=list(venues),
            relation_types=relation_types,
            etf_basket_weights=etf_basket_weights,
        )

    @classmethod
    def etf_basket(
        cls,
        etf: str,
        constituents: List[str],
        weights: List[float] | None = None,
    ) -> "AssetUniverse":
        """ETF (index 0) priced against a weighted basket of constituents.

        Bucket-4 scheme C. The spread group ``etf - sum_i w_i * constituent_i``
        is the textbook NAV no-arbitrage residual that ``arbitrage/energy.py``
        was written for. ``weights`` defaults to equal weights summing to 1.
        """
        if not constituents:
            raise ValueError("etf_basket needs at least one constituent")
        m = len(constituents)
        if weights is None:
            weights = [1.0 / m] * m
        if len(weights) != m:
            raise ValueError(f"weights length {len(weights)} != constituents {m}")
        assets = [etf] + list(constituents)
        relation_types: Dict[RelationKey, int] = {}
        rid = 0
        for i in range(1, m + 1):
            relation_types[(0, i)] = rid
            rid += 1
            relation_types[(i, 0)] = rid
            rid += 1
        etf_basket_weights = {0: {i + 1: float(weights[i]) for i in range(m)}}
        return cls(
            assets=assets,
            relation_types=relation_types,
            etf_basket_weights=etf_basket_weights,
        )
