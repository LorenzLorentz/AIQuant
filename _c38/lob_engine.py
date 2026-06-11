"""Minimal price-time-priority LOB matching engine for lob_bench eval.

Standalone (no repo deps) so it can be unit-tested off-cluster.

Purpose: turn a *stream of decoded orders* (limit / cancel / market-execution)
into (1) evolving L2 book snapshots and (2) per-message order_ids, so that
  - the generated book becomes a genuine function of the generated messages
    (book metrics: spread/imbalance/volume/ofi stop being trivially equal to
    the real book), and
  - cancels/executions reference real order_ids of resting orders, so
    log_time_to_cancel / *_ask_order_depth / ask_cancellation_depth stop being
    degenerate (1.0).

Order convention (decoded, raw units):
  etype: 1 = new limit order, 3 = cancel/delete, 4 = market order (-> execution
         against the opposite side, mirroring ABIDES WorldAgent).
  direction: +1 = buy side (bids), -1 = sell side (asks).
  price: absolute raw price (the crypto adapter has NO usable `depth` field --
         it is all-zero -- so prices are placed absolutely, snapped to `tick`).

Prices are quantized to integer ticks internally; the L2 snapshot returns raw
(tick * price_in_ticks) prices so downstream scaling is the caller's choice.
"""
from __future__ import annotations
from collections import deque, OrderedDict


class MatchingEngine:
    def __init__(self, tick: float, n_levels: int = 10):
        self.tick = float(tick)
        self.n_levels = n_levels
        # price_tick(int) -> deque[[order_id, size]]   (FIFO within a level)
        self.bids: dict[int, deque] = {}
        self.asks: dict[int, deque] = {}
        # order_id -> (side, price_tick)  for cancel/exec lookup
        self.oid: dict[int, tuple] = {}
        self.next_id = 1
        # diagnostics
        self.n_limit = self.n_cancel = self.n_exec = 0
        self.n_cross = self.n_skip = 0

    # ---- helpers -------------------------------------------------------
    def _pt(self, price: float) -> int:
        return int(round(price / self.tick))

    def _new_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def best_bid_pt(self):
        return max(self.bids) if self.bids else None

    def best_ask_pt(self):
        return min(self.asks) if self.asks else None

    def seed_from_l2(self, l2_raw):
        """Seed the book from a raw 40-col L2 snapshot:
        [ask_px, ask_sz, bid_px, bid_sz] * n_levels. Each nonzero level becomes
        a single resting order with a fresh id (so later cancels can hit it)."""
        for lvl in range(self.n_levels):
            ap, asz, bp, bsz = l2_raw[4 * lvl:4 * lvl + 4]
            if ap > 0 and asz > 0:
                pt = self._pt(ap)
                oid = self._new_id()
                self.asks.setdefault(pt, deque()).append([oid, float(asz)])
                self.oid[oid] = ("A", pt)
            if bp > 0 and bsz > 0:
                pt = self._pt(bp)
                oid = self._new_id()
                self.bids.setdefault(pt, deque()).append([oid, float(bsz)])
                self.oid[oid] = ("B", pt)

    # ---- core book ops -------------------------------------------------
    def _remove_oid(self, oid):
        if oid not in self.oid:
            return
        side, pt = self.oid.pop(oid)
        book = self.bids if side == "B" else self.asks
        q = book.get(pt)
        if not q:
            return
        for j, (o, _s) in enumerate(q):
            if o == oid:
                del q[j]
                break
        if not q:
            book.pop(pt, None)

    def _consume(self, book_side: str, size: float):
        """Execute `size` against `book_side` ('B' or 'A') from the best price.
        Returns (order_id, price_raw) of the first resting order hit."""
        book = self.bids if book_side == "B" else self.asks
        first_hit = 0
        first_px = 0.0
        remaining = size
        while remaining > 1e-12 and book:
            pt = max(book) if book_side == "B" else min(book)
            q = book[pt]
            while remaining > 1e-12 and q:
                oid, sz = q[0]
                if first_hit == 0:
                    first_hit = oid
                    first_px = pt * self.tick
                take = min(sz, remaining)
                sz -= take
                remaining -= take
                if sz <= 1e-9:
                    q.popleft()
                    self.oid.pop(oid, None)
                else:
                    q[0][1] = sz
            if not q:
                book.pop(pt, None)
        return first_hit, first_px

    def _nearest_order(self, side: str, pt: int, size: float):
        """Find the resting order on `side` to cancel: nearest price level to
        `pt`, then the resting order with size closest to `size`."""
        book = self.bids if side == "B" else self.asks
        if not book:
            return None
        best_pt = min(book, key=lambda p: (abs(p - pt), -p if side == "B" else p))
        q = book[best_pt]
        j = min(range(len(q)), key=lambda k: abs(q[k][1] - size))
        return q[j][0]

    def step(self, etype: int, price: float, size: float, direction: int):
        """Apply one decoded order; return (order_id, price_raw) to write in the
        message. For a limit the price is where it rests; for a cancel/execution
        it is the price of the resting order actually hit (LOBSTER convention:
        a delete/exec message carries the price of the affected order)."""
        if size <= 0:
            self.n_skip += 1
            return 0, price
        side = "B" if direction >= 0 else "A"

        if etype == 1:  # new limit order
            pt = self._pt(price)
            ba, bb = self.best_ask_pt(), self.best_bid_pt()
            # marketable? buy at/above best ask, or sell at/below best bid -> cross
            if side == "B" and ba is not None and pt >= ba:
                self.n_cross += 1
                return self._consume("A", size)
            if side == "A" and bb is not None and pt <= bb:
                self.n_cross += 1
                return self._consume("B", size)
            oid = self._new_id()
            book = self.bids if side == "B" else self.asks
            book.setdefault(pt, deque()).append([oid, float(size)])
            self.oid[oid] = (side, pt)
            self.n_limit += 1
            return oid, pt * self.tick

        if etype == 3:  # cancel / delete
            pt = self._pt(price)
            oid = self._nearest_order(side, pt, size)
            if oid is None:
                self.n_skip += 1
                return 0, price
            _side, hit_pt = self.oid[oid]
            self._remove_oid(oid)
            self.n_cancel += 1
            return oid, hit_pt * self.tick

        if etype == 4:  # market order -> execution of the opposite side
            # model emits a market order with `direction`; it executes against
            # the opposite book (a buy market consumes asks).
            opp = "A" if side == "B" else "B"
            self.n_exec += 1
            return self._consume(opp, size)

        self.n_skip += 1
        return 0, price

    # ---- snapshot ------------------------------------------------------
    def l2(self):
        """Return a raw 40-col L2 snapshot [ask_px, ask_sz, bid_px, bid_sz]*N."""
        out = [0.0] * (4 * self.n_levels)
        ask_pts = sorted(self.asks)[:self.n_levels]
        bid_pts = sorted(self.bids, reverse=True)[:self.n_levels]
        for lvl in range(self.n_levels):
            if lvl < len(ask_pts):
                pt = ask_pts[lvl]
                out[4 * lvl] = pt * self.tick
                out[4 * lvl + 1] = sum(s for _o, s in self.asks[pt])
            if lvl < len(bid_pts):
                pt = bid_pts[lvl]
                out[4 * lvl + 2] = pt * self.tick
                out[4 * lvl + 3] = sum(s for _o, s in self.bids[pt])
        return out

    def mid(self):
        ba, bb = self.best_ask_pt(), self.best_bid_pt()
        if ba is None or bb is None:
            return None
        return 0.5 * (ba + bb) * self.tick
