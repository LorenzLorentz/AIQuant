"""MultiAssetLOBDataset — yields aligned (cond_orders, x_0, cond_lob) triples
for N assets simultaneously, stacked along a leading asset axis.

The on-disk format is the same per-asset ``.npy`` produced by
``LOBSTERDataBuilder``: each row is ``[order(6) | lob(40)]``. Each path
supplied to this dataset corresponds to one asset.

Alignment policy for P0 (deliberately simple, documented here so it does not
silently bite us in P1+):
  * All asset files must have been produced by the same LOBSTERDataBuilder
    run so their rows already correspond to the same wall-clock bucket.
  * If lengths differ we truncate to ``min(len_i)``; we do NOT attempt to
    re-index by inter-arrival times. A proper timestamp-based join is
    deferred until we have actual ETF + constituent files to test on.
"""

import numpy as np
import torch
from torch.utils import data

import constants as cst


class MultiAssetLOBDataset(data.Dataset):
    """Multi-asset variant of :class:`LOBDataset`.

    Each ``__getitem__`` returns a triple of tensors with a leading asset
    axis of size ``N``:

    * ``cond_orders``  shape ``(N, cond_seq_size, LEN_ORDER)``
    * ``x_0``          shape ``(N, gen_seq_size, LEN_ORDER)``
    * ``cond_lob``     shape ``(N, cond_seq_size + 1, N_LOB_LEVELS * LEN_LEVEL)``
    """

    def __init__(
        self,
        paths,
        seq_size,
        gen_seq_size,
        is_val=False,
        batch_size=None,
        limit_val_batches=None,
    ):
        if len(paths) < 2:
            raise ValueError(
                "MultiAssetLOBDataset expects >=2 asset paths, got "
                f"{len(paths)}. Use LOBDataset for single-asset runs."
            )
        self.paths = list(paths)
        self.num_assets = len(self.paths)
        self.seq_size = seq_size
        self.gen_seq_size = gen_seq_size
        self.cond_seq_size = self.seq_size - self.gen_seq_size
        self.is_val = is_val
        self.batch_size = batch_size
        self.limit_val_batches = limit_val_batches

        self._load_aligned()

    # ---- public API ------------------------------------------------------

    def __len__(self):
        return self.length - self.seq_size + 1

    def __getitem__(self, index):
        index_cond = self.cond_seq_size + index
        index_x = self.cond_seq_size + index + self.gen_seq_size

        # Per-asset slicing then stack along a new leading axis.
        cond = torch.stack(
            [self.orders[i][index:index_cond] for i in range(self.num_assets)],
            dim=0,
        )
        x_0 = torch.stack(
            [self.orders[i][index_cond:index_x] for i in range(self.num_assets)],
            dim=0,
        )
        lob = torch.stack(
            [self.lob[i][index:index_cond + 1] for i in range(self.num_assets)],
            dim=0,
        )
        return cond, x_0, lob

    # ---- internals -------------------------------------------------------

    def _load_aligned(self):
        """Load all asset files, truncate to common length, stash per-asset
        ``orders`` and ``lob`` tensors. Also keeps a ``data`` attribute
        compatible with ``DataModule`` (which probes ``train_set.data``).
        """
        raw = []
        for path in self.paths:
            arr = np.load(path)
            raw.append(arr)

        min_len = min(arr.shape[0] for arr in raw)
        # Validate column layout against the single-asset format.
        for path, arr in zip(self.paths, raw):
            expected = cst.LEN_ORDER + cst.N_LOB_LEVELS * cst.LEN_LEVEL
            if arr.shape[1] != expected:
                raise ValueError(
                    f"{path}: expected {expected} columns "
                    f"(LEN_ORDER + N_LOB_LEVELS*LEN_LEVEL), got {arr.shape[1]}"
                )

        if self.is_val:
            # Mirror LOBDataset's val cap so wall-clock val batches match.
            cap = self.batch_size * self.limit_val_batches
            min_len = min(min_len, cap)

        self.length = min_len

        self.orders = []
        self.lob = []
        for arr in raw:
            arr = arr[:min_len]
            orders = torch.from_numpy(arr[:, :cst.LEN_ORDER]).float().contiguous()
            lob = arr[:, cst.LEN_ORDER:]
            # Same shift as LOBDataset: align lob[t] with the *pre-event* book.
            lob = np.roll(lob, 1, axis=0)
            lob[0, :] = 0
            lob = torch.from_numpy(lob).float().contiguous()
            self.orders.append(orders)
            self.lob.append(lob)

        # Lightning's DataModule.pin_memory check reads `.data.device.type`.
        # Stack one tensor so the attribute exists; not used in __getitem__.
        self.data = torch.stack(self.orders, dim=0)
