"""Exact environment checkpointing for TeNPy (envck tree).

``CheckpointedMPOEnvironment(..., ckpt_k=k)`` is an ``MPOEnvironment`` that keeps only every k-th left/right
environment (plus the segment next to the most recently stored one) and drops the rest. TeNPy's ``get_LP`` /
``get_RP`` already recompute a missing environment from the nearest stored one, so the dropped blocks are rebuilt
exactly (same contractions) when a sweep needs them. Memory ~ (L/k + 2k) environments instead of ~L, for ~1 extra
environment contraction per site and sweep.

Use with the DMRG engines via ``use_checkpointed_env(k)`` (patches the class the engines instantiate), or build it
directly like an ``MPOEnvironment``.
"""
import math
from tenpy.networks.mpo import MPOEnvironment
import tenpy.algorithms.mps_common as _mps_common


class CheckpointedMPOEnvironment(MPOEnvironment):
    ckpt_k_default = None      # set by use_checkpointed_env; None -> round(sqrt(L))

    ckpt_k = None              # instance value set after base init (base init already stores LP_0 / RP_{L-1})

    def __init__(self, *args, ckpt_k=None, **kwargs):
        super().__init__(*args, **kwargs)
        k = ckpt_k or self.ckpt_k_default or max(1, round(math.sqrt(self.L)))
        self.ckpt_k = int(k)
        self.n_dropped = 0

    def _stored(self, keys):
        return [j for j, key in enumerate(keys) if key in self.cache]

    def set_LP(self, i, LP, age):
        super().set_LP(i, LP, age)
        if self.finite and self.ckpt_k:
            i = self._to_valid_site_index(i)
            k = self.ckpt_k
            lo = i - i % k                  # keep LP_j for j in (lo, i] and every multiple of k (LP_0 always)
            for j in self._stored(self._LP_keys):
                if j % k and j <= lo:
                    self.del_LP(j)
                    self.n_dropped += 1

    def set_RP(self, i, RP, age):
        super().set_RP(i, RP, age)
        if self.finite and self.ckpt_k:
            i = self._to_valid_site_index(i)
            k, last = self.ckpt_k, self.L - 1
            hi = i + (-i) % k               # keep RP_j for j in [i, hi), multiples of k, and RP_{L-1}
            for j in self._stored(self._RP_keys):
                if j % k and j != last and j >= hi:
                    self.del_RP(j)
                    self.n_dropped += 1


def use_checkpointed_env(k=None):
    """Make TeNPy's MPS algorithms (DMRG, TDVP, ...) build CheckpointedMPOEnvironment instead of MPOEnvironment."""
    CheckpointedMPOEnvironment.ckpt_k_default = k
    _mps_common.MPOEnvironment = CheckpointedMPOEnvironment


def use_standard_env():
    _mps_common.MPOEnvironment = MPOEnvironment
