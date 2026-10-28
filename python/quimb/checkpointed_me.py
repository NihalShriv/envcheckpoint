"""Exact environment checkpointing for quimb's DMRG (envck tree).

quimb's ``MovingEnvironment`` builds every environment of a sweep up front (``init_segment``) and keeps all of them,
including the ones the sweep has already passed. ``CheckpointedMovingEnvironment`` keeps only every k-th one
(k ~ sqrt(N)), rebuilds a missing one from the nearest kept one with exactly the contractions ``init_segment`` uses,
and drops environments behind the current position. Results are unchanged.

Use ``use_checkpointed_me(k)`` before running ``quimb.tensor.DMRG2`` (patches the class the DMRG sweeps build).
"""
import math
import quimb.tensor.tn1d.dmrg as _qd

_Base = _qd.MovingEnvironment


class CheckpointedMovingEnvironment(_Base):
    ckpt_k = None
    n_rebuilt = 0

    def init_segment(self, begin, start, stop):
        if self.cyclic:
            raise NotImplementedError("CheckpointedMovingEnvironment supports open boundary conditions only")
        super().init_segment(begin, start, stop)
        self._begin = begin
        k = self.ckpt_k or max(1, round(math.sqrt(self.L)))
        ref = start if begin == "left" else stop - 1
        for i in list(self.envs):
            if i != self.pos and (i - ref) % k and i not in (start, stop - 1):
                del self.envs[i]

    def _rebuild(self, i):
        """Rebuild env i from the nearest kept env further along the build direction (same steps as init_segment)."""
        if self._begin == "left":
            c = i + 1
            while c not in self.envs:
                c += 1
                if c >= self.segment.stop:
                    raise RuntimeError(f"no stored environment to rebuild env {i} from")
            for j in reversed(range(i, c)):
                self.envs[j] = self.envs[j + 1].copy(virtual=True)
                self.envs[j] |= self.tnc.select(j)
                self.envs[j] ^= ("_RIGHT", self.site_tag(j + self.bsz))
                CheckpointedMovingEnvironment.n_rebuilt += 1
        else:
            c = i - 1
            while c not in self.envs:
                c -= 1
                if c < self.segment.start:
                    raise RuntimeError(f"no stored environment to rebuild env {i} from")
            for j in range(c + 1, i + 1):
                self.envs[j] = self.envs[j - 1].copy(virtual=True)
                self.envs[j] |= self.tnc.select(j + self.bsz - 1)
                self.envs[j] ^= ("_LEFT", self.site_tag(j - 1))
                CheckpointedMovingEnvironment.n_rebuilt += 1

    def move_right(self):
        i = (self.pos + 1) % self.L
        if i in self.segment and i not in self.envs and self._begin == "left":
            self._rebuild(i)
        super().move_right()
        for j in [j for j in self.envs if j < self.pos - 1]:   # passed: never used again this sweep
            del self.envs[j]

    def move_left(self):
        i = (self.pos - 1) % self.L
        if i in self.segment and i not in self.envs and self._begin == "right":
            self._rebuild(i)
        super().move_left()
        for j in [j for j in self.envs if j > self.pos + 1]:
            del self.envs[j]


def use_checkpointed_me(k=None):
    CheckpointedMovingEnvironment.ckpt_k = k
    _qd.MovingEnvironment = CheckpointedMovingEnvironment


def use_standard_me():
    _qd.MovingEnvironment = _Base
