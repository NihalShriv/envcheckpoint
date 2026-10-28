# envcheckpoint: exact low-memory environments for DMRG / TDVP
n[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23725023.svg)](https://doi.org/10.5281/zenodo.23725023)

Finite-MPS sweep algorithms (DMRG, TDVP) keep one left and one right environment tensor per site. These dominate
memory at large bond dimension. **envcheckpoint** keeps only every *k*-th environment (k ≈ √N) plus the segment in
use, and recomputes the others from the nearest stored one when the sweep reaches them — the same contractions, so
**results are unchanged** — for about one extra environment update per site and sweep.

Drop-in implementations (ITensorMPS.jl and TeNPy benchmarked; MPSKit.jl and quimb exactness-tested, experimental):

| library | file | usage |
|---|---|---|
| ITensorMPS.jl | `julia/itensor/CheckpointedProjMPOs.jl` | `dmrg(CheckpointedProjMPO(H), psi0, sweeps)`, `tdvp(CheckpointedProjMPO(H), t, psi0; ...)` |
| MPSKit.jl | `julia/mpskit/CheckpointedEnvs.jl` | `find_groundstate(ψ, H, DMRG2(; trunc), checkpointed(environments(ψ, H, ψ)))` |
| TeNPy | `python/tenpy/checkpointed_env.py` (or `patches/tenpy_env_checkpoint.patch`: option `env_checkpoint_every`) | `use_checkpointed_env(k)` then any TeNPy DMRG/TDVP engine |
| quimb | `python/quimb/checkpointed_me.py` | `use_checkpointed_me(k)` then `quimb.tensor.DMRG2(...)` |

## Correctness
Energies identical to the standard environments (difference 0.0) in every test: ITensor (dense, U(1)), MPSKit
(dense, U(1)), TeNPy (dense, U(1), single-site + subspace expansion, density-matrix mixer, with disk cache), quimb
(chains, J1-J2 cylinders); TDVP states agree to ~1e-15. Tests in `tests/`.

## Measured benefit (see the accompanying paper)
- Environment memory reduced up to 10x (long cylinders); peak memory 1.5-3.8x lower depending on length and MPO width.
- quimb: the saving also includes dropping environments the sweep has already passed.
- Optimal k = √N (verified for N = 96, 192, 384).
- Overhead ~6 % with standard eigensolvers (median of 10 repeats; ~15 % with a 2-iteration Davidson); faster than file-based offload (RAM disk, local disk, parallel filesystem).
- GPU, own CUDA code: bond dimension on one H200 1664 → 3712 (triangular J1-J2 XC6), 2176 → ≥4096 (square YC8); on an 80 GB H100 1280 → 2816 and 1664 → ≥3328.
- ITensorMPS.jl, U(1), H100 80 GB: maxdim 3584 → 4608 at equal sweep time.
- TeNPy (CPU): peak RAM 2.9-4.3x lower with no slowdown; with TeNPy's disk cache, 3.7x less disk.

## Options
- `k`: checkpoint spacing (default round(√N)).
- ITensor: `k2` (two-level schedule), `offload` (keep checkpoints in host RAM/disk), `after_drop` (e.g. `CUDA.reclaim`),
  `chunks` (chunked H_eff application for dense tensors).

## Citation
If you use this, please cite the paper (arXiv link to be added) and this repository: [doi:10.5281/zenodo.23725023](https://doi.org/10.5281/zenodo.23725023).

## License
MIT
