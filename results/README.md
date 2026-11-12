# Raw results

Raw outputs of every benchmark in the paper, in `raw/` (directories below are relative to it). Each run has a `.out` (log with per-sweep energy, time and memory),
usually a `.csv` (one row per sweep), and for the GPU code a `.timing` file. File names encode the setting:
`k0` = all environments stored, `kN` = checkpointed with spacing N, `2lvl` = two-level schedule, `cN` = N chunks,
`dN` = bond dimension, `rN` = repeat N. Account paths are replaced by `$HOME` / `$SCRATCH`.

`figures/make_figs.py` regenerates the paper figures from `figures/data/` (`cd figures && python make_figs.py`).

## GPU code (H100 80 GB / H200 141 GB)

| directory | paper | content |
|---|---|---|
| `reps`, `stats` | Table I (overhead) | repeated stored/checkpointed pairs, square YC8 D=2048 and triangular XC6 D=1024, standard and two-vector Davidson |
| `cost`, `kbig` | cost model, Eq. (3) | time per sweep against k for several models and D |
| `nscan` | Table II, Fig. 2 | square YC6, D=1024, L = 16, 32, 64, scan over k |
| `square`, `xc8`, `cost` | memory against MPO width | square YC8, triangular XC8 (and square YC10 in `cost`) |
| `pinned` | Fig. 3, Table III | memory breakdown; pinned asynchronous offload, three repeats |
| `mem`, `fastdisk` | Table III | file-based offload (RAM disk, local disk, parallel file system), synchronous and asynchronous |
| `chunk` | working memory | chunked effective Hamiltonian |
| `bisect` (H100), `bisect200` (H200), `h100reach`, `h100edge*`, `xc8big`, `main`, `feas` | Table IV | reach searches: largest D completing two sweeps |
| `robust` | Table IV (capped rows), Table V | H200 capped to 24/40/48 GB (checkpointed: `robust.sh`; stored: `robust_st.sh`, `r48s.sh` random start, the `c*s` stored runs of `robust.sh` failed at their first D with a cuTENSOR error and were rerun); ITensorMPS.jl TDVP, H16 and MPSKit.jl runs of job 420808 |
| `bisect200_like` | Table IV check | H200, `bis_h200_like.sh`: stored with `heff_chunks=8` (t0c: 1792 pass, 1920 fail) and one-level k=12 without chunking (t1: 2944 pass, 3072 fail) |
| `stab` | correctness | 20-sweep stored vs checkpointed comparison |
| `conv`, `flagship`, `trilad` | Fig. 5 | converged ladders, square YC8 and triangular XC6 up to D=3072 |

## Libraries (CPU unless stated)

| directory | paper | content |
|---|---|---|
| `matched`, `matched2` | ITensorMPS.jl, H100 | dense GPU runs, YC6 and YC10: `ProjMPO` vs `CheckpointedProjMPO` |
| `itensor` | correctness | ITensorMPS.jl exactness checks |
| `results/tenpy` | TeNPy memory | peak RAM, YC6 cylinders, chi = 512 to 2048 |
| `results/cpu_checks`, `results/quimb` | correctness, quimb | test runs and quimb benchmarks |
| `block2`, `results/block2` | out-of-core codes | block2 disk footprint and I/O share |
| `speed_rome` | Fig. 4, Tables V and VI | dedicated AMD EPYC 7542 node: TDVP, H16, MPSKit.jl (`*.csv`), TeNPy (`tenpy.jsonl`, 3 pairs) |
| `speed` | Table VI (Genoa row) | AMD EPYC 9654, one TeNPy pair (`tenpy_run1_partial.jsonl`) |

Software: TeNPy 1.1.1 with numpy 2.4.6 (OpenBLAS 0.3.31) for `speed_rome`; ITensorMPS.jl 0.4.2; Julia 1.10.

## Checks added after review (released v7 build, `cuda/scripts/review_*.sh`, `julia/itensor/bench/`)

| directory | content |
|---|---|
| `review` (`A_*`) | rerun of a `stats` pair with the released build: energies, discarded weights and residuals identical to `stats/r1_std_k*` |
| `review` (`C_*`) | ablation, triangular XC6 D=1024: stored/checkpointed x Davidson basis (8/160) x chunks (1/4); checkpointing changes no digit, chunking changes the last digits |
| `review` (`D_*`) | converged local solves (no matvec cap), stored vs checkpointed |
| `review` (`B_ladder`) | square YC8 ladder to D=3072 with the settings of `conv`; same final energy to all printed digits |
| `review` (`B_strict24`) | 2 sweeps at D=3072 from `B_ladder`, up to 24 matvecs per solve, tolerance 1e-8 |
| `tdvp_gc` | ITensorMPS.jl TDVP as in `speed_rome`, with `CheckpointedProjMPO(H; after_drop = () -> nothing)` (no forced collection), two pairs |
| `review_retain` | v8 build (`ck_retain`), triangular XC6 D=1024, 3 sweeps x 3 runs, both solvers: stored (`envck=0`), stored with caching (`envcache=1`), checkpointed with rebuild (`ck_retain=0`) and with retained checkpoints (`ck_retain=1`); identical energies in all runs |
