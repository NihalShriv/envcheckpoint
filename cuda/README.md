# GPU DMRG code used for the paper's GPU results

Standalone two-site DMRG in CUDA (double precision, cuBLAS/cuSOLVER/cuTENSOR, no quantum-number symmetries) with
exact environment checkpointing. All GPU tables and figures of the paper were produced with this code.

- `apps/dmrg_envck.cu`: the program (main loop, options, checkpointing in `ck_build_*` / `ck_need_*`).
- `apps/dmrg_cap.cu`: the same program plus `gpu_cap_gb=`, which reserves a fixed ballast so only that many GB stay
  usable (the capped-GPU rows of Table IV).
- `src/`: tensors, MPO/MPS, environments, effective Hamiltonian, Davidson and SVD.
- `scripts/`: the exact Slurm scripts that produced `../results/raw/` (see below).

## Build

```bash
export CUTENSOR_ROOT=/path/to/cutensor     # contains include/ and lib/
bash build.sh dmrg_envck                   # or dmrg_cap; ARCH=sm_80 for A100
```

Tested with CUDA 12.1, gcc 11.2 and cuTENSOR 2.x on H100 and H200. If nvcc picks an unusable host compiler, set
`CCBIN=/path/to/g++`.

## Main options

Options are `key=value`; unknown keys are rejected.

| option | meaning |
|---|---|
| `model=cyl Lx= Ly= J2=` | square J1-J2 cylinder (YC); `model=tric` triangular J1-J2 XC cylinder |
| `sched=D:n,D:n,...` | bond dimension schedule (n sweeps at each D) |
| `envck=k` | checkpointing: keep every k-th environment (0 = store all) |
| `ck_retain=0\|1` | 1 (default): checkpoints are kept from the previous half-sweep; 0: rebuilt at the start of every half-sweep, as in builds up to v7 |
| `envcache=1` | stored runs: keep environments across half-sweeps instead of rebuilding them |
| `envck2=k2` | two-level schedule inside a rebuilt segment |
| `heff_chunks=c` | split the effective-Hamiltonian intermediate into c chunks (changes summation order) |
| `dav_maxmv=m` | at most m matrix-vector products per local Davidson solve (0 = unlimited, converge to `dav_tol`) |
| `dav_cols=n` | Davidson basis size (default: `dav_maxmv`+4, or the maximum if unlimited) |
| `svd=gram\|jacobi\|warm` | truncation method |
| `envdisk=<dir>`, `envdisk_async=1`, `envpin=1` | offload baselines: dropped blocks go to disk or pinned host memory instead |
| `gpu_cap_gb=x` | (`dmrg_cap` only) emulate a card with x GB |
| `memprof=1`, `memtrack=1` | log the per-sweep peak split into MPS / environments / other |
| `init=random\|stripe_x`, `init_from=<file>` | initial state |

Stored and checkpointed runs differ only in `envck` (and `envck2`); every other option is identical within a
comparison.

## Solver settings used in the paper

The scripts use two settings, defined in `scripts/envck_hdr.sh`:

- `STD = svd=gram dav_maxmv=4 dav_fast=1`: Davidson limited to 4 matrix-vector products per local solve.
- `OURS = svd=warm warm_iters=1 dav_maxmv=2 dav_fast=1`: 2 matrix-vector products and a warm-started truncation.

Local solves are therefore not converged to `dav_tol` within a step; this is common in production sweeps, where
the sweep itself converges the state. Stored and checkpointed runs always use the same setting.

## Scripts, builds and results

Each script sources `scripts/envck_hdr.sh` and writes to `own_experimental/results/<JOBTAG>/`; the copies in
`../results/raw/` are those outputs. Paths refer to the original cluster layout (`~/DMRG_envck`), and
`$SCRATCH`/`$HOME` replace account-specific paths.

The runs were made over several days with successive builds of this source tree (`dmrg_envck_v1` to `_v7`; each
script names the build it used). Later builds add options but leave the default behaviour unchanged. The source
here is the final build (v7) and `dmrg_cap` (v7 plus `gpu_cap_gb`).

## Starting states

Some runs start from saved states (`init_from=...`) to skip the low-D part of the ladder. They were made with
earlier builds of this program (before checkpointing was added); every option below exists in the current build,
which therefore gives equivalent states, though not bitwise the same ones. With

```bash
P="model=tric Lx=24 seed=1 thresh=0 heff=cutensor compress=1 production=1 envcache=1"
STD="svd=gram dav_maxmv=4 dav_fast=1"; OURS="svd=warm warm_iters=1 dav_maxmv=2 dav_fast=1"
B=bin/dmrg_envck
```

the commands are, in order:

```bash
# ours_stripex.ckpt (intermediate)
$B $P Ly=6 J2=0.12 $OURS init=stripe_x sched=64:4,128:4,256:4,512:4,768:4,1024:12 ckpt=ours_stripex.ckpt out=a.csv
# v3_stab_std_from_low.ckpt: triangular XC6 at D=1024 (overhead, ablation and offload runs)
$B $P Ly=6 J2=0.12 symdiag=1 $STD init_from=ours_stripex.ckpt sched=1024:20 ckpt=v3_stab_std_from_low.ckpt out=b.csv
# d1280_ours_x.ckpt: triangular XC6 at D=1280 (D=2048 and 2560 runs)
$B $P Ly=6 J2=0.12 symdiag=1 $OURS init=stripe_x sched=64:4,128:4,256:4,512:4,768:4,1024:6,1280:12 ckpt=d1280_ours_x.ckpt out=c.csv
# x8_J0.ckpt (intermediate), then x8_ours_fromJ0.ckpt: triangular XC8 at D=1024
$B $P Ly=8 J2=0.0 symdiag=1 $OURS init=random sched=64:4,128:4,256:4,512:4,768:4,1024:12 ckpt=x8_J0.ckpt out=d.csv
$B $P Ly=8 J2=0.12 symdiag=1 $OURS init_from=x8_J0.ckpt sched=1024:12 ckpt=x8_ours_fromJ0.ckpt out=e.csv
```

The square-lattice runs start from a random state inside each script. The files themselves (2-3 GB each) are
available on request. Stored and checkpointed runs of a comparison start from the same state, so the comparisons
do not depend on it.
