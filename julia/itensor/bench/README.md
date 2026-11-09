# ITensorMPS.jl CPU benchmarks (TDVP and H16 rows of the paper)

`bench2.jl` runs one store per process (`ProjMPO` or `CheckpointedProjMPO`), so `Sys.maxrss` is that store's peak.

## Setup

Julia 1.10 (the paper's runs used 1.10.11). `Project.toml` pins ITensors 0.9.32, ITensorMPS 0.4.2 and NDTensors
0.4.32, the versions used for the paper.

```bash
cd julia/itensor/bench
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Small check (a few minutes; the first step includes compilation)

```bash
for s in proj ckpt ckpt_nogc; do
  STORE=$s MODE=tdvp LX=4 MAXDIM=32 NSTEPS=3 NBLAS=2 julia --project=. bench2.jl out_$s
done
cat out_*.csv     # energies agree between the three stores at every step
```

## Paper runs

Starting states are generated inside `bench2.jl`, so no state files are needed:

- TDVP (`MODE=tdvp`): Néel product state on the Heisenberg YC6 cylinder, `LX=12 MAXDIM=768 NSTEPS=12 DT=0.05`
  (`tdvp_gc.sh`: stored vs checkpointed without forced collection; the earlier `speed_rome` pairs used `STORE=ckpt`).
- H16 (`MODE=chem`): random MPS (`Random.seed!(1)`, link dimension 16, half filling) for the integrals of a linear
  H16 chain (R = 1.8 bohr, STO-6G), written by `chem_ints.py` (Python 3.11, PySCF 2.14.0, NumPy 2.4.6):

```bash
python chem_ints.py 16 h16.txt
STORE=proj MODE=chem DS=128,256,512 julia --project=. bench2.jl chem_proj h16.txt
STORE=ckpt MODE=chem DS=128,256,512 julia --project=. bench2.jl chem_ckpt h16.txt
```

Environment variables: `NBLAS` (BLAS threads, default 4); for the TDVP job also `OUT` (results directory) and
`BENCH_DIR` (this directory, needed under `sbatch`, which runs a copy of the script).
