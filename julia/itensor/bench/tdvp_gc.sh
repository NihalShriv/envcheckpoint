#!/bin/bash
#SBATCH --job-name=tdvp_gc
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --exclusive
#SBATCH --qos=cpulimit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=16:00:00
# Matched garbage-collection control for the ITensorMPS.jl TDVP timing (reviewer point): CheckpointedProjMPO by
# default calls GC.gc(false) after dropping blocks; here it does not (after_drop = () -> nothing), so neither store
# forces collections. Same model and node type as job 422828 (speed_rome); stored and checkpointed alternated.
set -uo pipefail
R=$SCRATCH/dmrg_envck/tdvp_gc; mkdir -p $R; E=~/DMRG_envck/own_experimental
echo "start=$(date) node=$SLURMD_NODENAME"; lscpu | grep "Model name"
cd ~/DMRG_v6; export JULIA_DEPOT_PATH=$HOME/.julia JULIA_NUM_THREADS=1 NBLAS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for rep in 1 2; do for s in proj ckpt_nogc; do
  echo "--- itensor tdvp $s rep=$rep $(date +%T)"
  STORE=$s MODE=tdvp LX=12 MAXDIM=768 NSTEPS=12 DT=0.05 timeout -k 60 3h julia --project=itensor $E/itensor/bench2.jl $R/tdvp_${s}_$rep > $R/tdvp_${s}_$rep.out 2>&1; echo "exit=$? $(date +%T)"
  tail -2 $R/tdvp_${s}_$rep.csv
done; done
echo "end=$(date)"
