#!/bin/bash
#SBATCH --job-name=review_strict
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:15:00
# Square YC8 at D=3072, from the ladder state of review_gpu.sh: 2 sweeps with tighter local solves
# (up to 24 matrix-vector products, tolerance 1e-8) instead of the 4-product cap. max_hours stops before a sweep
# that would not fit.
set -uo pipefail
SCRATCH=${SCRATCH:-/scratch/$USER}
B=$SCRATCH/cuda_reltest/cuda/bin/dmrg_envck
O=$SCRATCH/dmrg_envck/review_gpu
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
nvidia-smi --query-gpu=name --format=csv,noheader; echo "start=$(date)"
$B $SQ svd=gram dav_maxmv=24 dav_tol=1e-8 dav_cols=32 init_from=$O/B_ladder.ckpt sched=3072:2 envck=12 heff_chunks=4 \
  max_hours=1.95 ckpt=$O/B_strict24.ckpt out=$O/B_strict24.csv > $O/B_strict24.out 2>&1
echo "exit=$? end=$(date)"; cut -d, -f8,11,15,22,24,25 $O/B_strict24.csv
