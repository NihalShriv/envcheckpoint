#!/bin/bash
#SBATCH --job-name=review_retain
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:15:00
# Checkpoints kept from the previous half-sweep (ck_retain=1, v8) against the per-half-sweep rebuild of v7
# (ck_retain=0) and against stored environments with and without envcache. Triangular XC6 D=1024, the two solvers
# of Table I, same starting state and settings as job stats; the four variants alternate within each repeat.
set -uo pipefail
SCRATCH=${SCRATCH:-/scratch/$USER}
C=$SCRATCH/cuda_v8/cuda; B=$C/bin/dmrg_envck
O=$SCRATCH/dmrg_envck/review_retain; mkdir -p $O
L1=$HOME/DMRG_v6/own_experimental/results/tri2/v3_stab_std_from_low.ckpt
STD="svd=gram dav_maxmv=4 dav_fast=1"; OURS="svd=warm warm_iters=1 dav_maxmv=2 dav_fast=1"
TRI="model=tric Lx=24 Ly=6 J2=0.12 seed=1 thresh=0 heff=cutensor compress=1 production=1 memtrack=1"
nvidia-smi --query-gpu=name --format=csv,noheader; echo "start=$(date)"
export CCBIN=/apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-11.2.0-ov3qrz6n2r7ysjhyvps5rlyu2qskp5b2/bin/g++
export CUDA_HOME=/apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/cuda-12.1.0-s5o57xpc7sqqhfd6sxfealzbqxtmvh7p
export CUTENSOR_ROOT=$HOME/opt/cutensor_whl/cutensor
bash $C/build.sh dmrg_envck > $O/build.log 2>&1 || { echo "build failed"; tail $O/build.log; exit 1; }
run() { echo "--- $1: $2 $(date +%T)"; $B $2 ckpt=$O/$1.ckpt out=$O/$1.csv > $O/$1.out 2>&1; echo "exit=$?"
  rm -f $O/$1.ckpt; tail -n 1 $O/$1.csv | cut -d, -f8,11,15,22; grep -h "final:" $O/$1.out; }
for r in 1 2 3; do for s in std ours; do
  [ $s = std ] && S=$STD || S=$OURS
  run r${r}_${s}_stored   "$TRI $S init_from=$L1 sched=1024:3 envck=0"
  run r${r}_${s}_cache    "$TRI $S init_from=$L1 sched=1024:3 envck=0 envcache=1"
  run r${r}_${s}_rebuild  "$TRI $S init_from=$L1 sched=1024:3 envck=12 ck_retain=0"
  run r${r}_${s}_retain   "$TRI $S init_from=$L1 sched=1024:3 envck=12 ck_retain=1"
done; done
echo "end=$(date)"
