#!/bin/bash
#SBATCH --job-name=envck_feas
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:30:00
# Checkpointed environments (envck=k), separate tree ~/DMRG_envck. 1) exactness vs stored envs (D=512);
# 2) XC6-24 at D=2048 on one GPU (OOM without envck), ours vs standard, from the D=1280 low state.
set -uo pipefail
cd ~/DMRG_envck; O=own_experimental/results/feas; mkdir -p $O
B=own_experimental/bin/dmrg_envck_v1; T=~/DMRG_v6/own_experimental/results/tri2
nvidia-smi --query-gpu=name --format=csv,noheader
C="model=tric Lx=24 Ly=6 J2=0.12 seed=1 thresh=0 heff=cutensor compress=1 production=1"
STD="svd=gram dav_maxmv=4 dav_fast=1"; OURS="svd=warm warm_iters=1 dav_maxmv=2 dav_fast=1"
run() { rm -f $O/$1.*; echo "--- $1: $2 $(date)"; $B $C $2 ckpt=$O/$1.ckpt out=$O/$1.csv > $O/$1.out 2>&1 || echo "RUN FAILED $1: $(grep -h 'what()\|ERROR\|out of memory' $O/$1.out | tail -1)"
  cut -d, -f8,10,11,15,22 $O/$1.csv | sed 's/^/  /'; rm -f $O/$1.ckpt; }
run exact_k0  "$OURS init=stripe_x sched=128:2,512:3 envck=0"
run exact_k12 "$OURS init=stripe_x sched=128:2,512:3 envck=12"
run d2048_ours "$OURS init_from=$T/d1280_ours_x.ckpt sched=2048:3 envck=12"
run d2048_std  "$STD  init_from=$T/d1280_ours_x.ckpt sched=2048:2 envck=12"
echo "end=$(date)"
