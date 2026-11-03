#!/bin/bash
#SBATCH --job-name=envck_main
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=03:00:00
# Checkpointed environments, separate tree ~/DMRG_envck (binary v2 = streaming final energy).
# A) memory/time vs checkpoint spacing k at D=1024 and D=2048; B) longer D=2048 runs, ours vs standard;
# C) D=2560 on one GPU. All XC6-24 triangular J2=0.12, low state.
set -uo pipefail
cd ~/DMRG_envck; O=own_experimental/results/main; mkdir -p $O
B=own_experimental/bin/dmrg_envck_v2; T=~/DMRG_v6/own_experimental/results/tri2
nvidia-smi --query-gpu=name --format=csv,noheader; echo "start=$(date)"
C="model=tric Lx=24 Ly=6 J2=0.12 seed=1 thresh=0 heff=cutensor compress=1 production=1"
STD="svd=gram dav_maxmv=4 dav_fast=1"; OURS="svd=warm warm_iters=1 dav_maxmv=2 dav_fast=1"
run() { rm -f $O/$1.*; echo "--- $1: $2 $(date +%T)"; $B $C $2 ckpt=$O/$1.ckpt out=$O/$1.csv > $O/$1.out 2>&1 || echo "RUN FAILED $1: $(grep -h 'what()\|ERROR\|out of memory' $O/$1.out | tail -1)"
  cut -d, -f8,9,11,15,22 $O/$1.csv | tail -n +2 | sed 's/^/  sw,D,E,t,mem: /'; grep -h "E_final\|final\|<psi|H|psi>" $O/$1.out | tail -2; rm -f $O/$1.ckpt; }
# A) k scan (3 sweeps each, from converged low states)
for k in 0 4 8 12 24; do run k1024_$k "$OURS init_from=$T/v3_stab_std_from_low.ckpt sched=1024:3 envck=$k"; done
for k in 0 4 8 24; do run k2048_$k "$OURS init_from=$T/d1280_ours_x.ckpt sched=2048:3 envck=$k"; done
# B) longer D=2048 runs (about equal wall time)
run d2048_ours_long "$OURS init_from=$T/d1280_ours_x.ckpt sched=2048:12 envck=12"
run d2048_std_long  "$STD  init_from=$T/d1280_ours_x.ckpt sched=2048:8 envck=12"
# C) D=2560 on one GPU
run d2560_ours "$OURS init_from=$T/d1280_ours_x.ckpt sched=2560:3 envck=12"
echo "end=$(date)"
