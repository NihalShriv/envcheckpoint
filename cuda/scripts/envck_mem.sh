#!/bin/bash
#SBATCH --job-name=envck_mem
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=03:00:00
# envck v3 (separate tree ~/DMRG_envck): true per-bond peak memory (memtrack=1), shared H_eff buffers,
# disk-offload baseline (envdisk), and D=2560/3072 on one GPU. XC6-24 triangular J2=0.12, low state, ours.
set -uo pipefail
cd ~/DMRG_envck; O=own_experimental/results/mem; mkdir -p $O; DK=$SCRATCH/dmrg_envck/diskenv
B=own_experimental/bin/dmrg_envck_v3; T=~/DMRG_v6/own_experimental/results/tri2
nvidia-smi --query-gpu=name --format=csv,noheader; echo "start=$(date)"
C="model=tric Lx=24 Ly=6 J2=0.12 seed=1 thresh=0 heff=cutensor compress=1 production=1 memtrack=1"
OURS="svd=warm warm_iters=1 dav_maxmv=2 dav_fast=1"
L1=$T/v3_stab_std_from_low.ckpt; L2=$T/d1280_ours_x.ckpt
run() { rm -f $O/$1.*; echo "--- $1: $2 $(date +%T)"; $B $C $2 ckpt=$O/$1.ckpt out=$O/$1.csv > $O/$1.out 2>&1 || echo "RUN FAILED $1: $(grep -h 'what()\|ERROR\|out of memory' $O/$1.out | tail -1)"
  cut -d, -f8,9,11,15,22 $O/$1.csv | tail -n +2 | sed 's/^/  sw,D,E,t,peakGB: /'; grep -h "^final" $O/$1.out; rm -f $O/$1.ckpt; }
disk() { rm -rf $DK; mkdir -p $DK; run $1 "$2 envck=1000 envdisk=$DK"; du -sh $DK; rm -rf $DK; }
# A) D=1024: peak memory vs k, plus disk offload of all environments
for k in 0 4 8 12 24 36 72; do run m1024_k$k "$OURS init_from=$L1 sched=1024:3 envck=$k"; done
disk m1024_disk "$OURS init_from=$L1 sched=1024:2"
# B) D=2048: k scan (all-stored k=0 = control), disk offload (1 sweep)
for k in 0 4 8 12 24; do run m2048_k$k "$OURS init_from=$L2 sched=2048:2 envck=$k"; done
disk m2048_disk "$OURS init_from=$L2 sched=2048:1"
# C) beyond: D=2560, D=3072 on one GPU
run m2560_k12 "$OURS init_from=$L2 sched=2560:3 envck=12"
run m3072_k12 "$OURS init_from=$L2 sched=3072:2 envck=12"
echo "end=$(date)"
