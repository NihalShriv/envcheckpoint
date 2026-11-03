#!/bin/bash
#SBATCH --job-name=envck_xc8
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
# Widest MPO case: triangular XC8-24 (N=192, W=50). All-stored limit vs checkpointed reach on one H200 (v5).
JOBTAG=xc8; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
B=own_experimental/bin/dmrg_envck_v5
X8="model=tric Lx=24 Ly=8 J2=0.12 seed=1 thresh=0 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
S=~/DMRG_v6/own_experimental/results/tri2/x8_ours_fromJ0.ckpt
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
runp x8_d1024_k0   "$X8 $OURS init_from=$S sched=1024:2 envck=0"
runp x8_d1024_k14  "$X8 $OURS init_from=$S sched=1024:2 envck=14"
runp x8_d1536_k0   "$X8 $OURS init_from=$S sched=1536:2 envck=0"
runp x8_d2048_k14  "$X8 $OURS init_from=$S sched=2048:2 envck=14"
runp x8_d2560_2lvl "$X8 $OURS init_from=$S sched=2560:2 envck=48 envck2=7"
echo "end=$(date)"
