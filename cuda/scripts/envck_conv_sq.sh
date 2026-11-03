#!/bin/bash
#SBATCH --job-name=conv_sq
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=10:00:00
# Converged one-GPU D ladder, square J1-J2 YC8 16x8 J2=0.5 (standard benchmark), standard solver, checkpointed (v6).
# conv=1: stay at each D until |dE|/sweep < stage_tol (max n sweeps), final D until < final_tol.
JOBTAG=conv; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
run sq8_conv "$SQ $STD sched=256:6,512:8,1024:16,1536:14,2048:12,2560:10,3072:12 conv=1 stage_tol=1e-6 final_tol=1e-7 kick_tol=1e-7 envck=12 heff_chunks=4"
grep -h "^CONV" $O/sq8_conv.out | tail -12
echo "end=$(date)"
