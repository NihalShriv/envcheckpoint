#!/bin/bash
#SBATCH --job-name=conv_tri
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=10:00:00
# Converged one-GPU D ladder, triangular XC6-24 J2=0.12 low state, standard solver, checkpointed (v6).
# conv=1: stay at each D until |dE|/sweep < stage_tol (max n sweeps), final D until < final_tol.
JOBTAG=conv; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
X="$TRI init_from=$L1"
run tri_conv "$X $STD sched=1024:16,1536:14,2048:12,2560:10,3072:12 conv=1 stage_tol=1e-6 final_tol=1e-7 kick_tol=1e-7 envck=36 envck2=6 heff_chunks=8 memprof=1"
grep -h "^CONV" $O/tri_conv.out | tail -12
echo "end=$(date)"
