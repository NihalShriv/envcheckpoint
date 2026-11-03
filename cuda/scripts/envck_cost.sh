#!/bin/bash
#SBATCH --job-name=envck_cost
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:15:00
# Item 3 (cost-model data): overhead and peak memory vs D, N and W, k = all-stored vs 12 vs best-sqrt.
JOBTAG=cost; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
for D in 512 768 1280 1536; do for k in 0 12; do run tri_d${D}_k$k "$TRI $OURS init_from=$L1 sched=$D:2 envck=$k"; done; done
for k in 0 12; do run tri_d1024_std_k$k "$TRI $STD init_from=$L1 sched=1024:2 envck=$k"; done
# different N and W: square YC6 24x6 (N=144) and YC10 8x10 (N=80), D=1024
for k in 0 12; do run sq6_k$k "model=cyl Lx=24 Ly=6 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 $OURS sched=256:2,1024:3 envck=$k"; done
for k in 0 9; do run sq10_k$k "model=cyl Lx=8 Ly=10 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 $OURS sched=256:2,1024:3 envck=$k"; done
echo "end=$(date)"
