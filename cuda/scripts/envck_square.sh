#!/bin/bash
#SBATCH --job-name=envck_square
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=03:00:00
# Item 5 (second model) + item 7 probe: square J1-J2 (J2=0.5) YC8 16x8, full ramp from random.
JOBTAG=square; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1"
R="sched=128:2,256:2,512:3,1024:4"
for k in 0 12 4 8 24; do run sq8_k$k "$SQ $OURS $R envck=$k"; done
run sq8_std_k0  "$SQ $STD $R envck=0"
run sq8_std_k12 "$SQ $STD $R envck=12"
run sq8_d2048_k12 "$SQ $OURS sched=128:2,256:2,512:2,1024:2,2048:3 envck=12"
run sq8_d2560_k12 "$SQ $OURS sched=128:2,256:2,512:2,1024:2,2048:1,2560:2 envck=12"
echo "end=$(date)"
