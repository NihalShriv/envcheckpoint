#!/bin/bash
#SBATCH --job-name=envck_h100
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:45:00
# Capability on a common 80 GB card (H100): largest D with all environments stored vs checkpointed (v5 binary).
# Triangular XC6 from the D=1280 low state; square YC8 16x8 (close to the ITensor-forum case that crashed at chi~2500 on A100 80GB).
JOBTAG=h100reach; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
B=own_experimental/bin/dmrg_envck_v5
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
# triangular: stored-all limit vs checkpointing
runp t_d1280_k0   "$TRI $OURS init_from=$L2 sched=1280:2 envck=0 memprof=1"
runp t_d1536_k0   "$TRI $OURS init_from=$L2 sched=1536:2 envck=0 memprof=1"
runp t_d2048_k12  "$TRI $OURS init_from=$L2 sched=2048:2 envck=12 memprof=1"
runp t_d2560_2lvl "$TRI $OURS init_from=$L2 sched=2560:2 envck=36 envck2=6 memprof=1"
# square YC8: stored-all vs checkpointed, growing D
runp s_d2048_k0   "$SQ $OURS sched=256:2,1024:2,2048:2 envck=0"
runp s_d2560_k12  "$SQ $OURS sched=256:2,1024:2,2048:1,2560:2 envck=12"
runp s_d3072_2lvl "$SQ $OURS sched=256:2,1024:2,2048:1,3072:2 envck=36 envck2=6"
echo "end=$(date)"
