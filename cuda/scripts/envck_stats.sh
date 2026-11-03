#!/bin/bash
#SBATCH --job-name=envck_stats
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:30:00
# Paper error bars + stored-all limit on the square benchmark (v5).
# A) overhead repeatability: all-stored vs k=12, 3 alternating repeats, both solvers (triangular D=1024, 3 sweeps).
# B) square YC8: largest D with all environments stored on one H200 (expected limit ~2048-2560).
JOBTAG=stats; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
B=own_experimental/bin/dmrg_envck_v5
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
for r in 1 2 3; do for k in 0 12; do
  run r${r}_ours_k$k "$TRI $OURS init_from=$L1 sched=1024:3 envck=$k"
  run r${r}_std_k$k  "$TRI $STD  init_from=$L1 sched=1024:3 envck=$k"
done; done
run s_stored_d2048 "$SQ $OURS sched=256:2,1024:2,2048:2 envck=0"; grep -h MEMPROF $O/s_stored_d2048.out | tail -1
run s_stored_d2560 "$SQ $OURS sched=256:2,1024:2,2048:1,2560:2 envck=0"; grep -h MEMPROF $O/s_stored_d2560.out | tail -1
echo "end=$(date)"
