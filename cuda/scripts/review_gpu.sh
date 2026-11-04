#!/bin/bash
#SBATCH --job-name=review_gpu
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Review checks with the released build (cuda/, v7). Short parts first, the long ladder last.
# A  reproduction: overhead pair of job stats (v5) rerun with v7; energies compared with results/raw/stats.
# C  ablation, triangular XC6 D=1024: stored/checkpointed x Davidson basis (sized 8 / unsized 160) x chunks (1/4).
# D  overhead with converged local solves (no matvec cap, basis 32 with restarts).
# B  square YC8 ladder to D=3072 with the original settings (job conv, v6), state kept; then sweeps at D=3072 with
#    no matvec cap. max_hours stops cleanly before a sweep that would not fit in the job.
set -uo pipefail
SCRATCH=${SCRATCH:-/scratch/$USER}
B=$SCRATCH/cuda_reltest/cuda/bin/dmrg_envck
O=$SCRATCH/dmrg_envck/review_gpu; mkdir -p $O
T=$HOME/DMRG_v6/own_experimental/results/tri2; L1=$T/v3_stab_std_from_low.ckpt
STD="svd=gram dav_maxmv=4 dav_fast=1"
TRI="model=tric Lx=24 Ly=6 J2=0.12 seed=1 thresh=0 heff=cutensor compress=1 production=1 memtrack=1"
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
nvidia-smi --query-gpu=name --format=csv,noheader; echo "start=$(date)"; T0=$(date +%s)
run() { echo "--- $1: $2 $(date +%T)"; $B $2 ckpt=$O/$1.ckpt out=$O/$1.csv > $O/$1.out 2>&1; echo "exit=$? $(date +%T)"
  tail -n 2 $O/$1.csv | cut -d, -f8,9,11,15,22,24,25; }
# A
for k in 0 12; do run A_std_k$k "$TRI $STD init_from=$L1 sched=1024:3 envck=$k"; rm -f $O/A_std_k$k.ckpt; done
# C
for k in 0 12; do for cols in 8 160; do for c in 1 4; do
  run C_k${k}_cols${cols}_c$c "$TRI $STD init_from=$L1 sched=1024:2 envck=$k dav_cols=$cols heff_chunks=$c"; rm -f $O/C_k${k}_cols${cols}_c$c.ckpt
done; done; done
# D
for k in 0 12; do run D_conv_k$k "$TRI svd=gram dav_maxmv=0 dav_cols=32 init_from=$L1 sched=1024:3 envck=$k"; rm -f $O/D_conv_k$k.ckpt; done
# B
run B_ladder "$SQ $STD sched=256:6,512:8,1024:16,1536:14,2048:12,2560:10,3072:12 conv=1 stage_tol=1e-6 final_tol=1e-7 kick_tol=1e-7 envck=12 heff_chunks=4"
LEFT=$(python3 -c "print(max(0.2, (8*3600 - ($(date +%s) - $T0) - 900) / 3600))")
run B_strict "$SQ svd=gram dav_maxmv=0 dav_cols=32 init_from=$O/B_ladder.ckpt sched=3072:3 envck=12 heff_chunks=4 max_hours=$LEFT"
echo "end=$(date)"
