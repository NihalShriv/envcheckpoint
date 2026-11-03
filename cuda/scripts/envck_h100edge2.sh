#!/bin/bash
#SBATCH --job-name=h100edge2
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:30:00
# 80 GB edge, settled: square YC8 D=3072 (two-level + 16 chunks) and triangular D=3072 for 3 sweeps (v6).
JOBTAG=h100edge2; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
runp s_d3072_2lvl_c16 "$SQ $OURS sched=256:2,1024:2,2048:1,3072:3 envck=32 envck2=6 heff_chunks=16"
runp t_d3072_2lvl_c16 "$TRI $OURS init_from=$L2 sched=3072:3 envck=36 envck2=6 memprof=1 heff_chunks=16"
echo "end=$(date)"
