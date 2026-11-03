#!/bin/bash
#SBATCH --job-name=envck_h100edge
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
# 80 GB card at the edge with chunked H_eff (v6).
JOBTAG=h100edge; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
runp t_d2560_2lvl_c8 "$TRI $OURS init_from=$L2 sched=2560:2 envck=36 envck2=6 memprof=1 heff_chunks=8"
runp t_d3072_2lvl_c8 "$TRI $OURS init_from=$L2 sched=3072:1 envck=36 envck2=6 memprof=1 heff_chunks=8"
runp s_d3072_k12_c8  "$SQ $OURS sched=256:2,1024:2,2048:1,3072:2 envck=12 heff_chunks=8"
echo "end=$(date)"
