#!/bin/bash
#SBATCH --job-name=envck_chunk
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:30:00
# H_eff chunk count: memory saved vs time cost (v6).
JOBTAG=chunk; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
for c in 1 2 4 8 16; do runp d2048_c$c "$TRI $OURS init_from=$L2 sched=2048:2 envck=12 memprof=1 heff_chunks=$c"; done
for c in 4 8 16; do runp d3072_c$c "$TRI $OURS init_from=$L2 sched=3072:1 envck=36 envck2=6 memprof=1 heff_chunks=$c"; done
echo "end=$(date)"
