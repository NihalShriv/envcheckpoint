#!/bin/bash
#SBATCH --job-name=envck_kbig
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:45:00
# k scan at the limit (v6, chunked H_eff): sqrt(N) optimum at D=2048 and 3072.
JOBTAG=kbig; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
for k in 6 12 24 36; do runp d2048_k$k "$TRI $OURS init_from=$L2 sched=2048:2 envck=$k memprof=1 heff_chunks=4"; done
for k in 12 24; do runp d3072_k$k "$TRI $OURS init_from=$L2 sched=3072:2 envck=$k memprof=1 heff_chunks=8"; done
echo "end=$(date)"
