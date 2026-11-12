#!/bin/bash
#SBATCH --job-name=common1536
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# Sweep times at a common D=1536 (triangular XC6, Davidson-2, three sweeps from the D=1280 state), v8 build with
# caching: stored (envcache=1) with and without chunking, one-level k=12 and two-level k=36/k2=6 with 8 chunks,
# both with retained checkpoints. Quote sweeps 2 and 3.
JOBTAG=common1536; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
B=${SCRATCH:-/scratch/$USER}/cuda_v8/cuda/bin/dmrg_envck
run c1536_stored    "$TRI $OURS init_from=$L2 sched=1536:3 envck=0 envcache=1 memprof=1"
run c1536_stored_c8 "$TRI $OURS init_from=$L2 sched=1536:3 envck=0 envcache=1 heff_chunks=8 memprof=1"
run c1536_k12       "$TRI $OURS init_from=$L2 sched=1536:3 envck=12 ck_retain=1 memprof=1"
run c1536_2lvl_c8   "$TRI $OURS init_from=$L2 sched=1536:3 envck=36 envck2=6 heff_chunks=8 ck_retain=1 memprof=1"
for t in c1536_stored c1536_stored_c8 c1536_k12 c1536_2lvl_c8; do
  grep -q "^final" $O/$t.out && echo "PASS $t $(grep -h MEMPROF $O/$t.out | tail -1)" || echo "FAIL $t"; done
echo "end=$(date)"
