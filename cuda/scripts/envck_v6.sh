#!/bin/bash
#SBATCH --job-name=envck_v6
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:45:00
# v6: chunked H_eff intermediates (heff_chunks=c). A) agreement with c=1 (summation order only) + cost;
# B) reach beyond D=3072 on one H200.
JOBTAG=v6; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
B=own_experimental/bin/dmrg_envck_v6
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
for c in 1 4 8; do runp a_c$c "$TRI $OURS init_from=$L1 sched=1024:3 envck=12 memprof=1 heff_chunks=$c"; done
runp b_tri_d3072_k12_c8  "$TRI $OURS init_from=$L2 sched=3072:2 envck=12 memprof=1 heff_chunks=8"
runp b_tri_d3584_2lvl_c8 "$TRI $OURS init_from=$L2 sched=3584:2 envck=36 envck2=6 memprof=1 heff_chunks=8"
runp b_sq_d3584_k12_c8   "$SQ $OURS sched=256:2,1024:2,2048:1,3584:2 envck=12 heff_chunks=8"
echo "end=$(date)"
