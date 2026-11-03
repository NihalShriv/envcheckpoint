#!/bin/bash
#SBATCH --job-name=envck_v5
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
# v5 (separate tree ~/DMRG_envck): Davidson basis sized to dav_maxmv (was 160 columns), workspace fits loaded states.
# A) regression: energies must equal v3 exactly; B) restart to smaller D (crashed before); C) reach D=2048..3072.
JOBTAG=v5; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
B=own_experimental/bin/dmrg_envck_v5
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
# A) regression (v3: tri E=-73.1635923488 peak 29.3 GB; square E=-63.3960435775 peak 24.6 GB)
runp a_tri_k12 "$TRI $OURS init_from=$L1 sched=1024:3 envck=12 memprof=1"
runp a_tri_k0  "$TRI $OURS init_from=$L1 sched=1024:3 envck=0 memprof=1"
runp a_sq_k12  "$SQ $OURS sched=128:2,256:2,512:3,1024:4 envck=12"
# B) restart to smaller D (completes the cost-model D scan)
for D in 512 768; do for k in 0 12; do run b_tri_d${D}_k$k "$TRI $OURS init_from=$L1 sched=$D:2 envck=$k"; done; done
# C) reach on one H200 (from the D=1280 low state)
runp c_tri_d2048_k12  "$TRI $OURS init_from=$L2 sched=2048:2 envck=12 memprof=1"
runp c_tri_d2048_std  "$TRI $STD  init_from=$L2 sched=2048:2 envck=12 memprof=1"
runp c_tri_d2560_k12  "$TRI $OURS init_from=$L2 sched=2560:2 envck=12 memprof=1"
runp c_tri_d3072_k12  "$TRI $OURS init_from=$L2 sched=3072:2 envck=12 memprof=1"
runp c_tri_d3072_2lvl "$TRI $OURS init_from=$L2 sched=3072:2 envck=36 envck2=6 memprof=1"
runp c_sq_d3072_k12   "$SQ $OURS sched=256:2,1024:2,2048:1,3072:2 envck=12"
echo "end=$(date)"
