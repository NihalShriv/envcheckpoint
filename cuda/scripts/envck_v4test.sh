#!/bin/bash
#SBATCH --job-name=envck_v4
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:50:00
# v4 features at D=1024 (triangular XC6-24 low state, ours): memory breakdown (memprof), two-level schedule (envck2,
# exactness + memory/time), async disk offload with prefetch vs synchronous disk vs recompute.
JOBTAG=v4test; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
B=own_experimental/bin/dmrg_envck_v4; DK=$SCRATCH/dmrg_envck/diskenv_v4
X="$TRI $OURS init_from=$L1 sched=1024:3 memprof=1"
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
runp p_k0   "$X envck=0"
runp p_k12  "$X envck=12"
runp p_k36_6 "$X envck=36 envck2=6"
runp p_k48_7 "$X envck=48 envck2=7"
runp p_k72_9 "$X envck=72 envck2=9"
for a in 0 1; do rm -rf $DK; mkdir -p $DK; runp p_disk_async$a "$TRI $OURS init_from=$L1 sched=1024:2 memprof=1 envck=1000 envdisk=$DK envdisk_async=$a"; rm -rf $DK; done
echo "end=$(date)"
