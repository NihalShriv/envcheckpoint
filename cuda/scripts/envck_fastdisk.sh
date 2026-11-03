#!/bin/bash
#SBATCH --job-name=fastdisk
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=02:30:00
# Strongest offload baselines: async offload with prefetch to node-local disk and to host RAM (/dev/shm = no disk at
# all, the best case for any host/unified-memory offload), vs recomputation. Triangular XC6 D=1024 and 2048 (v5).
JOBTAG=fastdisk; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v5
df -h /tmp /dev/shm ${TMPDIR:-/tmp} /scratch 2>/dev/null; lsblk -d -o NAME,ROTA,SIZE,MODEL 2>/dev/null | head
disk() { # tag dir args
  rm -rf $2; mkdir -p $2 || { echo "SKIP $1 (cannot use $2)"; return; }
  run $1 "$3 envck=1000 envdisk=$2 envdisk_async=1"; du -sh $2; rm -rf $2; }
for D in 1024 2048; do
  X="$TRI $OURS init_from=$L1 sched=$D:2"
  [ $D = 2048 ] && X="$TRI $OURS init_from=$L2 sched=2048:2"
  run rec_d$D "$X envck=12"
  disk local_d$D /tmp/envck_$SLURM_JOB_ID "$X"
  disk shm_d$D /dev/shm/envck_$SLURM_JOB_ID "$X"
  disk scratch_d$D $SCRATCH/dmrg_envck/diskenv_fd "$X"
done
echo "end=$(date)"
