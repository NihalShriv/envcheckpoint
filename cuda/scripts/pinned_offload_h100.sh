#!/bin/bash
#SBATCH --job-name=pinned_off_h100
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=03:00:00
# Red-team control: recompute (checkpointing) vs pinned async offload (cudaMemcpyAsync on a separate stream,
# prefetch 2) vs store-all, same binary v7 (stream-ordered allocator), triangular XC6-24, alternating repeats.
# Energies must be identical across the three; host RAM for pinned offload ~ all environments.
JOBTAG=pinned; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v7
free -g | head -2
runm() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; grep -h "VmHWM\|maxrss" $O/$1.out | tail -1; }
for r in 1 2 3; do
  runm d1024_r${r}_store "$TRI $OURS init_from=$L1 sched=1024:3 envck=0 memprof=1"
  runm d1024_r${r}_ckpt  "$TRI $OURS init_from=$L1 sched=1024:3 envck=12 memprof=1"
  runm d1024_r${r}_pin   "$TRI $OURS init_from=$L1 sched=1024:3 envck=1000 envpin=1 memprof=1"
done
for r in 1 2; do
  runm d1280_r${r}_store "$TRI $OURS init_from=$L2 sched=1280:3 envck=0 memprof=1"
  runm d1280_r${r}_ckpt  "$TRI $OURS init_from=$L2 sched=1280:3 envck=12 memprof=1"
  runm d1280_r${r}_pin   "$TRI $OURS init_from=$L2 sched=1280:3 envck=1000 envpin=1 memprof=1"
done
echo "end=$(date)"
