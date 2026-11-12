#!/bin/bash
#SBATCH --job-name=robust_st
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
# Rerun of robust.sh stored ladders: random start at the target D (loading the D=1280 state OOMs on a capped card).
# Checkpointed ladders in robust.sh loaded the D=1280 state (workspace sized for 1280), so this is conservative for checkpointing.
JOBTAG=robust; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_cap
ladder() { local t=$1 a=$2; shift 2; for D in "$@"; do run ${t}_$D "${a//@D@/$D}"
  if grep -q "^final" $O/${t}_$D.out; then echo "PASS $t D=$D $(grep -h MEMPROF $O/${t}_$D.out | tail -1)"; else echo "FAIL $t D=$D"; break; fi; done; }
ST="$TRI $OURS sched=@D@:2 envck=0 memprof=1"
ladder r24s "$ST gpu_cap_gb=24" 256 384 512 640 768 896 1024
ladder r40s "$ST gpu_cap_gb=40" 512 640 768 896 1024 1152 1280
ladder r48s "$ST gpu_cap_gb=48" 640 768 896 1024 1152 1280 1408
echo "end=$(date)"
