#!/bin/bash
#SBATCH --job-name=r48s
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
# Rerun of robust.sh stored ladders: random start at the target D (loading the D=1280 state OOMs on a capped card).
# Checkpointed ladders in robust.sh loaded the D=1280 state (workspace sized for 1280), so this is conservative for checkpointing.
JOBTAG=robust; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_cap
ladder() { local t=$1 a=$2; shift 2; for D in "$@"; do run ${t}_$D "${a//@D@/$D}"
  if grep -q "^final" $O/${t}_$D.out; then echo "PASS $t D=$D $(grep -h MEMPROF $O/${t}_$D.out | tail -1)"; else echo "FAIL $t D=$D"; break; fi; done; }
ST="$TRI $OURS sched=@D@:2 envck=0 memprof=1"
ladder r48s "$ST gpu_cap_gb=48" 896 1024 1152 1280
ladder r24c "$TRI $OURS sched=@D@:2 envck=36 envck2=6 heff_chunks=8 memprof=1 gpu_cap_gb=24" 1408
echo "end=$(date)"
