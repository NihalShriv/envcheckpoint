#!/bin/bash
#SBATCH --job-name=bis_h200
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=09:00:00
JOBTAG=bisect200; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
# ladder tag "args with @D@" D1 D2 ...: run each D (2 sweeps at D), stop at the first failure; "PASS/FAIL D" lines
ladder() { local t=$1 a=$2; shift 2; for D in "$@"; do run ${t}_$D "${a//@D@/$D}"
  if grep -q "^final" $O/${t}_$D.out; then echo "PASS $t D=$D $(grep -h MEMPROF $O/${t}_$D.out | tail -1)"; else echo "FAIL $t D=$D"; break; fi; done; }
# M3: bisected limits on H200 (141 GB)
ladder t0 "$TRI $OURS init_from=$L2 sched=@D@:2 envck=0 memprof=1" 1536 1664 1792 1920
ladder s0 "$SQ $OURS sched=256:2,1024:2,@D@:2 envck=0" 2048 2176 2304 2432 2560
ladder sc "$SQ $OURS sched=256:2,1024:2,2048:1,@D@:2 envck=32 envck2=6 heff_chunks=16" 3584 3712 3840 3968 4096
ladder tc "$TRI $OURS init_from=$L2 sched=@D@:2 envck=36 envck2=6 heff_chunks=8 memprof=1" 3584 3712 3840 3968
echo "end=$(date)"
