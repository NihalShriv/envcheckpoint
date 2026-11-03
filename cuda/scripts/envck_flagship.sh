#!/bin/bash
#SBATCH --job-name=envck_flag
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=03:00:00
# Flagship probe: one-GPU D ladder on the standard J1-J2 square benchmark (YC8 16x8, J2=0.5) up to D=3072 with
# checkpointed environments (v5), standard solver. Energy and discarded weight per D -> extrapolation in truncation error.
JOBTAG=flagship; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
B=own_experimental/bin/dmrg_envck_v5
SQ="model=cyl Lx=16 Ly=8 J2=0.5 thresh=0 seed=1 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
run sq8_ladder "$SQ $STD sched=256:3,512:3,1024:4,1536:4,2048:4,2560:4,3072:5 envck=12"
grep -h MEMPROF $O/sq8_ladder.out | tail -3
python3 - $O/sq8_ladder.csv <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]))); last = {}
for r in rows: last[int(r["D_cap"])] = r
print("D, E_last, sum_disc, sweep_s, peak_GB")
for D in sorted(last): r = last[D]; print(D, r["energy"], r["sum_disc"], r["sweep_s"], r["peak_mem_gb"])
PY
echo "end=$(date)"
