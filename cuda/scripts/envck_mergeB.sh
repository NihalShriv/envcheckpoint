#!/bin/bash
#SBATCH --job-name=envck_mergeB
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=08:00:00
# merged job (submit limit): parts run one after another, each with its own results folder.
echo '##### part: envck_xc8big.sh'
# Widest MPO (XC8-24, W=50) at D=3072 on one H200: two-level checkpointing + chunked H_eff (v6).
JOBTAG=xc8big; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
X8="model=tric Lx=24 Ly=8 J2=0.12 seed=1 thresh=0 heff=cutensor compress=1 production=1 memtrack=1 memprof=1"
S=~/DMRG_v6/own_experimental/results/tri2/x8_ours_fromJ0.ckpt
run x8_d3072_2lvl_c8 "$X8 $OURS init_from=$S sched=3072:2 envck=48 envck2=7 heff_chunks=8"; grep -h MEMPROF $O/x8_d3072_2lvl_c8.out | tail -1
echo '##### part: envck_trilad.sh'
# Triangular XC6-24 low state, standard solver, D ladder to 3072 on one H200 (two-level + chunked) for extrapolation.
JOBTAG=trilad; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
run tri_ladder "$TRI $STD init_from=$L1 sched=1024:3,1536:4,2048:4,2560:4,3072:5 envck=36 envck2=6 heff_chunks=8 memprof=1"
python3 - $O/tri_ladder.csv <<'PY'
import csv, sys
last = {}
for r in csv.DictReader(open(sys.argv[1])): last[int(r["D_cap"])] = r
for D in sorted(last): r = last[D]; print(D, r["energy"], r["sum_disc"], r["sweep_s"], r["peak_mem_gb"])
PY
echo "end=$(date)"
