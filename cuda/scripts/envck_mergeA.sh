#!/bin/bash
#SBATCH --job-name=envck_mergeA
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=05:00:00
# merged job (submit limit): parts run one after another, each with its own results folder.
echo '##### part: envck_kbig.sh'
# k scan at the limit (v6, chunked H_eff): sqrt(N) optimum at D=2048 and 3072.
JOBTAG=kbig; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
for k in 6 12 24 36; do runp d2048_k$k "$TRI $OURS init_from=$L2 sched=2048:2 envck=$k memprof=1 heff_chunks=4"; done
for k in 12 24; do runp d3072_k$k "$TRI $OURS init_from=$L2 sched=3072:2 envck=$k memprof=1 heff_chunks=8"; done
echo '##### part: envck_chunk.sh'
# H_eff chunk count: memory saved vs time cost (v6).
JOBTAG=chunk; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v6
runp() { run "$@"; grep -h MEMPROF $O/$1.out | tail -1; }
for c in 1 2 4 8 16; do runp d2048_c$c "$TRI $OURS init_from=$L2 sched=2048:2 envck=12 memprof=1 heff_chunks=$c"; done
for c in 4 8 16; do runp d3072_c$c "$TRI $OURS init_from=$L2 sched=3072:1 envck=36 envck2=6 memprof=1 heff_chunks=$c"; done
echo '##### part: envck_stab.sh'
# 20 sweeps, all-stored vs checkpointed, same seed: energies must stay identical sweep by sweep (no drift).
JOBTAG=stab; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v5
for k in 0 12; do run d1536_k$k "$TRI $OURS init_from=$L1 sched=1536:20 envck=$k"; done
paste -d' ' <(cut -d, -f8,11 $O/d1536_k0.csv) <(cut -d, -f11 $O/d1536_k12.csv) | awk '{split($1,a,","); d=a[2]-$2; if(d<0)d=-d; if(d>m)m=d} END{print "max |E_k0-E_k12| over sweeps =", m}'
echo "end=$(date)"
