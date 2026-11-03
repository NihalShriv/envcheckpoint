#!/bin/bash
#SBATCH --job-name=envck_stab
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:30:00
# 20 sweeps, all-stored vs checkpointed, same seed: energies must stay identical sweep by sweep (no drift).
JOBTAG=stab; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_envck_v5
for k in 0 12; do run d1536_k$k "$TRI $OURS init_from=$L1 sched=1536:20 envck=$k"; done
paste -d' ' <(cut -d, -f8,11 $O/d1536_k0.csv) <(cut -d, -f11 $O/d1536_k12.csv) | awk '{split($1,a,","); d=a[2]-$2; if(d<0)d=-d; if(d>m)m=d} END{print "max |E_k0-E_k12| over sweeps =", m}'
echo "end=$(date)"
