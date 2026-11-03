# sourced: common setup for envck jobs (separate tree ~/DMRG_envck, binary v3, memtrack on)
set -uo pipefail
cd ~/DMRG_envck; O=own_experimental/results/$JOBTAG; mkdir -p $O
B=own_experimental/bin/dmrg_envck_v3; T=~/DMRG_v6/own_experimental/results/tri2
nvidia-smi --query-gpu=name --format=csv,noheader; echo "start=$(date)"
OURS="svd=warm warm_iters=1 dav_maxmv=2 dav_fast=1"; STD="svd=gram dav_maxmv=4 dav_fast=1"
TRI="model=tric Lx=24 Ly=6 J2=0.12 seed=1 thresh=0 heff=cutensor compress=1 production=1 memtrack=1"
L1=$T/v3_stab_std_from_low.ckpt; L2=$T/d1280_ours_x.ckpt
run() { rm -f $O/$1.*; echo "--- $1: $2 $(date +%T)"; $B $2 ckpt=$O/$1.ckpt out=$O/$1.csv > $O/$1.out 2>&1 || echo "RUN FAILED $1: $(grep -h 'what()\|ERROR\|out of memory' $O/$1.out | tail -1)"
  cut -d, -f8,9,11,15,22 $O/$1.csv | tail -n +2 | sed 's/^/  sw,D,E,t,peakGB: /'; grep -h "^final" $O/$1.out; rm -f $O/$1.ckpt; }
