#!/bin/bash
#SBATCH --job-name=robust
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=280G
#SBATCH --time=10:00:00
# Paper robustness, one job. GPU (foreground): reach on emulated 24/40/48 GB cards (dmrg_cap gpu_cap_gb=, ballast).
# CPU (two background streams, one store per process): S1 = ITensor real-time TDVP then MPSKit DMRG2;
# S2 = ITensor ab initio DMRG on an H16 chain. Timings in S1/S2 share the node and are indicative only.
JOBTAG=robust; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh; B=own_experimental/bin/dmrg_cap
E=~/DMRG_envck/own_experimental; R=$SCRATCH/dmrg_envck/robust; mkdir -p $R
export JULIA_DEPOT_PATH=$HOME/.julia JULIA_NUM_THREADS=1 NBLAS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_HOME=/apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/cuda-12.1.0-s5o57xpc7sqqhfd6sxfealzbqxtmvh7p

s1() { cd ~/DMRG_v6
  for s in proj ckpt; do STORE=$s MODE=tdvp LX=12 MAXDIM=768 NSTEPS=30 DT=0.05 timeout 150m \
      julia --project=itensor $E/itensor/bench2.jl $R/tdvp_$s > $R/tdvp_$s.out 2>&1; echo "S1 tdvp_$s exit $? $(date +%T)"; done
  cd ~/DMRG_envck/julia_mpskit
  for s in proj ckpt; do STORE=$s LX=24 DS=128,256,512,768 timeout 100m \
      julia --project=. $E/mpskit/bench.jl $R/mpskit_$s.csv > $R/mpskit_$s.out 2>&1; echo "S1 mpskit_$s exit $? $(date +%T)"; done; }
s2() { ~/DMRG_envck/conda_tn/bin/python $E/itensor/chem_ints.py 16 $R/h16.txt > $R/h16_ints.out 2>&1; cd ~/DMRG_v6
  for s in proj ckpt; do STORE=$s MODE=chem DS=128,256,512 timeout 240m \
      julia --project=itensor $E/itensor/bench2.jl $R/chem_$s $R/h16.txt > $R/chem_$s.out 2>&1; echo "S2 chem_$s exit $? $(date +%T)"; done; }
s1 > $R/s1.log 2>&1 &
s2 > $R/s2.log 2>&1 &

# GPU: ladders as in bis_h200.sh, stop at first failure; PASS/FAIL lines
ladder() { local t=$1 a=$2; shift 2; for D in "$@"; do run ${t}_$D "${a//@D@/$D}"
  if grep -q "^final" $O/${t}_$D.out; then echo "PASS $t D=$D $(grep -h MEMPROF $O/${t}_$D.out | tail -1)"; else echo "FAIL $t D=$D"; break; fi; done; }
run cap_smoke "$TRI $OURS init_from=$L2 sched=512:1 envck=0 memprof=1 gpu_cap_gb=24"
grep -h "gpu_cap_gb\|^final" $O/cap_smoke.out
grep -q "ballast_gb" $O/cap_smoke.out || echo "CAP SMOKE: no ballast line"
ST="$TRI $OURS init_from=$L2 sched=@D@:2 envck=0 memprof=1"
CK="$TRI $OURS init_from=$L2 sched=@D@:2 envck=36 envck2=6 heff_chunks=8 memprof=1"
ladder c24s "$ST gpu_cap_gb=24" 384 512 640 768 896 1024
ladder c24c "$CK gpu_cap_gb=24" 768 896 1024 1152 1280 1408 1536 1664 1792
ladder c40s "$ST gpu_cap_gb=40" 768 896 1024 1152 1280
ladder c40c "$CK gpu_cap_gb=40" 1408 1536 1664 1792 1920 2048 2176 2304 2432
ladder c48s "$ST gpu_cap_gb=48" 896 1024 1152 1280 1408
ladder c48c "$CK gpu_cap_gb=48" 1664 1792 1920 2048 2176 2304 2432 2560 2688
echo "GPU done $(date)"
wait
cat $R/s1.log $R/s2.log
for f in $R/*.csv; do echo "== $f"; cat $f; done
echo "end=$(date)"
