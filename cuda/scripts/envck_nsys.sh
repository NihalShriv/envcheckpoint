#!/bin/bash
#SBATCH --job-name=envck_nsys
#SBATCH --output=$SCRATCH/dmrg_envck/logs/%x_%j.log
#SBATCH --error=$SCRATCH/dmrg_envck/logs/%x_%j.err
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=00:30:00
# GPU idle fraction during sweeps (decides whether background environment rebuilds can be hidden).
# Busy time = sum of kernel + memcpy/memset time (single stream, so no double counting); wall = sum of sweep times.
JOBTAG=nsys; source ~/DMRG_envck/own_experimental/scripts/envck_hdr.sh
NSYS=/apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/cuda-12.1.0-s5o57xpc7sqqhfd6sxfealzbqxtmvh7p/bin/nsys
prof() {  # tag args
  rm -f $O/$1.*; echo "--- $1: $2 $(date +%T)"
  $NSYS profile -t cuda -s none --cpuctxsw=none -o $O/$1 -f true $B $2 ckpt=$O/$1.ckpt out=$O/$1.csv > $O/$1.out 2>&1 || echo "FAILED $1"
  $NSYS stats -q -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv -o $O/$1 $O/$1.nsys-rep > /dev/null 2>&1
  python3 - $O/$1 <<'PY'
import csv, glob, sys
p = sys.argv[1]
def tot(f):
    try: return sum(float(r["Total Time (ns)"]) for r in csv.DictReader(open(f))) / 1e9
    except Exception as e: return float("nan")
k = tot(glob.glob(p + "_cuda_gpu_kern_sum*.csv")[0]); m = tot(glob.glob(p + "_cuda_gpu_mem_time_sum*.csv")[0])
rows = list(csv.DictReader(open(p + ".csv"))); wall = sum(float(r["sweep_s"]) for r in rows)
top = sorted(csv.DictReader(open(glob.glob(p + "_cuda_gpu_kern_sum*.csv")[0])), key=lambda r: -float(r["Total Time (ns)"]))[:6]
print(f"  kernels={k:.1f}s memcpy={m:.1f}s sweeps_wall={wall:.1f}s  busy~{100*(k+m)/wall:.0f}% (incl. init/final, upper bound)")
for r in top: print(f"    {float(r['Total Time (ns)'])/1e9:7.2f}s {r['Instances']:>7} {r['Name'][:90]}")
PY
  rm -f $O/$1.ckpt $O/$1.sqlite; }
prof n_k0  "$TRI $OURS init_from=$L1 sched=1024:3 envck=0"
prof n_k12 "$TRI $OURS init_from=$L1 sched=1024:3 envck=12"
prof n_std_k12 "$TRI $STD init_from=$L1 sched=1024:2 envck=12"
echo "end=$(date)"
