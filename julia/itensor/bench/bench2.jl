# CPU benchmarks of CheckpointedProjMPO vs ProjMPO (one store per process, so Sys.maxrss is that store's peak).
#   MODE=tdvp : real-time 2-site TDVP quench from Neel, Heisenberg YC6 (Lx x 6), U(1); MAXDIM, NSTEPS, DT
#   MODE=chem : ab initio DMRG from an integral file (chem_ints.py), U(1)xU(1) electrons; DS = maxdim ladder
# Usage: STORE=proj|ckpt MODE=... julia --project=<itensor env> bench2.jl <out_prefix> [intfile]
using ITensors, ITensorMPS, Printf, Random, LinearAlgebra
include(joinpath(@__DIR__, "CheckpointedProjMPOs.jl")); using .CheckpointedProjMPOs
function j1j2_bonds(Lx, Ly, J2)
    s(x, y) = x * Ly + mod(y, Ly) + 1
    b = Tuple{Int,Int,Float64}[]
    for x in 0:Lx-1, y in 0:Ly-1
        push!(b, (s(x, y), s(x, y + 1), 1.0))
        x + 1 < Lx || continue
        push!(b, (s(x, y), s(x + 1, y), 1.0)); push!(b, (s(x, y), s(x + 1, y + 1), J2)); push!(b, (s(x, y), s(x + 1, y - 1), J2))
    end
    b
end
function hamiltonian(sites, bl)
    os = OpSum()
    for (i, j, J) in bl
        os += J, "Sz", i, "Sz", j; os += J / 2, "S+", i, "S-", j; os += J / 2, "S-", i, "S+", j
    end
    MPO(Float64, os, sites)
end
neel(N) = [isodd(n) ? "Up" : "Dn" for n in 1:N]

BLAS.set_num_threads(parse(Int, get(ENV, "NBLAS", "4")))
out = ARGS[1]; mode = ENV["MODE"]; store = ENV["STORE"]
proj(H) = store == "ckpt" ? CheckpointedProjMPO(H) : store == "ckpt_nogc" ? CheckpointedProjMPO(H; after_drop = () -> nothing) : ProjMPO(H)
rss() = Sys.maxrss() / 1e9
function main()
io = open(out * ".csv", "w")

if mode == "tdvp"
    Lx = parse(Int, get(ENV, "LX", "24")); Ly = 6; N = Lx * Ly
    D = parse(Int, get(ENV, "MAXDIM", "1024")); nst = parse(Int, get(ENV, "NSTEPS", "40")); dt = parse(Float64, get(ENV, "DT", "0.05"))
    sites = siteinds("S=1/2", N; conserve_qns = true)
    H = hamiltonian(sites, j1j2_bonds(Lx, Ly, 0.0))
    psi = MPS(ComplexF64, sites, neel(N))
    println(io, "store,step,t,maxlinkdim,energy,step_s,maxrss_gb,gc_s"); flush(io)
    for s in 1:nst
        t0 = time()
        st = @timed tdvp(proj(H), -im * dt, psi; nsteps = 1, nsite = 2, maxdim = D, cutoff = 1e-12, outputlevel = 0)
        psi = st.value
        E = real(inner(psi', H, psi))
        @printf(io, "%s,%d,%.3f,%d,%.12f,%.2f,%.2f,%.2f\n", store, s, s * dt, maxlinkdim(psi), E, time() - t0, rss(), st.gctime); flush(io)
        @printf("%s step %d t=%.2f chi=%d E=%.12f %.1fs rss=%.1fGB\n", store, s, s * dt, maxlinkdim(psi), E, time() - t0, rss())
        flush(stdout)
    end
else   # chem
    f = open(ARGS[2]); n, nel = parse.(Int, split(readline(f))[1:2]); ecore = parse(Float64, split(readline(f))[1])
    os = OpSum(); spins = ("up", "dn")
    t0 = time()
    for ln in eachline(f)
        w = split(ln)
        if w[1] == "h"
            i, j = parse(Int, w[2]), parse(Int, w[3]); v = parse(Float64, w[4])
            for s in spins; os += v, "Cdag$s", i, "C$s", j; end
        else   # (ij|kl) chemists' notation: 1/2 sum v c+_is c+_kt c_lt c_js
            i, j, k, l = parse.(Int, w[2:5]); v = parse(Float64, w[6])
            for s in spins, t in spins
                (s == t && (i == k || j == l)) && continue
                os += 0.5 * v, "Cdag$s", i, "Cdag$t", k, "C$t", l, "C$s", j
            end
        end
    end
    close(f)
    sites = siteinds("Electron", n; conserve_qns = true)
    H = MPO(os, sites)
    @printf("MPO built in %.0fs, W=%d, rss=%.1fGB\n", time() - t0, maxlinkdim(H), rss()); flush(stdout)
    Random.seed!(1); psi = random_mps(sites, [i <= nel ÷ 2 ? "UpDn" : "Emp" for i in 1:n]; linkdims = 16)
    Ds = parse.(Int, split(get(ENV, "DS", "128,256,512,1024"), ","))
    println(io, "store,D,sweep,energy,sweep_s,maxrss_gb,W"); flush(io)
    for D in Ds, s in 1:2
        t0 = time()
        E, psi2 = dmrg(proj(H), psi, Sweeps(1; maxdim = D, cutoff = 0.0); outputlevel = 0)
        psi = psi2
        @printf(io, "%s,%d,%d,%.12f,%.2f,%.2f,%d\n", store, D, s, E + ecore, time() - t0, rss(), maxlinkdim(H)); flush(io)
        @printf("%s D=%d sweep %d E=%.12f %.1fs rss=%.1fGB\n", store, D, s, E + ecore, time() - t0, rss()); flush(stdout)
    end
end
close(io)
end
main()
