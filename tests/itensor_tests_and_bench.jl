# CheckpointedProjMPO tests and benchmark (envck tree).
#   MODE=exact : CPU, U(1) and dense, DMRG and imaginary-time TDVP: ProjMPO vs CheckpointedProjMPO must agree.
#   MODE=bench : GPU, square J1-J2 YC8 16x8, maxdim ladder until out of memory, STORE=proj|disk|ckpt, QN=0|1.
# Usage: julia --project=<itensor env> ckpt_bench.jl <out_prefix>
using ITensors, ITensorMPS, Printf, Random
include(joinpath(@__DIR__, "..", "julia", "itensor", "CheckpointedProjMPOs.jl")); using .CheckpointedProjMPOs

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

using Serialization
# offload = (store, load) to files in dir; `todev` moves a loaded (CPU) tensor back to the device
diskstash(dir, todev) = (T -> (f = tempname(dir); serialize(f, ITensors.NDTensors.cpu(T)); f),
                         f -> (x = todev(deserialize(f)); rm(f); x))

struct Log <: AbstractObserver; E::Vector{Float64}; end
ITensorMPS.checkdone!(o::Log; energy, kwargs...) = (push!(o.E, energy); false)

out = ARGS[1]; mode = get(ENV, "MODE", "exact")

if mode == "exact"
    io = open(out * ".txt", "w")
    for qn in (true, false)
        Random.seed!(1)
        sites = siteinds("S=1/2", 6 * 4; conserve_qns = qn)
        H = hamiltonian(sites, j1j2_bonds(6, 4, 0.5))
        psi0 = qn ? random_mps(sites, neel(24); linkdims = 8) : random_mps(sites; linkdims = 8)
        sw = Sweeps(8); maxdim!(sw, 16, 32, 64, 96, 128); cutoff!(sw, 0.0)
        r = Dict{String,Vector{Float64}}()
        st = Dict{String,MPS}()
        for (name, PH) in (("proj", ProjMPO(H)), ("ckpt_k3", CheckpointedProjMPO(H; k = 3)), ("ckpt_k5", CheckpointedProjMPO(H; k = 5)),
                           ("ckpt_2lvl", CheckpointedProjMPO(H; k = 8, k2 = 3)),
                           ("ckpt_offload", CheckpointedProjMPO(H; k = 4, offload = diskstash(mktempdir(), identity))),
                           ("ckpt_chunk3", CheckpointedProjMPO(H; k = 4, chunks = 3)))
            o = Log(Float64[])
            E, psi = dmrg(PH, copy(psi0), sw; observer = o, outputlevel = 0)
            r[name] = o.E; st[name] = psi
            name != "proj" && @printf(io, "qn=%d %s: rebuilt env updates = %d\n", qn, name, PH.nrebuilt)
        end
        for name in ("ckpt_k3", "ckpt_k5", "ckpt_2lvl", "ckpt_offload", "ckpt_chunk3")
            @printf(io, "qn=%d DMRG %s: max |E - E_proj| over %d sweeps = %.3e  final E = %.14f\n", qn, name,
                    length(r[name]), maximum(abs.(r[name] .- r["proj"])), r[name][end])
        end
        # imaginary-time TDVP from the DMRG state with a kick (2-site, exact = same contractions)
        phi0 = st["proj"]
        for (name, PH) in (("proj", ProjMPO(H)), ("ckpt_k3", CheckpointedProjMPO(H; k = 3)))
            st["tdvp_" * name] = tdvp(PH, -0.4, copy(psi0); nsteps = 4, maxdim = 64, cutoff = 0.0, outputlevel = 0)
        end
        a, b = st["tdvp_proj"], st["tdvp_ckpt_k3"]
        @printf(io, "qn=%d TDVP(imag, 4 steps): |1 - <a|b>/(|a||b|)| = %.3e   E_a=%.14f E_b=%.14f\n", qn,
                abs(1 - inner(a, b) / (norm(a) * norm(b))), inner(a', H, a) / inner(a, a), inner(b', H, b) / inner(b, b))
    end
    close(io); print(read(out * ".txt", String))
else
    using CUDA, Adapt
    store = get(ENV, "STORE", "ckpt"); qn = get(ENV, "QN", "0") == "1"
    Ds = parse.(Int, split(get(ENV, "DS", "512,1024,1536,2048,2560,3072"), ","))
    nsw = parse(Int, get(ENV, "NSW", "2")); Lx, Ly = 16, 8; N = Lx * Ly
    Random.seed!(1)
    sites = siteinds("S=1/2", N; conserve_qns = qn)
    H = adapt(CuArray, hamiltonian(sites, j1j2_bonds(Lx, Ly, 0.5)))
    psi = adapt(CuArray, qn ? random_mps(sites, neel(N); linkdims = 64) : random_mps(Float64, sites; linkdims = 64))
    io = open(out * ".csv", "w"); println(io, "store,qn,D,sweep,energy,sweep_s,used_gb"); flush(io)
    # CK_K (default sqrt N), CK_K2 (two-level), CK_RECLAIM=1 (return freed blocks to the driver), CK_OFFLOAD=<dir>
    function ckpt_for(H)
        k = parse(Int, get(ENV, "CK_K", string(round(Int, sqrt(length(H))))))
        drop = get(ENV, "CK_RECLAIM", "0") == "1" ? () -> (GC.gc(false); CUDA.reclaim()) : () -> GC.gc(false)
        off = haskey(ENV, "CK_OFFLOAD") ? diskstash(mkpath(ENV["CK_OFFLOAD"]), x -> adapt(CuArray, x)) : nothing
        pre = get(ENV, "CK_PRESOLVE", "0") == "1" ? () -> (GC.gc(false); CUDA.reclaim()) : () -> nothing
        CheckpointedProjMPO(H; k, k2 = parse(Int, get(ENV, "CK_K2", "0")), after_drop = drop, offload = off,
                            chunks = parse(Int, get(ENV, "CK_CHUNKS", "1")), todev = x -> adapt(CuArray, x),
                            pre_solve = pre)
    end
    function used()   # device memory in use (GB); API differs across CUDA.jl versions
        for f in (() -> CUDA.Mem.info(), () -> CUDA.memory_info(), () -> (CUDA.available_memory(), CUDA.total_memory()))
            try; fr, tot = f(); return (tot - fr) / 1e9; catch; end
        end
        return NaN
    end
    for D in Ds
        sw = Sweeps(nsw); maxdim!(sw, D); cutoff!(sw, 0.0)
        kw = store == "disk" ? (; write_when_maxdim_exceeds = 0, write_path = get(ENV, "WPATH", tempdir())) : (;)
        for s in 1:nsw
            one = Sweeps(1); maxdim!(one, D); cutoff!(one, 0.0)
            t = time()
            ok = try
                PH = store == "ckpt" ? ckpt_for(H) : ProjMPO(H)    # fresh per call: same env rebuild for all
                global psi   # script top level: loop/try bodies are soft scope
                E, psi = dmrg(store == "disk" ? H : PH, psi, one; outputlevel = 0, kw...)
                CUDA.synchronize()
                @printf(io, "%s,%d,%d,%d,%.12f,%.2f,%.2f\n", store, qn, D, s, E, time() - t, used()); flush(io)
                @printf("%s qn=%d D=%d sweep %d E=%.12f t=%.1fs used=%.1fGB\n", store, qn, D, s, E, time() - t, used())
                true
            catch e
                println("FAILED at D=$D sweep $s: ", sprint(showerror, e)[1:min(end, 200)]); false
            end
            ok || (close(io); exit(0))
            flush(stdout)
        end
    end
    close(io)
end
