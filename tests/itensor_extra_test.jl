# Extra ITensor checks (CPU): one-site TDVP, two-level schedule with k not a multiple of k2 (rebuild count stable),
# copy() independence. Run: julia --project=<itensor env> test_extra.jl
using ITensors, ITensorMPS, Random
include(joinpath(@__DIR__, "..", "julia", "itensor", "CheckpointedProjMPOs.jl")); using .CheckpointedProjMPOs

N = 24
s = siteinds("S=1/2", N; conserve_qns = true)
os = OpSum(); for j in 1:N-1; global os; os += "Sz", j, "Sz", j + 1; os += 0.5, "S+", j, "S-", j + 1; os += 0.5, "S-", j, "S+", j + 1; end
H = MPO(os, s)
Random.seed!(1)
psi0 = random_mps(s, [isodd(n) ? "Up" : "Dn" for n in 1:N]; linkdims = 16)

# 1. TDVP with nsite = 1 (needs a state with grown bond dimension first)
_, psi = dmrg(H, psi0; nsweeps = 4, maxdim = 32, cutoff = 0.0, outputlevel = 0)
a = tdvp(ProjMPO(H), -0.2, copy(psi); nsteps = 4, nsite = 1, outputlevel = 0)
b = tdvp(CheckpointedProjMPO(H; k = 4), -0.2, copy(psi); nsteps = 4, nsite = 1, outputlevel = 0)
println("TDVP nsite=1: 1 - |<a|b>|/(|a||b|) = ", abs(1 - abs(inner(a, b)) / (norm(a) * norm(b))))

# 2. two-level with k=7, k2=3: exact and rebuild count per sweep stable (no lost checkpoints)
function sweeps_with(PH)
    psi = copy(psi0); Es = Float64[]; counts = Int[]
    for sw in 1:6
        E, psi = dmrg(PH, psi, Sweeps(1; maxdim = 48, cutoff = 0.0); outputlevel = 0)
        push!(Es, E); hasproperty(PH, :nrebuilt) && push!(counts, PH.nrebuilt)
    end
    Es, counts
end
Eref, _ = sweeps_with(ProjMPO(H))
P = CheckpointedProjMPO(H; k = 7, k2 = 3)
E2, c2 = sweeps_with(P)
println("two-level k=7 k2=3: max|dE| = ", maximum(abs.(E2 .- Eref)), "  rebuilds per sweep = ", diff(c2))

# 3. copy() gives an independent object
P1 = CheckpointedProjMPO(H; k = 4); _, psi1 = dmrg(P1, copy(psi0), Sweeps(2; maxdim = 32, cutoff = 0.0); outputlevel = 0)
P2 = copy(P1); _, psi2 = dmrg(P2, copy(psi1), Sweeps(1; maxdim = 32, cutoff = 0.0); outputlevel = 0)
_, psi3 = dmrg(P1, copy(psi1), Sweeps(1; maxdim = 32, cutoff = 0.0); outputlevel = 0)
println("copy independence: |E(P2) - E(P1)| = ", abs(inner(psi2', H, psi2) - inner(psi3', H, psi3)))
