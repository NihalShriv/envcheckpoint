# Extra MPSKit checks: expectation values from checkpointed environments vs freshly built ones, mid-run and at the
# end; error on unsupported (infinite) states is MPSKit's own (checkpointed only wraps FiniteEnvironments).
using MPSKit, MPSKitModels, TensorKit, Random
include(joinpath(@__DIR__, "..", "julia", "mpskit", "CheckpointedEnvs.jl")); using .CheckpointedEnvs

L = 24
H = heisenberg_XXX(U1Irrep, FiniteChain(L); spin = 1)
Random.seed!(1)
ψ = FiniteMPS(rand, ComplexF64, L, physicalspace(H)[1], U1Space(0 => 4, 1 => 2, -1 => 2))
envs = checkpointed(environments(ψ, H, ψ); k = 4)
alg = DMRG2(; trunc = truncrank(48), maxiter = 1, verbosity = 0)
maxdiff = 0.0
for sw in 1:6
    global ψ, envs, maxdiff
    ψ, envs, _ = find_groundstate(ψ, H, alg, envs)
    e_ck = real(expectation_value(ψ, H, envs))
    e_fresh = real(expectation_value(ψ, H, environments(ψ, H, ψ)))
    maxdiff = max(maxdiff, abs(e_ck - e_fresh))
end
println("MPSKit expectation_value: max |E(checkpointed envs) - E(fresh envs)| over 6 sweeps = ", maxdiff)
