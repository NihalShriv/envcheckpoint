# MPSKit checkpointed environments: exactness test (DMRG2, U(1) and dense; TDVP) and environment count.
using MPSKit, MPSKitModels, TensorKit, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "julia", "mpskit", "CheckpointedEnvs.jl")); using .CheckpointedEnvs

function run(H, ψ0, alg, mk; nsweeps = 6)
    ψ = copy(ψ0)
    envs = mk(environments(ψ, H, ψ))
    Es = Float64[]
    for _ in 1:nsweeps
        ψ, envs, _ = find_groundstate(ψ, H, alg, envs)
        push!(Es, real(expectation_value(ψ, H, envs)))
    end
    return Es, ψ, envs
end
nstored(e) = count(!isnothing, e.GLs) + count(!isnothing, e.GRs)

for (name, sym) in (("U(1)", U1Irrep), ("dense", Trivial))
    L = 24
    H = heisenberg_XXX(sym, FiniteChain(L); spin = 1)   # spin-1: integer U(1) charges, simple virtual spaces
    Random.seed!(1)
    ψ0 = FiniteMPS(rand, ComplexF64, L, physicalspace(H)[1], sym === Trivial ? ℂ^16 : U1Space(0 => 4, 1 => 2, -1 => 2))
    alg = DMRG2(; trunc = truncrank(48), maxiter = 1, verbosity = 0)
    ref, _, e0 = run(H, ψ0, alg, identity)
    for k in (3, 5)
        res, _, ek = run(H, ψ0, alg, e -> checkpointed(e; k))
        println("$name DMRG2 k=$k: max |dE| = ", maximum(abs.(res .- ref)), "  stored envs ",
                count(!isnothing, e0.GLs) + count(!isnothing, e0.GRs), " -> ", nstored(ek))
    end
    # TDVP (real time, two-site)
    ψa, _ = timestep(copy(ψ0), H, 0.0, 0.05, TDVP2(; trunc = truncrank(48)), environments(ψ0, H, ψ0))
    ψb, _ = timestep(copy(ψ0), H, 0.0, 0.05, TDVP2(; trunc = truncrank(48)), checkpointed(environments(ψ0, H, ψ0); k = 4))
    println("$name TDVP2 one step: |1 - |<a|b>|| = ", abs(1 - abs(dot(ψa, ψb)) / (norm(ψa) * norm(ψb))))
end
