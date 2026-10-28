"""
    checkpointed(envs::MPSKit.FiniteEnvironments; k = round(Int, sqrt(N))) -> CheckpointedFiniteEnvironments

Environment manager for finite MPS with the same dependency tracking as MPSKit's `FiniteEnvironments`, but it keeps
only every `k`-th left/right environment (plus the segment next to the one last requested). A dropped environment is
recomputed from the nearest stored valid one when `leftenv` / `rightenv` asks for it, with the same transfer-matrix
contractions, so results are unchanged. Memory ~ (N/k + 2k) environments instead of ~N.

    envs = checkpointed(environments(ψ, H))
    ψ, envs, δ = find_groundstate(ψ, H, DMRG2(; trscheme), envs)
"""
module CheckpointedEnvs

using MPSKit, TensorKit
import MPSKit: leftenv, rightenv, poison!, AbstractMPSEnvironments, TransferMatrix, DefaultBackend,
    DefaultAllocator, AbstractBackend

export checkpointed, CheckpointedFiniteEnvironments

struct CheckpointedFiniteEnvironments{A, B, C, D} <: AbstractMPSEnvironments
    above::A
    operator::B
    ldependencies::Vector{C}      # same meaning as in FiniteEnvironments (state tensor each env was built from)
    rdependencies::Vector{C}
    GLs::Vector{Union{Nothing, D}}  # nothing = dropped (recomputable)
    GRs::Vector{Union{Nothing, D}}
    k::Int
end

function checkpointed(e::MPSKit.FiniteEnvironments; k::Int = max(1, round(Int, sqrt(length(e.GLs) - 1))))
    D = eltype(e.GLs)
    return CheckpointedFiniteEnvironments(e.above, e.operator, e.ldependencies, e.rdependencies,
                                          Vector{Union{Nothing, D}}(e.GLs), Vector{Union{Nothing, D}}(e.GRs), k)
end

function poison!(ca::CheckpointedFiniteEnvironments, ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    return ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

# GLs[i] contracts sites 1..i-1 and depends on ldependencies[1..i-1]; GLs[1] is the boundary (always kept).
function leftenv(ca::CheckpointedFiniteEnvironments, ind, state;
                 backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator())
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind - 1))   # first site whose tensor changed
    valid_upto = isnothing(a) ? ind : a                                          # GLs[1..valid_upto] are correct if stored
    if ca.GLs[ind] === nothing || valid_upto < ind
        s = valid_upto
        while ca.GLs[s] === nothing; s -= 1; end                                 # nearest stored correct env
        for j in s:(ind - 1)
            above = isnothing(ca.above) ? state.AL[j] : ca.above.AL[j]
            ca.GLs[j + 1] = ca.GLs[j] * TransferMatrix(above, ca.operator[j], state.AL[j]; backend, allocator)
            ca.ldependencies[j] = state.AL[j]
        end
    end
    k = ca.k; lo = ind - mod(ind - 1, k)                                         # keep [lo, ind] and 1 + multiples of k
    for j in 2:length(ca.GLs)
        # behind the segment: keep checkpoints only; ahead of ind: keep checkpoints only (rebuilt on demand)
        ((j < lo || j > ind) && mod(j - 1, k) != 0) && (ca.GLs[j] = nothing)
    end
    return ca.GLs[ind]
end

# GRs[i+1] contracts sites i+1..N; rightenv(ind) returns GRs[ind+1]; GRs[N+1] is the boundary (always kept).
function rightenv(ca::CheckpointedFiniteEnvironments, ind, state;
                  backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator())
    N = length(state)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), N:-1:(ind + 1))
    a = isnothing(a) ? nothing : N - a + 1                                       # last (rightmost) changed site
    valid_from = isnothing(a) ? ind + 1 : a + 1                                  # GRs[valid_from..N+1] correct if stored
    if ca.GRs[ind + 1] === nothing || valid_from > ind + 1
        s = valid_from
        while ca.GRs[s] === nothing; s += 1; end
        for j in (s - 1):-1:(ind + 1)
            above = isnothing(ca.above) ? state.AR[j] : ca.above.AR[j]
            ca.GRs[j] = TransferMatrix(above, ca.operator[j], state.AR[j]; backend, allocator) * ca.GRs[j + 1]
            ca.rdependencies[j] = state.AR[j]
        end
    end
    k = ca.k; hi = ind + 1 + mod(N + 1 - (ind + 1), k)                           # keep [ind+1, hi) and N+1 - multiples of k
    for j in 1:N
        ((j > hi || j < ind + 1) && mod(N + 1 - j, k) != 0) && (ca.GRs[j] = nothing)
    end
    return ca.GRs[ind + 1]
end

end # module
