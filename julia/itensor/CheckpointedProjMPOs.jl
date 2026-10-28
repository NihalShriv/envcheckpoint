"""
    CheckpointedProjMPO(H::MPO; k = round(Int, sqrt(length(H))), k2 = 0, after_drop = () -> GC.gc(false),
                        offload = nothing)

Drop-in replacement for ITensorMPS `ProjMPO` that keeps only every `k`-th environment block (plus the segment
currently in use) and recomputes the dropped ones from the nearest kept block when a sweep reaches them.
Exact: the rebuilt blocks are the same contractions as in `ProjMPO`. Memory ~ (N/k + 2k) blocks instead of N,
for ~1 extra environment update per site and sweep. Works wherever an `AbstractProjMPO` is accepted:

    dmrg(CheckpointedProjMPO(H), psi0; nsweeps, maxdim, ...)
    tdvp(CheckpointedProjMPO(H), t, psi0; ...)      # reduced_operator(PH) == PH

Options:
- `k2 > 0`: two-level schedule - a rebuilt segment keeps only every `k2`-th block (memory ~ N/k + k/k2 + 2 k2,
  ~2 extra environment updates per site).
- `after_drop`: called after blocks are dropped (GPU: `() -> (GC.gc(false); CUDA.reclaim())`).
- `offload = (store, load)`: checkpoint blocks away from the current position are moved out of device memory with
  `store(::ITensor) -> handle` and brought back with `load(handle) -> ITensor` (e.g. to host RAM or disk), so the
  device holds only the active segments.
"""
module CheckpointedProjMPOs

using ITensors, ITensorMPS
import ITensorMPS: AbstractProjMPO, lproj, rproj, position!, set_nsite!, makeL!, makeR!, nsite, site_range
import ITensors: contract
import Base: copy

export CheckpointedProjMPO

struct Stashed; handle::Any; end

mutable struct CheckpointedProjMPO <: AbstractProjMPO
    lpos::Int
    rpos::Int
    nsite::Int
    H::MPO
    LR::Vector{Any}        # nothing | ITensor | Stashed
    k::Int
    k2::Int
    nrebuilt::Int          # environment updates spent on rebuilding dropped blocks (diagnostics)
    after_drop::Function
    offload::Union{Nothing, Tuple{Function, Function}}
    chunks::Int            # > 1: apply H_eff in chunks of the outer MPO links (dense tensors only)
    todev::Function        # moves a CPU ITensor to the device of the environments (for the chunk projectors)
    pre_solve::Function    # called after every position! (i.e. right before each local solve)
end

function CheckpointedProjMPO(H::MPO; k::Int = max(1, round(Int, sqrt(length(H)))), k2::Int = 0,
                             after_drop::Function = () -> GC.gc(false), offload = nothing, chunks::Int = 1,
                             todev::Function = identity, pre_solve::Function = () -> nothing)
    return CheckpointedProjMPO(0, length(H) + 1, 2, H, Vector{Any}(nothing, length(H)), k, k2, 0, after_drop,
                               offload, chunks, todev, pre_solve)
end

copy(P::CheckpointedProjMPO) = CheckpointedProjMPO(P.lpos, P.rpos, P.nsite, copy(P.H), copy(P.LR), P.k, P.k2,
                                                   P.nrebuilt, P.after_drop, P.offload, P.chunks, P.todev, P.pre_solve)
set_nsite!(P::CheckpointedProjMPO, n) = (P.nsite = n; P)

has(P, j) = 1 <= j <= length(P.LR) && P.LR[j] !== nothing

# Block j in device memory (loads an offloaded checkpoint back).
function block(P::CheckpointedProjMPO, j::Int)::ITensor
    b = P.LR[j]
    b isa Stashed && (b = P.offload[2](b.handle); P.LR[j] = b)
    return b
end

lproj(P::CheckpointedProjMPO) = P.lpos <= 0 ? ITensors.OneITensor() : block(P, P.lpos)
rproj(P::CheckpointedProjMPO) = P.rpos >= length(P.H) + 1 ? ITensors.OneITensor() : block(P, P.rpos)

# Rebuild L block l (left environment of sites 1..l) from the nearest kept block below it.
function rebuild_L!(P::CheckpointedProjMPO, psi::MPS, l::Int)
    c = l - 1
    while c >= 1 && !has(P, c); c -= 1; end
    L = c >= 1 ? block(P, c) : ITensors.OneITensor()
    for j in (c + 1):l
        L = L * psi[j] * P.H[j] * dag(prime(psi[j]))
        P.LR[j] = L
        P.nrebuilt += 1
        # two-level: inside the rebuilt segment keep only every k2-th block (the gaps are rebuilt from those)
        P.k2 > 0 && j - 1 > c && mod(j - 1, P.k2) != 0 && mod(j - 1, P.k) != 0 && (P.LR[j - 1] = nothing)
    end
    return P
end

# Rebuild R block r (right environment of sites r..N) from the nearest kept block above it.
function rebuild_R!(P::CheckpointedProjMPO, psi::MPS, r::Int)
    N = length(P.H); c = r + 1
    while c <= N && !has(P, c); c += 1; end
    R = c <= N ? block(P, c) : ITensors.OneITensor()
    for j in (c - 1):-1:r
        R = R * psi[j] * P.H[j] * dag(prime(psi[j]))
        P.LR[j] = R
        P.nrebuilt += 1
        P.k2 > 0 && j + 1 < c && mod(j + 1, P.k2) != 0 && mod(j + 1, P.k) != 0 && (P.LR[j + 1] = nothing)
    end
    return P
end

# Keep: checkpoints (multiples of k), the current lpos/rpos, and the open segments next to them (they are the
# blocks the sweep will use in the next steps; dropping them would rebuild every step). With `offload`, the
# checkpoints outside the open segments leave device memory.
function prune!(P::CheckpointedProjMPO)
    N = length(P.H); k = P.k; dropped = false
    lo = P.lpos - mod(P.lpos, k)                      # L blocks in (lo, lpos] stay
    hi = P.rpos + (mod(P.rpos, k) == 0 ? 0 : k - mod(P.rpos, k))   # R blocks in [rpos, hi) stay
    for j in 1:N
        P.LR[j] === nothing && continue
        active = (lo < j <= P.lpos) || (P.rpos <= j < hi) || j == lo || j == hi
        keep = j <= P.lpos ? (mod(j, k) == 0 || j > lo) : j >= P.rpos ? (mod(j, k) == 0 || j < hi) : false
        if !keep
            P.LR[j] = nothing; dropped = true
        elseif P.offload !== nothing && !active && P.LR[j] isa ITensor
            P.LR[j] = Stashed(P.offload[1](P.LR[j])); dropped = true
        end
    end
    # GPU arrays are only returned to the device pool when collected; without this the dropped blocks stay
    # allocated until the next out-of-memory retry. Young-generation GC is cheap next to a two-site update.
    dropped && P.after_drop()
    return P
end

function position!(P::CheckpointedProjMPO, psi::MPS, pos::Int)
    l, r = pos - 1, pos + nsite(P)
    # Moving the left projection back onto a dropped block, or the right projection forward onto one:
    # rebuild it (and the segment to it) from the nearest kept block. Forward-built blocks use ProjMPO's code.
    (1 <= l < P.lpos && !has(P, l)) && rebuild_L!(P, psi, l)
    (P.rpos < r <= length(P.H) && !has(P, r)) && rebuild_R!(P, psi, r)
    makeL!(P, psi, l)
    makeR!(P, psi, r)
    prune!(P)
    P.pre_solve()
    return P
end

# 0/1 matrix selecting entries r of index i into a new index of length(r)
function chunk_projector(i::Index, r::UnitRange{Int})
    ic = Index(length(r), "chunk")
    M = zeros(dim(i), length(r)); for (a, j) in enumerate(r); M[j, a] = 1.0; end
    return ITensor(M, i, ic)
end

# H_eff * v with the two outer MPO links (L-H1 and H2-R) summed in chunks: only one full intermediate (after H1)
# instead of two or three live at once. Same result up to summation order. Dense, two-site, bulk only; otherwise
# the standard ITensorMPS contraction.
function contract(P::CheckpointedProjMPO, v::ITensor)::ITensor
    L, R = lproj(P), rproj(P)
    if P.chunks <= 1 || nsite(P) != 2 || !(L isa ITensor) || !(R isa ITensor) || hasqns(L)
        return invoke(contract, Tuple{AbstractProjMPO, ITensor}, P, v)
    end
    H1, H2 = P.H[first(site_range(P))], P.H[last(site_range(P))]
    x, z = commonind(L, H1), commonind(H2, R)
    T2 = nothing
    for r in Iterators.partition(1:dim(x), cld(dim(x), P.chunks))
        Pc = P.todev(chunk_projector(x, first(r):last(r)))
        T2c = ((L * Pc) * v) * (Pc * H1)
        T2 === nothing ? (T2 = T2c) : (T2 .+= T2c)
    end
    Hv = nothing
    for r in Iterators.partition(1:dim(z), cld(dim(z), P.chunks))
        Qc = P.todev(chunk_projector(z, first(r):last(r)))
        Hvc = (T2 * (H2 * Qc)) * (Qc * R)
        Hv === nothing ? (Hv = Hvc) : (Hv .+= Hvc)
    end
    return Hv
end

end # module
