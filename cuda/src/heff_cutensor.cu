#include "heff_cutensor.h"
#include <vector>
#include <map>
#include <algorithm>

#define CUTENSOR_CHECK(expr) do { cutensorStatus_t s_ = (expr); if (s_ != CUTENSOR_STATUS_SUCCESS) \
    throw std::runtime_error(std::string("cuTENSOR error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                             " for " #expr ": " + cutensorGetErrorString(s_)); } while (0)

// Mode lists are fastest-varying first (cuTENSOR default with NULL strides).
// Existing layouts: theta row-major [a,p,q,c] -> {c,q,p,a}; L env_idx -> {a,b,x}; R -> {c,d2,z};
// W mpo_idx(wl,wr,p,q) -> {wl,wr,p,q}.
namespace {
void *sT1 = nullptr, *sT2 = nullptr, *sT3 = nullptr, *sW = nullptr;
size_t nT1 = 0, nT2 = 0, nT3 = 0, nW = 0;
void grow(void*& p, size_t& n, size_t need) {
    if (need <= n) return;
    CUDA_CHECK(cudaDeviceSynchronize()); cudaFree(p); p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, need)); n = need;
}
int g_chunks = 1;
inline int chunk_len(int n, int c) { return (n + c - 1) / c; }
enum : int32_t { A='a', B='b', C='c', D2='d', P='p', Q='q', Rr='r', S='s', X='x', Y='y', Z='z' };
}

void heff_set_chunks(int c) { g_chunks = std::max(1, c); }

HeffCutensor::CPlan HeffCutensor::make(const std::vector<int32_t>& mA, const std::vector<int64_t>& eA,
                                       const std::vector<int64_t>& sA,
                                       const std::vector<int32_t>& mB, const std::vector<int64_t>& eB,
                                       const std::vector<int64_t>& sB,
                                       const std::vector<int32_t>& mD, const std::vector<int64_t>& eD)
{
    // slices start at arbitrary element offsets -> 8-byte alignment
    auto mk = [&](const std::vector<int64_t>& e, const std::vector<int64_t>& st) {
        cutensorTensorDescriptor_t t;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle_, &t, (uint32_t)e.size(), e.data(),
                                                      st.empty() ? nullptr : st.data(), CUTENSOR_R_64F, 8));
        cdescs_.push_back(t); return t;
    };
    cutensorTensorDescriptor_t a = mk(eA, sA), b = mk(eB, sB), dd = mk(eD, {});
    CPlan p;
    CUTENSOR_CHECK(cutensorCreateContraction(handle_, &p.op, a, mA.data(), CUTENSOR_OP_IDENTITY, b, mB.data(),
        CUTENSOR_OP_IDENTITY, dd, mD.data(), CUTENSOR_OP_IDENTITY, dd, mD.data(), CUTENSOR_COMPUTE_DESC_64F));
    cutensorPlanPreference_t pref;
    CUTENSOR_CHECK(cutensorCreatePlanPreference(handle_, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(handle_, p.op, pref, CUTENSOR_WORKSPACE_DEFAULT, &p.ws));
    CUTENSOR_CHECK(cutensorCreatePlan(handle_, &p.plan, p.op, pref, p.ws));
    CUTENSOR_CHECK(cutensorDestroyPlanPreference(pref));
    work_size_ = std::max(work_size_, p.ws);
    return p;
}

HeffCutensor::HeffCutensor(int Dl, int d, int Dr, int Wl, int Wm, int Wr)
  : chunks_(std::min(g_chunks, std::min(Wl, Wr))), Dl_(Dl), d_(d), Dr_(Dr), Wl_(Wl), Wm_(Wm), Wr_(Wr)
{
    std::map<int32_t, int64_t> ext = { {A,Dl},{B,Dl},{C,Dr},{D2,Dr},{P,d},{Q,d},{Rr,d},{S,d},{X,Wl},{Y,Wm},{Z,Wr} };
    const std::vector<std::vector<int32_t>> modes = {
        {A, B, X},              // 0 L
        {D2, S, Rr, B},         // 1 theta_in (row-major [b,r,s,d2])
        {A, X, Rr, S, D2},      // 2 T1
        {X, Y, P, Rr},          // 3 W1
        {A, Y, P, S, D2},       // 4 T2
        {Y, Z, Q, S},           // 5 W2
        {A, P, Z, Q, D2},       // 6 T3
        {C, D2, Z},             // 7 R
        {C, Q, P, A},           // 8 out (row-major [a,p,q,c])
    };
    CUTENSOR_CHECK(cutensorCreate(&handle_));
    for (int i = 0; i < 9; ++i) {
        std::vector<int64_t> e;
        for (int32_t m : modes[i]) e.push_back(ext[m]);
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle_, &desc_[i], (uint32_t)e.size(), e.data(),
                                                      nullptr, CUTENSOR_R_64F, 128));
    }
    // (A operand, B operand, C/D output)
    const int ops[4][3] = { {0,1,2}, {2,3,4}, {4,5,6}, {6,7,8} };
    cutensorPlanPreference_t pref;
    CUTENSOR_CHECK(cutensorCreatePlanPreference(handle_, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    for (int k = 0; k < 4; ++k) {
        int a = ops[k][0], b = ops[k][1], c = ops[k][2];
        CUTENSOR_CHECK(cutensorCreateContraction(handle_, &op_[k],
            desc_[a], modes[a].data(), CUTENSOR_OP_IDENTITY,
            desc_[b], modes[b].data(), CUTENSOR_OP_IDENTITY,
            desc_[c], modes[c].data(), CUTENSOR_OP_IDENTITY,
            desc_[c], modes[c].data(), CUTENSOR_COMPUTE_DESC_64F));
        uint64_t ws = 0;
        CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(handle_, op_[k], pref, CUTENSOR_WORKSPACE_DEFAULT, &ws));
        CUTENSOR_CHECK(cutensorCreatePlan(handle_, &plan_[k], op_[k], pref, ws));
        work_size_ = std::max(work_size_, ws);

        // flops = 2 * prod(extent of all distinct modes in the contraction)
        std::vector<int32_t> all = modes[a];
        for (int32_t m : modes[b]) if (std::find(all.begin(), all.end(), m) == all.end()) all.push_back(m);
        double f = 2.0;
        for (int32_t m : all) f *= ext[m];
        flops_ += f;
    }
    CUTENSOR_CHECK(cutensorDestroyPlanPreference(pref));

    // envck tree: intermediates and workspace are shared by all cached plans (grown on demand) instead of one set
    // per plan; apply() runs one plan at a time on the default stream, so sharing is safe.
    if (chunks_ > 1) {
        // x chunks for L*theta and T1c*W1 (T2 accumulated), z chunks for T2*W2 and T3c*R (out accumulated)
        const int64_t xc = chunk_len(Wl, chunks_), zc = chunk_len(Wr, chunks_);
        const int64_t xl = Wl - (Wl - 1) / xc * xc, zl = Wr - (Wr - 1) / zc * zc;   // last chunk lengths
        const int64_t dl = Dl, dr = Dr, dd = d, wl = Wl, wm = Wm, wr = Wr;
        for (int last = 0; last < 2; ++last) {
            const int64_t x = last ? xl : xc, z = last ? zl : zc;
            cplans_.push_back(make({A,B,X}, {dl,dl,x}, {}, {D2,S,Rr,B}, {dr,dd,dd,dl}, {},
                                   {A,X,Rr,S,D2}, {dl,x,dd,dd,dr}));
            cplans_.push_back(make({A,X,Rr,S,D2}, {dl,x,dd,dd,dr}, {}, {X,Y,P,Rr}, {x,wm,dd,dd},
                                   {1,wl,wl*wm,wl*wm*dd}, {A,Y,P,S,D2}, {dl,wm,dd,dd,dr}));
            cplans_.push_back(make({A,Y,P,S,D2}, {dl,wm,dd,dd,dr}, {}, {Y,Z,Q,S}, {wm,z,dd,dd},
                                   {1,wm,wm*wr,wm*wr*dd}, {A,P,Z,Q,D2}, {dl,dd,z,dd,dr}));
            cplans_.push_back(make({A,P,Z,Q,D2}, {dl,dd,z,dd,dr}, {}, {C,D2,Z}, {dr,dr,z}, {},
                                   {C,Q,P,A}, {dr,dd,dd,dl}));
        }
        size_t t1 = (size_t)Dl * xc * d * d * Dr, t2 = (size_t)Dl * Wm * d * d * Dr, t3 = (size_t)Dl * d * zc * d * Dr;
        grow(sT1, nT1, t1 * sizeof(double)); grow(sT2, nT2, t2 * sizeof(double)); grow(sT3, nT3, t3 * sizeof(double));
        grow(sW, nW, work_size_);
        return;
    }
    size_t t1 = (size_t)Dl * Wl * d * d * Dr, t2 = (size_t)Dl * Wm * d * d * Dr, t3 = (size_t)Dl * d * Wr * d * Dr;
    grow(sT1, nT1, t1 * sizeof(double)); grow(sT2, nT2, t2 * sizeof(double)); grow(sT3, nT3, t3 * sizeof(double));
    grow(sW, nW, work_size_);
}

HeffCutensor::~HeffCutensor()
{
    for (auto p : plan_) if (p) cutensorDestroyPlan(p);
    for (auto o : op_) if (o) cutensorDestroyOperationDescriptor(o);
    for (auto d : desc_) if (d) cutensorDestroyTensorDescriptor(d);
    for (auto& c : cplans_) { if (c.plan) cutensorDestroyPlan(c.plan); if (c.op) cutensorDestroyOperationDescriptor(c.op); }
    for (auto d : cdescs_) cutensorDestroyTensorDescriptor(d);
    if (handle_) cutensorDestroy(handle_);
}

void HeffCutensor::apply(Tensor& out, const Tensor& theta_in, const Tensor& L,
                         const Tensor& W1, const Tensor& W2, const Tensor& R)
{
    if (chunks_ > 1) { apply_chunked(out, theta_in, L, W1, W2, R); return; }
    double* T1_ = (double*)sT1; double* T2_ = (double*)sT2; double* T3_ = (double*)sT3; void* work_ = sW;
    const double one = 1.0, zero = 0.0;
    CUTENSOR_CHECK(cutensorContract(handle_, plan_[0], &one, L.d_data,  theta_in.d_data, &zero, T1_, T1_, work_, work_size_, 0));
    CUTENSOR_CHECK(cutensorContract(handle_, plan_[1], &one, T1_,       W1.d_data,       &zero, T2_, T2_, work_, work_size_, 0));
    CUTENSOR_CHECK(cutensorContract(handle_, plan_[2], &one, T2_,       W2.d_data,       &zero, T3_, T3_, work_, work_size_, 0));
    CUTENSOR_CHECK(cutensorContract(handle_, plan_[3], &one, T3_,       R.d_data,        &zero, out.d_data, out.d_data, work_, work_size_, 0));
}

void HeffCutensor::apply_chunked(Tensor& out, const Tensor& theta_in, const Tensor& L,
                                 const Tensor& W1, const Tensor& W2, const Tensor& R)
{
    double* T1c = (double*)sT1; double* T2 = (double*)sT2; double* T3c = (double*)sT3; void* work = sW;
    const double one = 1.0, zero = 0.0;
    const int xc = chunk_len(Wl_, chunks_), zc = chunk_len(Wr_, chunks_);
    for (int x0 = 0; x0 < Wl_; x0 += xc) {
        const CPlan* p = &cplans_[x0 + xc > Wl_ ? 4 : 0];          // shorter last chunk
        CUTENSOR_CHECK(cutensorContract(handle_, p[0].plan, &one, L.d_data + (size_t)x0 * Dl_ * Dl_, theta_in.d_data,
                                        &zero, T1c, T1c, work, nW, 0));
        CUTENSOR_CHECK(cutensorContract(handle_, p[1].plan, &one, T1c, W1.d_data + x0, x0 ? &one : &zero, T2, T2,
                                        work, nW, 0));
    }
    for (int z0 = 0; z0 < Wr_; z0 += zc) {
        const CPlan* p = &cplans_[z0 + zc > Wr_ ? 4 : 0];
        CUTENSOR_CHECK(cutensorContract(handle_, p[2].plan, &one, T2, W2.d_data + (size_t)z0 * Wm_, &zero, T3c, T3c,
                                        work, nW, 0));
        CUTENSOR_CHECK(cutensorContract(handle_, p[3].plan, &one, T3c, R.d_data + (size_t)z0 * Dr_ * Dr_,
                                        z0 ? &one : &zero, out.d_data, out.d_data, work, nW, 0));
    }
}
