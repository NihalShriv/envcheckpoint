// two_site_split_timed.cu
// -----------------------------------------------------------------------
// Timed two-site DMRG update — profile-focused rewrite.
//
// Changes vs. the original two_site_split.cu:
//   1. SweepWorkspace pre-allocates Davidson buffers at max size once per
//      (N, D) configuration — no Tensor malloc/free inside the sweep loop
//      for workspace arrays (basis, image, projected, temporaries).
//   2. Chi selection is on CPU (one D→H cudaMemcpy of singular values;
//      no cudaMallocManaged + select_bond_dim_kernel + DeviceSynchronize).
//   3. SVD / syevd reuse LinAlg cached workspace (prewarm_svd / prewarm_syevd).
//   4. Timing is explicit per category in TwoSiteUpdateStats; no remainder.
//   5. newA1 / newA2 still allocate per site (psi takes ownership) — these
//      are the only in-loop allocs and are measured via t_alloc_ms.
//   6. energy filled from the Davidson eigenvalue.
// -----------------------------------------------------------------------

#include "two_site_split_timed.h"
#include "ct_contract.h"
#include "heff_builder.h"
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// -----------------------------------------------------------------------
// GPU kernels — identical to two_site_split.cu (no logic changed)
// -----------------------------------------------------------------------
namespace {

__global__ void merge_two_site_theta(double* theta, const double* A1, const double* A2,
                                      int Dl, int d, int Dm, int Dr)
{
    int dim = Dl * d * d * Dr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dim) return;
    int a = idx / (d * d * Dr);
    int tmp = idx % (d * d * Dr);
    int p = tmp / (d * Dr);
    tmp %= (d * Dr);
    int q = tmp / Dr;
    int c = tmp % Dr;
    double sum = 0.0;
    for(int m = 0; m < Dm; ++m)
        sum += A1[a + Dl * (p + d * m)] * A2[m + Dm * (q + d * c)];
    theta[idx] = sum;
}

__global__ void ground_to_theta(double* theta, const double* ground, int Dl, int d, int Dr)
{
    int dim = Dl * d * d * Dr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dim) return;
    int a = idx / (d * d * Dr);
    int tmp = idx % (d * d * Dr);
    int p = tmp / (d * Dr);
    tmp %= (d * Dr);
    int q = tmp / Dr;
    int c = tmp % Dr;
    int left_dim = Dl * d;
    int row = a * d + p;
    int col = q * Dr + c;
    theta[row + left_dim * col] = ground[idx];
}

__global__ void build_left_tensor(double* A1, const double* U, int Dl, int d, int chi, int left_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Dl * d * chi) return;
    int a = idx % Dl;
    int tmp = idx / Dl;
    int p = tmp % d;
    int k = tmp / d;
    int row = a * d + p;
    A1[idx] = U[row + left_dim * k];
}

__global__ void build_left_tensor_with_s(double* A1, const double* U, const double* S,
                                          int Dl, int d, int chi, int left_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Dl * d * chi) return;
    int a = idx % Dl;
    int tmp = idx / Dl;
    int p = tmp % d;
    int k = tmp / d;
    int row = a * d + p;
    A1[idx] = U[row + left_dim * k] * S[k];
}

__global__ void build_right_tensor(double* A2, const double* S, const double* V,
                                    int chi, int d, int Dr, int right_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= chi * d * Dr) return;
    int k = idx % chi;
    int tmp = idx / chi;
    int q = tmp % d;
    int c = tmp / d;
    int col = q * Dr + c;
    A2[idx] = S[k] * V[col + right_dim * k];
}

// dst[:, j] = src[:, g-1-j] for j < chi (syevd returns eigenvalues ascending; we keep the largest first)
__global__ void copy_cols_reversed(double* dst, const double* src, int rows, int g, int chi)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= (size_t)rows * chi) return;
    int r = idx % rows, j = idx / rows;
    dst[idx] = src[r + (size_t)rows * (g - 1 - j)];
}

// Inverses of build_left_tensor / build_right_tensor_plain: old MPS tensor -> column-major seed factor
__global__ void seed_left_factor(double* U, const double* A1, int Dl, int d, int chi, int left_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Dl * d * chi) return;
    int a = idx % Dl, tmp = idx / Dl, p = tmp % d, k = tmp / d;
    U[a * d + p + (size_t)left_dim * k] = A1[idx];
}
__global__ void seed_right_factor(double* V, const double* A2, int chi, int d, int Dr, int right_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= chi * d * Dr) return;
    int k = idx % chi, tmp = idx / chi, q = tmp % d, c = tmp / d;
    V[q * Dr + c + (size_t)right_dim * k] = A2[idx];
}

__global__ void build_right_tensor_plain(double* A2, const double* V,
                                          int chi, int d, int Dr, int right_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= chi * d * Dr) return;
    int k = idx % chi;
    int tmp = idx / chi;
    int q = tmp % d;
    int c = tmp / d;
    int col = q * Dr + c;
    A2[idx] = V[col + right_dim * k];
}

__global__ void copy_vector_to_column(double* matrix, const double* vec, int ld, int col, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    matrix[idx + ld * col] = vec[idx];
}

__global__ void copy_column_to_vector(double* vec, const double* matrix, int ld, int col, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    vec[idx] = matrix[idx + ld * col];
}

__global__ void write_symmetric_column(double* projected, const double* overlap,
                                        int stride, int col, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count) return;
    double value = overlap[idx];
    projected[idx + stride * col] = value;
    projected[col + stride * idx] = value;
}

__global__ void copy_leading_block(double* packed, const double* full, int packed_dim, int full_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= packed_dim * packed_dim) return;
    int row = idx % packed_dim;
    int col = idx / packed_dim;
    packed[row + packed_dim * col] = full[row + full_dim * col];
}

__global__ void build_effective_diag(
    double* diag,
    const double* L,
    const double* R,
    const double* W1,
    const double* W2,
    int Dl, int d, int Dr,
    int W1l, int W1r, int W2r,
    int Ld0, int Ld1, int Rd0, int Rd1,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dim) return;
    int a = idx / (d * d * Dr);
    int tmp = idx % (d * d * Dr);
    int p = tmp / (d * Dr);
    tmp %= (d * Dr);
    int q = tmp / Dr;
    int c = tmp % Dr;
    double value = 0.0;
    for(int x = 0; x < W1l; ++x)
    for(int y = 0; y < W1r; ++y)
    for(int z = 0; z < W2r; ++z)
    {
        double Lval  = L[layout::env_idx(a, a, x, Ld0, Ld1)];
        double Rval  = R[layout::env_idx(c, c, z, Rd0, Rd1)];
        double W1val = W1[layout::mpo_idx(x, y, p, p, W1l, W1r, d)];
        double W2val = W2[layout::mpo_idx(y, z, q, q, W1r, W2r, d)];
        value += Lval * W1val * W2val * Rval;
    }
    diag[idx] = value;
}

__global__ void apply_diagonal_preconditioner(double* out, const double* residual,
                                               const double* diag, double eigenvalue,
                                               double eps, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    double denom = diag[idx] - eigenvalue;
    if(fabs(denom) < eps) denom = (denom >= 0.0) ? eps : -eps;
    out[idx] = residual[idx] / denom;
}

// -----------------------------------------------------------------------
// cuBLAS wrappers (same signature as in two_site_split.cu)
// -----------------------------------------------------------------------
void gemm_tn_device(cublasHandle_t blas, const double* A, const double* B, double* C,
                     int m, int k, int n, int lda, int ldb, int ldc)
{
    double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                              m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

void gemm_nn_device(cublasHandle_t blas, const double* A, const double* B, double* C,
                     int m, int k, int n, int lda, int ldb, int ldc)
{
    double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

// -----------------------------------------------------------------------
// Non-owning Tensor view — wraps a pre-allocated device pointer so existing
// Tensor-based APIs can be called without any allocation.
// Caller must call release() before the struct is destroyed to prevent
// the Tensor destructor from freeing workspace memory.
// -----------------------------------------------------------------------
struct TensorView {
    Tensor t;           // default-constructed: d_data = nullptr
    void init(double* ptr, int n) {
        t.d_data = ptr; t.size = n; t.shape = {n};
    }
    void init2d(double* ptr, int rows, int cols) {
        t.d_data = ptr; t.size = rows * cols; t.shape = {rows, cols};
    }
    void release() { t.d_data = nullptr; }
    ~TensorView()  { t.d_data = nullptr; } // safety net
};

// Wall-clock delta in milliseconds (CPU-side sections)
static inline float wall_ms(struct timespec& a, struct timespec& b) {
    return (float)((b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) * 1e-6);
}

} // anonymous namespace

// -----------------------------------------------------------------------
// SweepWorkspace — constructor allocates all buffers at max size
// -----------------------------------------------------------------------
void SweepWorkspace::alloc(double*& ptr, size_t n) {
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(double)));
}

SweepWorkspace::SweepWorkspace(int D, int d, cusolverDnHandle_t solver, int cap)
{
    subspace_cap = std::max(4, std::min(cap, (int)MAX_SUBSPACE));
    max_left_dim  = D * d;
    max_right_dim = d * D;
    max_rank      = std::min(max_left_dim, max_right_dim);
    max_dim       = max_left_dim * max_right_dim;  // D*d*d*D worst case

    alloc(projected,    (size_t)MAX_SUBSPACE * MAX_SUBSPACE);
    // envck tree: only subspace_cap columns (was MAX_SUBSPACE = 160: 2 x 160 x D^2 d^2 doubles, 43 GB at D=2048)
    alloc(basis_mat,    (size_t)max_dim      * subspace_cap);
    alloc(image_mat,    (size_t)max_dim      * subspace_cap);
    alloc(diagonal,     (size_t)max_dim);
    alloc(seed_v,       (size_t)max_dim);
    alloc(current_v,    (size_t)max_dim);
    alloc(basis_v,      (size_t)max_dim);
    alloc(image_v,      (size_t)max_dim);
    alloc(overlap_v,    (size_t)MAX_SUBSPACE);
    alloc(ground_v,     (size_t)max_dim);
    alloc(hground_v,    (size_t)max_dim);
    alloc(residual_v,   (size_t)max_dim);
    alloc(correction_v, (size_t)max_dim);
    alloc(projection_v, (size_t)max_dim);
    alloc(packed_v,     (size_t)MAX_SUBSPACE * MAX_SUBSPACE);
    alloc(eigvec_v,     (size_t)MAX_SUBSPACE);
    alloc(eig_vals,     (size_t)MAX_SUBSPACE);
    alloc(ground,       (size_t)max_dim);
    alloc(Theta,        (size_t)max_left_dim * max_right_dim);

    // SVD output buffers — pre-allocated at max size so no per-site cudaMalloc/cudaFree
    alloc(U_ws,  (size_t)max_left_dim  * max_left_dim);
    alloc(S_ws,  (size_t)max_rank);
    alloc(Vt_ws, (size_t)max_right_dim * max_right_dim);

    // cuBLAS Heff scratch buffers
    alloc(heff_ML,  (size_t)max_left_dim  * max_left_dim);
    alloc(heff_MR,  (size_t)max_right_dim * max_right_dim);
    alloc(heff_tmp, (size_t)max_right_dim * max_left_dim);

    CUDA_CHECK(cudaMalloc(&d_info_ws, sizeof(int)));

    // Pre-size syevd workspace for MAX_SUBSPACE x MAX_SUBSPACE subspace problems
    {
        double *dA, *dW;
        CUDA_CHECK(cudaMalloc(&dA, (size_t)MAX_SUBSPACE * MAX_SUBSPACE * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dW, (size_t)MAX_SUBSPACE * sizeof(double)));
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
            solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, MAX_SUBSPACE,
            dA, MAX_SUBSPACE, dW, &eig_lwork));
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dW));
        alloc(eig_work, (size_t)eig_lwork);
    }

    // Truncation workspaces (Gesvd / Gram methods), sized for the largest two-site matrix
    {
        int g = std::max(max_left_dim, max_right_dim);
        alloc(gram_ws,   (size_t)g * g);
        alloc(gram_eval, (size_t)g);
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                                   g, gram_ws, g, gram_eval, &gram_lwork));
        alloc(gram_work, (size_t)gram_lwork);
        CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(solver, g, g, &gesvd_lwork));
        alloc(gesvd_work,  (size_t)gesvd_lwork);
        alloc(gesvd_rwork, (size_t)g);
        // QR (geqrf + orgqr) of a g x g block, for GramRefine
        int lq1 = 0, lq2 = 0;
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver, g, g, gram_ws, g, &lq1));
        alloc(qr_tau, (size_t)g);
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(solver, g, g, g, gram_ws, g, qr_tau, &lq2));
        qr_lwork = std::max(lq1, lq2);
        alloc(qr_work, (size_t)qr_lwork);
    }

    sv_host.resize(max_rank);
}

#define CURAND_CHECK(x) do { if((x) != CURAND_STATUS_SUCCESS) throw std::runtime_error("cuRAND error at " __FILE__); } while(0)
// Feasibility proxy for reduced-precision H_eff (heff_noise > 0): out += eps * ||out|| / sqrt(n) * N(0,1) noise,
// i.e. a random relative error of size eps on every H_eff application (TF32 ~ 1e-3, FP32 ~ 1e-6).
static void add_heff_noise(SweepWorkspace& wk, LinAlg& la, double* out, int n)
{
    if(wk.heff_noise <= 0.0) return;
    if(!wk.noise_gen) {
        CURAND_CHECK(curandCreateGenerator(&wk.noise_gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(wk.noise_gen, 12345ULL));
    }
    const size_t m = (size_t)n + (n & 1);                   // curandGenerateNormal needs an even count
    if(wk.noise_n < m) {
        if(wk.noise_buf) cudaFree(wk.noise_buf);
        CUDA_CHECK(cudaMalloc(&wk.noise_buf, m * sizeof(double))); wk.noise_n = m;
    }
    CURAND_CHECK(curandGenerateNormalDouble(wk.noise_gen, wk.noise_buf, m, 0.0, 1.0));
    double nrm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(la.blas, n, out, 1, &nrm));
    const double a = wk.heff_noise * nrm / std::sqrt((double)n);
    CUBLAS_CHECK(cublasDaxpy(la.blas, n, &a, wk.noise_buf, 1, out, 1));
}

void sweep_apply_heff(SweepWorkspace& wk, double* out, const double* in, const MPS& psi, const MPO& H,
                      const Environment& env, LinAlg& la, int site)
{
    const Tensor& W1 = H.W[site];
    const Tensor& W2 = H.W[site + 1];
    int Dl = psi.A[site].shape[0], d = psi.d, Dr = psi.A[site + 1].shape[2];
    int dim = Dl * d * d * Dr;
    TensorView o, i;
    o.init(out, dim); i.init(const_cast<double*>(in), dim);
    if(wk.heff_backend == HeffBackend::Gemm)
        apply_two_site_heff_gemm(o.t, i.t, psi, H, env, site, la.blas, wk.heff_ML, wk.heff_MR, wk.heff_tmp);
    else if(wk.heff_backend == HeffBackend::Cutensor && wk.heff_lowp)
        wk.lowp_for(Dl, d, Dr, W1.shape[0], W1.shape[1], W2.shape[1])
          .apply(out, in, env.L[site], W1, W2, env.R[site + 2]);
    else if(wk.heff_backend == HeffBackend::Cutensor)
        wk.cutensor_for(Dl, d, Dr, W1.shape[0], W1.shape[1], W2.shape[1])
          .apply(o.t, i.t, env.L[site], W1, W2, env.R[site + 2]);
    else
        wk.struct_for(Dl, d, Dr, W1.shape[0], W2.shape[1])
          .apply(o.t, i.t, env.L[site], W1, W2, env.R[site + 2], la.blas);
    add_heff_noise(wk, la, out, dim);
    o.release(); i.release();
}

void sweep_merge_theta(double* theta, const MPS& psi, int site)
{
    const Tensor& A1 = psi.A[site];
    const Tensor& A2 = psi.A[site + 1];
    int Dl = A1.shape[0], d = psi.d, Dm = A1.shape[2], Dr = A2.shape[2];
    // theta[a,p,q,c] (row-major, i.e. modes {c,q,p,a} fastest first) = sum_m A1{a,p,m} A2{m,q,c}: one O(D^3)
    // cuTENSOR contraction (the Davidson path still uses the naive merge_two_site_theta kernel, unchanged)
    ct_contract({'a','p','m'}, {Dl,d,Dm}, A1.d_data, {'m','q','c'}, {Dm,d,Dr}, A2.d_data,
                {'c','q','p','a'}, {Dr,d,d,Dl}, theta);
}

HeffCutensor& SweepWorkspace::cutensor_for(int Dl, int d, int Dr, int Wl, int Wm, int Wr)
{
    std::array<int,5> key{Dl, Dr, Wl, Wm, Wr};
    auto it = cutensor_plans.find(key);
    if(it != cutensor_plans.end()) return *it->second;
    // ponytail: drop the whole cache when it grows; shapes repeat once the bond dimension saturates
    if(cutensor_plans.size() >= 8) cutensor_plans.clear();
    return *(cutensor_plans[key] = std::make_unique<HeffCutensor>(Dl, d, Dr, Wl, Wm, Wr));
}

HeffLowp& SweepWorkspace::lowp_for(int Dl, int d, int Dr, int Wl, int Wm, int Wr)
{
    std::array<int,6> key{Dl, Dr, Wl, Wm, Wr, heff_lowp};
    auto it = lowp_plans.find(key);
    if(it != lowp_plans.end()) return *it->second;
    if(lowp_plans.size() >= 8) lowp_plans.clear();   // ponytail: same drop-all cache as cutensor_for
    return *(lowp_plans[key] = std::make_unique<HeffLowp>(Dl, d, Dr, Wl, Wm, Wr, heff_lowp == 1));
}

HeffStruct& SweepWorkspace::struct_for(int Dl, int d, int Dr, int Wl, int Wr)
{
    std::array<int,4> key{Dl, Dr, Wl, Wr};
    auto it = struct_plans.find(key);
    if(it != struct_plans.end()) return *it->second;
    if(struct_plans.size() >= 8) struct_plans.clear();
    return *(struct_plans[key] = std::make_unique<HeffStruct>(Dl, d, Dr, Wl, Wr));
}

SweepWorkspace::~SweepWorkspace()
{
    auto f = [](double*& p){ if(p){ cudaFree(p); p = nullptr; } };
    f(projected); f(basis_mat); f(image_mat); f(diagonal);
    f(seed_v); f(current_v);
    f(basis_v); f(image_v); f(overlap_v); f(ground_v); f(hground_v);
    f(residual_v); f(correction_v); f(projection_v);
    f(packed_v); f(eigvec_v); f(eig_vals); f(eig_work);
    f(ground); f(Theta);
    f(U_ws); f(S_ws); f(Vt_ws);
    f(heff_ML); f(heff_MR); f(heff_tmp);
    f(gram_ws); f(gram_eval); f(gram_work); f(gesvd_work); f(gesvd_rwork); f(qr_tau); f(qr_work);
    if(rsvd_work) { cudaFree(rsvd_work); rsvd_work = nullptr; }
    if(d_info_ws){ cudaFree(d_info_ws); d_info_ws = nullptr; }
}

// -----------------------------------------------------------------------
// Davidson solver — mirrors solve_ground_state_davidson() from
// two_site_split.cu exactly, except:
//   a) All inner-loop Tensor allocations replaced with workspace views.
//   b) CUDA events time each apply_two_site_heff call.
//   c) Subspace syevd uses wk.packed_v/eig_vals/eig_work directly
//      (same workspace query done in the SweepWorkspace constructor).
// -----------------------------------------------------------------------
namespace {

// diag(H_eff)[a,p,q,c] = sum_{x,y,z} L[a,a,x] W1[x,y,p,p] W2[y,z,q,q] R[c,c,z], factorised as
//   W12[x,z,p,q] = sum_y W1[x,y,p,p] W2[y,z,q,q]   (tiny, cached per MPO tensor pair)
//   T[a,z,p,q] = sum_x Ld[a,x] W12[x,z,p,q] ;  diag[a,p,q,c] = sum_z T[a,z,p,q] Rd[c,z]
// instead of the naive per-element triple loop (build_effective_diag, ~30% of GPU time at D=1024).
__global__ void gather_env_diag(double* out, const double* E, int D, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= D * W) return;
    int a = idx % D, x = idx / D;
    out[idx] = E[a + (size_t)D * (a + (size_t)D * x)];   // env_idx(a, a, x)
}

void fast_effective_diag(double* diag, const Tensor& L, const Tensor& R, const Tensor& W1, const Tensor& W2,
                         int Dl, int d, int Dr)
{
    const int Wl = W1.shape[0], Wm = W1.shape[1], Wr = W2.shape[1];
    static std::map<std::pair<const double*, const double*>, double*> w12_cache;
    auto key = std::make_pair((const double*)W1.d_data, (const double*)W2.d_data);
    double*& w12 = w12_cache[key];
    if(!w12) {
        auto w1 = W1.to_host(), w2 = W2.to_host();
        std::vector<double> h((size_t)Wl * Wr * d * d, 0.0);           // modes {x,z,p,q}
        for(int x = 0; x < Wl; ++x) for(int z = 0; z < Wr; ++z) for(int p = 0; p < d; ++p) for(int q = 0; q < d; ++q) {
            double s = 0;
            for(int y = 0; y < Wm; ++y)
                s += w1[layout::mpo_idx(x, y, p, p, Wl, Wm, d)] * w2[layout::mpo_idx(y, z, q, q, Wm, Wr, d)];
            h[x + Wl * (z + Wr * (p + d * q))] = s;
        }
        CUDA_CHECK(cudaMalloc(&w12, h.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(w12, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    double* Ld = ct_scratch(3, (size_t)Dl * Wl);
    double* Rd = ct_scratch(4, (size_t)Dr * Wr);
    double* T  = ct_scratch(5, (size_t)Dl * Wr * d * d);
    const int threads = 256;
    CUDA_KERNEL_CHECK((gather_env_diag<<<(Dl * Wl + threads - 1) / threads, threads>>>(Ld, L.d_data, Dl, Wl)));
    CUDA_KERNEL_CHECK((gather_env_diag<<<(Dr * Wr + threads - 1) / threads, threads>>>(Rd, R.d_data, Dr, Wr)));
    ct_contract({'a','x'}, {Dl,Wl}, Ld, {'x','z','p','q'}, {Wl,Wr,d,d}, w12, {'a','z','p','q'}, {Dl,Wr,d,d}, T);
    ct_contract({'a','z','p','q'}, {Dl,Wr,d,d}, T, {'c','z'}, {Dr,Wr}, Rd, {'c','q','p','a'}, {Dr,d,d,Dl}, diag);
}

bool davidson_timed(
    SweepWorkspace& wk,
    const MPS& psi,
    const MPO& H,
    const Environment& env,
    LinAlg& la,
    int site,
    double truncation_target,
    double& residual_out,
    double& eigenvalue_out,
    float& heff_ms_accum,
    int& iter_count,
    bool& converged_out,
    double& tol_out)
{
    const Tensor& A1 = psi.A[site];
    const Tensor& A2 = psi.A[site + 1];
    const Tensor& W1 = H.W[site];
    const Tensor& W2 = H.W[site + 1];
    const Tensor& L  = env.L[site];
    const Tensor& R  = env.R[site + 2];

    int Dl  = A1.shape[0];
    int d   = psi.d;
    int Dm  = A1.shape[2];
    int Dr  = A2.shape[2];
    int dim = Dl * d * d * Dr;
    int threads = 256;
    int blocks  = (dim + threads - 1) / threads;
    int column_blocks = blocks;
    int max_subspace_cap = wk.subspace_cap;

    bool success = false;

    // Non-owning views into workspace memory — released at end, no deallocation
    TensorView seed_t, current_t, diagonal_t;
    TensorView basis_vec_t, image_vec_t;
    TensorView ground_vec_t, hground_vec_t, residual_t, correction_t;
    seed_t.init     (wk.seed_v,    dim);
    current_t.init  (wk.current_v, dim);
    diagonal_t.init (wk.diagonal,  dim);
    basis_vec_t.init(wk.basis_v,   dim);
    image_vec_t.init(wk.image_v,   dim);
    ground_vec_t.init (wk.ground_v,   dim);
    hground_vec_t.init(wk.hground_v,  dim);
    residual_t.init   (wk.residual_v,   dim);
    correction_t.init (wk.correction_v, dim);

    cudaEvent_t ev_a = nullptr, ev_b = nullptr;
    if(!wk.production_mode) {
        cudaEventCreate(&ev_a); cudaEventCreate(&ev_b);
    }

    // Seed = merged two-site tensor
    if(wk.davidson_fast_kernels)
        sweep_merge_theta(wk.seed_v, psi, site);          // one O(D^3) cuTENSOR contraction
    else
        CUDA_KERNEL_CHECK((merge_two_site_theta<<<blocks, threads>>>(
            wk.seed_v, A1.d_data, A2.d_data, Dl, d, Dm, Dr)));
    double seed_norm = la.nrm2(seed_t.t);
    if(seed_norm < 1e-12 || std::isnan(seed_norm)) goto done;
    la.scal(seed_t.t, 1.0 / seed_norm);

    // Diagonal preconditioner
    if(wk.davidson_fast_kernels)
        fast_effective_diag(wk.diagonal, L, R, W1, W2, Dl, d, Dr);   // factorised, O(D^2 W d^2)
    else
        CUDA_KERNEL_CHECK((build_effective_diag<<<blocks, threads>>>(
            wk.diagonal, L.d_data, R.d_data, W1.d_data, W2.d_data,
            Dl, d, Dr, W1.shape[0], W1.shape[1], W2.shape[1],
            L.shape[0], L.shape[1], R.shape[0], R.shape[1], dim)));

    {
        int current_subspace = std::min(max_subspace_cap,
                                std::max(48, std::min(96, dim / 128 + 32)));
        double residual_tol  = (truncation_target > 0.0)
                                 ? std::max(1e-13, truncation_target)
                                 : 1e-12;  // fixed-D mode: still require tight convergence
        if(wk.davidson_tol_override > 0.0) residual_tol = wk.davidson_tol_override;
        double last_residual = std::numeric_limits<double>::infinity();
        double ground_energy = 0.0;
        tol_out = residual_tol;

        current_t.t.copy_from(seed_t.t);

        for(int restart = 0; restart < 8; ++restart)
        {
            CUDA_CHECK(cudaMemset(wk.projected, 0,
                (size_t)max_subspace_cap * max_subspace_cap * sizeof(double)));

            double cnorm = la.nrm2(current_t.t);
            if(cnorm < 1e-12 || std::isnan(cnorm)) goto done;
            la.scal(current_t.t, 1.0 / cnorm);

            CUDA_KERNEL_CHECK((copy_vector_to_column<<<column_blocks, threads>>>(
                wk.basis_mat, wk.current_v, dim, 0, dim)));

            bool converged  = false;
            int  basis_count = 1;

            for(int iter = 0; iter < current_subspace; ++iter)
            {
                int new_idx = basis_count - 1;

                CUDA_KERNEL_CHECK((copy_column_to_vector<<<column_blocks, threads>>>(
                    wk.basis_v, wk.basis_mat, dim, new_idx, dim)));

                // ---- H_eff apply — CUDA-event timed (instrumented mode only) ----
                if(!wk.production_mode) cudaEventRecord(ev_a);
                if(wk.heff_backend == HeffBackend::Gemm)
                    apply_two_site_heff_gemm(image_vec_t.t, basis_vec_t.t, psi, H, env, site,
                                             la.blas, wk.heff_ML, wk.heff_MR, wk.heff_tmp);
                else if(wk.heff_backend == HeffBackend::Cutensor && wk.heff_lowp)
                    wk.lowp_for(Dl, d, Dr, W1.shape[0], W1.shape[1], W2.shape[1])
                      .apply(image_vec_t.t.d_data, basis_vec_t.t.d_data, L, W1, W2, R);
                else if(wk.heff_backend == HeffBackend::Cutensor)
                    wk.cutensor_for(Dl, d, Dr, W1.shape[0], W1.shape[1], W2.shape[1])
                      .apply(image_vec_t.t, basis_vec_t.t, L, W1, W2, R);
                else
                    wk.struct_for(Dl, d, Dr, W1.shape[0], W2.shape[1])
                      .apply(image_vec_t.t, basis_vec_t.t, L, W1, W2, R, la.blas);
                add_heff_noise(wk, la, image_vec_t.t.d_data, dim);
                if(!wk.production_mode) {
                    cudaEventRecord(ev_b);
                    cudaEventSynchronize(ev_b);  // removed in production mode
                    float ms = 0.f;
                    cudaEventElapsedTime(&ms, ev_a, ev_b);
                    heff_ms_accum += ms;
                }
                ++iter_count;
                // ---------------------------------------------------------------

                CUDA_KERNEL_CHECK((copy_vector_to_column<<<column_blocks, threads>>>(
                    wk.image_mat, wk.image_v, dim, new_idx, dim)));

                // overlap (basis_count) = basis_mat^T * image_v
                gemm_tn_device(la.blas, wk.basis_mat, wk.image_v, wk.overlap_v,
                                basis_count, dim, 1, dim, dim, basis_count);

                int ov_blocks = (basis_count + threads - 1) / threads;
                CUDA_KERNEL_CHECK((write_symmetric_column<<<ov_blocks, threads>>>(
                    wk.projected, wk.overlap_v, max_subspace_cap, new_idx, basis_count)));

                // Subspace eigenproblem using workspace packed_v / eig_vals / eig_work
                {
                    int pb_blocks = (basis_count * basis_count + threads - 1) / threads;
                    CUDA_KERNEL_CHECK((copy_leading_block<<<pb_blocks, threads>>>(
                        wk.packed_v, wk.projected, basis_count, max_subspace_cap)));
                    CUSOLVER_CHECK(cusolverDnDsyevd(
                        la.solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                        basis_count, wk.packed_v, basis_count, wk.eig_vals,
                        wk.eig_work, wk.eig_lwork, wk.d_info_ws));
                    CUDA_CHECK(cudaMemcpy(&ground_energy, wk.eig_vals,
                        sizeof(double), cudaMemcpyDeviceToHost));
                    int cv_blocks = (basis_count + threads - 1) / threads;
                    CUDA_KERNEL_CHECK((copy_column_to_vector<<<cv_blocks, threads>>>(
                        wk.eigvec_v, wk.packed_v, basis_count, 0, basis_count)));
                }

                // ground_vec = basis_mat * eigvec, hground_vec = image_mat * eigvec
                gemm_nn_device(la.blas, wk.basis_mat, wk.eigvec_v, wk.ground_v,
                                dim, basis_count, 1, dim, basis_count, dim);
                gemm_nn_device(la.blas, wk.image_mat, wk.eigvec_v, wk.hground_v,
                                dim, basis_count, 1, dim, basis_count, dim);

                double gnorm = la.nrm2(ground_vec_t.t);
                if(gnorm < 1e-12 || std::isnan(gnorm)) goto done;
                la.scal(ground_vec_t.t, 1.0 / gnorm);
                la.scal(hground_vec_t.t, 1.0 / gnorm);

                // residual = hground - ground_energy * ground
                residual_t.t.copy_from(hground_vec_t.t);
                la.axpy(-ground_energy, ground_vec_t.t, residual_t.t);
                last_residual = la.nrm2(residual_t.t);
                if(last_residual > residual_tol && wk.davidson_max_matvec > 0
                   && iter_count >= wk.davidson_max_matvec) {
                    // ritz_T > 0 (experimental): instead of the lowest Ritz vector take v0 + a v1 (normalised),
                    // a = random sign * U(0,1) * exp(-(l1 - l0)/T), i.e. a thermal-like kick along the lowest local
                    // excitation; the reported energy is the exact Rayleigh quotient (l0 + a^2 l1) / (1 + a^2).
                    if(wk.ritz_T > 0.0 && basis_count >= 2) {
                        const int bc = basis_count;
                        std::vector<double> V((size_t)bc * bc), lam(2), c(bc);
                        CUDA_CHECK(cudaMemcpy(V.data(), wk.packed_v, sizeof(double) * bc * bc, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(lam.data(), wk.eig_vals, sizeof(double) * 2, cudaMemcpyDeviceToHost));
                        std::uniform_real_distribution<double> U(0.0, 1.0);
                        const double a = (U(wk.ritz_rng) < 0.5 ? -1.0 : 1.0) * U(wk.ritz_rng)
                                       * std::exp(-(lam[1] - lam[0]) / wk.ritz_T);
                        const double inv = 1.0 / std::sqrt(1.0 + a * a);
                        for(int i = 0; i < bc; ++i) c[i] = (V[i] + a * V[i + bc]) * inv;
                        CUDA_CHECK(cudaMemcpy(wk.eigvec_v, c.data(), sizeof(double) * bc, cudaMemcpyHostToDevice));
                        gemm_nn_device(la.blas, wk.basis_mat, wk.eigvec_v, wk.ground_v,
                                        dim, basis_count, 1, dim, basis_count, dim);
                        ground_energy = (lam[0] + a * a * lam[1]) / (1.0 + a * a);
                    }
                    goto budget_exhausted;   // ground_v holds the (possibly kicked) Ritz vector
                }

                if(last_residual <= residual_tol) { converged = true; break; }

                if(basis_count >= current_subspace || iter + 1 >= current_subspace) {
                    current_t.t.copy_from(ground_vec_t.t);
                    break;
                }

                // Preconditioned correction + orthonormalization
                CUDA_KERNEL_CHECK((apply_diagonal_preconditioner<<<blocks, threads>>>(
                    wk.correction_v, wk.residual_v, wk.diagonal, ground_energy, 1e-8, dim)));

                // Two-pass Gram-Schmidt (reuses overlap_v and projection_v as scratch)
                for(int pass = 0; pass < 2; ++pass) {
                    gemm_tn_device(la.blas, wk.basis_mat, wk.correction_v, wk.overlap_v,
                                    basis_count, dim, 1, dim, dim, basis_count);
                    gemm_nn_device(la.blas, wk.basis_mat, wk.overlap_v, wk.projection_v,
                                    dim, basis_count, 1, dim, basis_count, dim);
                    double neg = -1.0;
                    CUBLAS_CHECK(cublasDaxpy(la.blas, dim, &neg,
                                             wk.projection_v, 1, wk.correction_v, 1));
                }
                double corr_norm = la.nrm2(correction_t.t);
                if(corr_norm < 1e-12 || std::isnan(corr_norm)) {
                    current_t.t.copy_from(ground_vec_t.t);
                    break;
                }
                la.scal(correction_t.t, 1.0 / corr_norm);

                CUDA_KERNEL_CHECK((copy_vector_to_column<<<column_blocks, threads>>>(
                    wk.basis_mat, wk.correction_v, dim, basis_count, dim)));
                ++basis_count;
                current_t.t.copy_from(ground_vec_t.t);
            }

            if(converged || last_residual <= residual_tol) {
                CUDA_CHECK(cudaMemcpy(wk.ground, wk.ground_v,
                    dim * sizeof(double), cudaMemcpyDeviceToDevice));
                residual_out   = last_residual;
                eigenvalue_out = ground_energy;
                success = true;
                converged_out = true;
                goto done;
            }

            if(last_residual > 10.0 * residual_tol && current_subspace < max_subspace_cap)
                current_subspace = std::min(max_subspace_cap, current_subspace + 16);

            current_t.t.copy_from(ground_vec_t.t);
        }

        // Max restarts (or matvec budget) exhausted — return best result
    budget_exhausted:
        CUDA_CHECK(cudaMemcpy(wk.ground, wk.ground_v,
            dim * sizeof(double), cudaMemcpyDeviceToDevice));
        residual_out   = last_residual;
        eigenvalue_out = ground_energy;
        success = std::isfinite(last_residual);
    }

done:
    seed_t.release(); current_t.release(); diagonal_t.release();
    basis_vec_t.release(); image_vec_t.release();
    ground_vec_t.release(); hground_vec_t.release();
    residual_t.release(); correction_t.release();
    if(!wk.production_mode) { cudaEventDestroy(ev_a); cudaEventDestroy(ev_b); }
    return success;
}

} // anonymous namespace

// -----------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------
TwoSiteUpdateStats two_site_update_timed(
    MPS& psi,
    const MPO& H,
    Environment& env,
    LinAlg& la,
    SweepWorkspace& wk,
    int site,
    SweepDirection direction)
{
    TwoSiteUpdateStats stats;

    const Tensor& A1 = psi.A[site];
    const Tensor& A2 = psi.A[site + 1];
    int Dl        = A1.shape[0];
    int d         = psi.d;
    int Dr        = A2.shape[2];
    int dim       = Dl * d * d * Dr;
    int left_dim  = Dl * d;
    int right_dim = d * Dr;
    int rank      = std::min(left_dim, right_dim);
    if(rank <= 0) return stats;

    const double truncation_target = wk.svd_threshold;
    int threads = 256;
    struct timespec cpu0, cpu1;

    // ---- Davidson — outer event captures FULL solver time (instrumented mode) ----
    {
        cudaEvent_t dav0 = nullptr, dav1 = nullptr;
        if(!wk.production_mode) { cudaEventCreate(&dav0); cudaEventCreate(&dav1); }
        float heff_ms = 0.f;
        if(!wk.production_mode) cudaEventRecord(dav0);
        bool ok = wk.local_solver
            ? wk.local_solver(wk, psi, H, env, la, site, stats.residual_norm, stats.energy,
                              stats.davidson_iters, stats.davidson_converged, stats.davidson_tol)
            : davidson_timed(wk, psi, H, env, la, site, truncation_target,
                             stats.residual_norm, stats.energy,
                             heff_ms, stats.davidson_iters,
                             stats.davidson_converged, stats.davidson_tol);
        if(!wk.production_mode) {
            cudaEventRecord(dav1);
            cudaEventSynchronize(dav1);
            cudaEventElapsedTime(&stats.t_davidson_ms, dav0, dav1);
            stats.t_heff_ms = heff_ms;
            cudaEventDestroy(dav0); cudaEventDestroy(dav1);
        }
        if(!ok) return stats;
    }
    // optional post-processing of the local ground state before truncation (experimental, e.g. DIIS)
    if(wk.post_solve) wk.post_solve(wk, psi, H, env, la, site, stats.energy);

    // ---- Reshape ground state into Theta -----------
    {
        int blocks = (dim + threads - 1) / threads;
        if(!wk.production_mode) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            CUDA_KERNEL_CHECK((ground_to_theta<<<blocks, threads>>>(wk.Theta, wk.ground, Dl, d, Dr)));
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&stats.t_reshape_ms, e0, e1);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        } else {
            CUDA_KERNEL_CHECK((ground_to_theta<<<blocks, threads>>>(wk.Theta, wk.ground, Dl, d, Dr)));
        }
    }

    // Normalize Theta (matching original two_site_split.cu)
    {
        TensorView theta_t;
        theta_t.init2d(wk.Theta, left_dim, right_dim);
        double theta_norm = la.nrm2(theta_t.t);
        theta_t.release();
        if(theta_norm < 1e-12 || std::isnan(theta_norm)) return stats;
        double inv = 1.0 / theta_norm;
        CUBLAS_CHECK(cublasDscal(la.blas, left_dim * right_dim, &inv, wk.Theta, 1));
    }

    // ---- SVD — U/S/Vt use pre-allocated workspace (no per-site cudaMalloc) ---
    // Non-owning views into workspace; release() prevents destructor from freeing.
    TensorView U_tv, S_tv, Vt_tv;
    U_tv.init2d (wk.U_ws,  left_dim,  left_dim);
    S_tv.init   (wk.S_ws,  rank);
    Vt_tv.init2d(wk.Vt_ws, right_dim, right_dim);

    // Gram route: which side is orthonormal (L->R: U = top eigvecs of Theta Theta^T; R->L: V of Theta^T Theta)
    const bool gram   = wk.svd_method == SweepWorkspace::SvdMethod::Gram
                     || wk.svd_method == SweepWorkspace::SvdMethod::GramRefine
                     || wk.svd_method == SweepWorkspace::SvdMethod::Warm;
    const bool refine = wk.svd_method == SweepWorkspace::SvdMethod::GramRefine;
    // Warm: seed from the current tensor on the orthonormal side when its bond dim equals the new chi
    const int  old_chi = direction == SweepDirection::LeftToRight ? psi.A[site].shape[2] : psi.A[site + 1].shape[0];
    const bool warm = wk.svd_method == SweepWorkspace::SvdMethod::Warm && wk.svd_threshold <= 0.0
                   && old_chi == std::min(psi.D, rank);
    if(wk.svd_method == SweepWorkspace::SvdMethod::Warm) (warm ? wk.warm_hits : wk.warm_fallbacks)++;
    // randomized SVD only pays off when chi + oversampling fits well inside the rank (else: gesvd)
    const int  rsvd_k = std::min(psi.D, rank);
    const int  rsvd_p = std::max(8, rsvd_k / 8);
    const bool rsvd   = wk.svd_method == SweepWorkspace::SvdMethod::Rsvd && rsvd_k + rsvd_p <= rank;
    const bool exact_disc = gram || rsvd;   // discarded weight from 1 - kept weight
    const bool gram_u = direction == SweepDirection::LeftToRight;
    const int  gdim   = gram_u ? left_dim : right_dim;
    {
        TensorView theta_t;
        theta_t.init2d(wk.Theta, left_dim, right_dim);
        auto decompose = [&]() {
            const double one = 1.0, zero = 0.0;
            if(wk.svd_method == SweepWorkspace::SvdMethod::Jacobi) {
                la.svd(theta_t.t, U_tv.t, S_tv.t, Vt_tv.t, left_dim, right_dim);
            } else if(rsvd) {
                cusolverDnParams_t pr; CUSOLVER_CHECK(cusolverDnCreateParams(&pr));
                size_t dws = 0, hws = 0;
                CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(la.solver, pr, 'S', 'S', left_dim, right_dim, rsvd_k,
                    rsvd_p, 2, CUDA_R_64F, wk.Theta, left_dim, CUDA_R_64F, wk.S_ws, CUDA_R_64F, wk.U_ws, left_dim,
                    CUDA_R_64F, wk.Vt_ws, right_dim, CUDA_R_64F, &dws, &hws));
                if(dws > wk.rsvd_work_bytes) {
                    if(wk.rsvd_work) cudaFree(wk.rsvd_work);
                    CUDA_CHECK(cudaMalloc(&wk.rsvd_work, dws)); wk.rsvd_work_bytes = dws;
                }
                if(hws > wk.rsvd_host_work.size()) wk.rsvd_host_work.resize(hws);
                CUSOLVER_CHECK(cusolverDnXgesvdr(la.solver, pr, 'S', 'S', left_dim, right_dim, rsvd_k, rsvd_p, 2,
                    CUDA_R_64F, wk.Theta, left_dim, CUDA_R_64F, wk.S_ws, CUDA_R_64F, wk.U_ws, left_dim,
                    CUDA_R_64F, wk.Vt_ws, right_dim, CUDA_R_64F, wk.rsvd_work, dws, wk.rsvd_host_work.data(), hws,
                    wk.d_info_ws));
                CUSOLVER_CHECK(cusolverDnDestroyParams(pr));
            } else if(wk.svd_method == SweepWorkspace::SvdMethod::Gesvd
                      || wk.svd_method == SweepWorkspace::SvdMethod::Rsvd) {
                // gesvd needs m >= n: factor Theta (m >= n) or Theta^T (m < n); outputs U (m x r), V (n x r)
                // Workspace queried per shape: cuSOLVER's gesvd lwork is not monotone in (m, n), so the size for
                // the largest square case can be too small for a small non-square one (seen: CUSOLVER code 7).
                {
                    int lw = 0;
                    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(la.solver, std::max(left_dim, right_dim),
                                                               std::min(left_dim, right_dim), &lw));
                    if(lw > wk.gesvd_lwork) {
                        CUDA_CHECK(cudaFree(wk.gesvd_work));
                        CUDA_CHECK(cudaMalloc(&wk.gesvd_work, (size_t)lw * sizeof(double)));
                        wk.gesvd_lwork = lw;
                    }
                }
                if(left_dim >= right_dim) {
                    CUSOLVER_CHECK(cusolverDnDgesvd(la.solver, 'S', 'S', left_dim, right_dim, wk.Theta, left_dim,
                        wk.S_ws, wk.U_ws, left_dim, wk.gram_ws, right_dim,
                        wk.gesvd_work, wk.gesvd_lwork, wk.gesvd_rwork, wk.d_info_ws));
                    // gram_ws holds V^T (r x n, ld n since r == n): V = (V^T)^T
                    CUBLAS_CHECK(cublasDgeam(la.blas, CUBLAS_OP_T, CUBLAS_OP_N, right_dim, rank, &one,
                        wk.gram_ws, right_dim, &zero, wk.Vt_ws, right_dim, wk.Vt_ws, right_dim));
                } else {
                    // Theta^T (n x m) = U' S VT'  =>  V = U' (n x r), U = VT'^T (m x r)
                    CUBLAS_CHECK(cublasDgeam(la.blas, CUBLAS_OP_T, CUBLAS_OP_N, right_dim, left_dim, &one,
                        wk.Theta, left_dim, &zero, wk.gram_ws, right_dim, wk.gram_ws, right_dim));
                    CUSOLVER_CHECK(cusolverDnDgesvd(la.solver, 'S', 'S', right_dim, left_dim, wk.gram_ws, right_dim,
                        wk.S_ws, wk.Vt_ws, right_dim, wk.U_ws, left_dim,
                        wk.gesvd_work, wk.gesvd_lwork, wk.gesvd_rwork, wk.d_info_ws));
                    // U_ws now holds VT' (r x m, ld m since r == m): transpose in place is unsafe -> via gram_ws
                    CUBLAS_CHECK(cublasDgeam(la.blas, CUBLAS_OP_T, CUBLAS_OP_N, left_dim, rank, &one,
                        wk.U_ws, left_dim, &zero, wk.gram_ws, left_dim, wk.gram_ws, left_dim));
                    CUDA_CHECK(cudaMemcpyAsync(wk.U_ws, wk.gram_ws, sizeof(double) * left_dim * rank,
                        cudaMemcpyDeviceToDevice));
                }
            } else if(!warm) {
                // rho = Theta Theta^T (L->R) or Theta^T Theta (R->L), symmetric gdim x gdim
                CUBLAS_CHECK(cublasDsyrk(la.blas, CUBLAS_FILL_MODE_LOWER, gram_u ? CUBLAS_OP_N : CUBLAS_OP_T,
                    gdim, gram_u ? right_dim : left_dim, &one, wk.Theta, left_dim, &zero, wk.gram_ws, gdim));
                CUSOLVER_CHECK(cusolverDnDsyevd(la.solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                    gdim, wk.gram_ws, gdim, wk.gram_eval, wk.gram_work, wk.gram_lwork, wk.d_info_ws));
            }
        };
        if(!wk.production_mode) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            decompose();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&stats.t_svd_ms, e0, e1);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        } else {
            decompose();
        }
        theta_t.release();
    }

    // ---- Chi selection on CPU (single D→H cudaMemcpy, no GPU kernel) ---
    {
        clock_gettime(CLOCK_MONOTONIC, &cpu0);
        if(!gram) {
            const int nsv = rsvd ? rsvd_k : rank;   // gesvdr returns only the chi kept values
            CUDA_CHECK(cudaMemcpy(wk.sv_host.data(), wk.S_ws,
                nsv * sizeof(double), cudaMemcpyDeviceToHost));
            std::fill(wk.sv_host.begin() + nsv, wk.sv_host.begin() + rank, 0.0);
        } else if(warm) {
            std::fill(wk.sv_host.begin(), wk.sv_host.begin() + rank, 0.0);   // fixed-D: chi = min(D, rank)
        } else {
            // eigenvalues ascending: sigma_i = sqrt(lambda_{gdim-1-i}), i < rank (descending)
            std::vector<double> lam(gdim);
            CUDA_CHECK(cudaMemcpy(lam.data(), wk.gram_eval, gdim * sizeof(double), cudaMemcpyDeviceToHost));
            for(int i = 0; i < rank; ++i) wk.sv_host[i] = std::sqrt(std::max(0.0, lam[gdim - 1 - i]));
        }
        clock_gettime(CLOCK_MONOTONIC, &cpu1);
        stats.t_sync_ms += wall_ms(cpu0, cpu1);

        clock_gettime(CLOCK_MONOTONIC, &cpu0);
        int D   = psi.D;
        int chi = std::min(D, rank);
        if(truncation_target > 0.0) {
            double tail = 0.0;
            for(int k = rank - 1; k >= 0; --k) {
                tail += wk.sv_host[k] * wk.sv_host[k];
                if(k < D && tail > truncation_target) { chi = k + 1; break; }
            }
            chi = std::max(1, chi);
        }
        // fixed-D mode (truncation_target <= 0): chi already = min(D, rank)
        double discarded = 0.0;
        for(int k = chi; k < rank; ++k)
            discarded += wk.sv_host[k] * wk.sv_host[k];
        clock_gettime(CLOCK_MONOTONIC, &cpu1);
        stats.t_cpu_ms += wall_ms(cpu0, cpu1);

        stats.kept_chi         = chi;
        stats.discarded_weight = discarded;
    }

    // ---- Build new MPS tensors (newA1/newA2 must alloc — psi takes ownership) ---
    clock_gettime(CLOCK_MONOTONIC, &cpu0);
    Tensor newA1({Dl, d, stats.kept_chi});
    Tensor newA2({stats.kept_chi, d, Dr});
    clock_gettime(CLOCK_MONOTONIC, &cpu1);
    stats.t_alloc_ms += wall_ms(cpu0, cpu1);

    {
        int chi         = stats.kept_chi;
        int total_left  = Dl * d * chi;
        int total_right = chi * d * Dr;
        int bl = (total_left  + threads - 1) / threads;
        int br = (total_right + threads - 1) / threads;

        auto do_build = [&]() {
            if(gram) {
                const double one = 1.0, zero = 0.0;
                int cb = (int)(((size_t)gdim * chi + threads - 1) / threads);
                // GramRefine: one subspace-iteration step X <- qr(rho X) on the kept orthonormal factor X
                auto refine_step = [&](double* X, int xrows, double* T, int trows, bool x_is_u) {
                    // T = Theta^T X (x_is_u) or Theta X ; Y = Theta T or Theta^T T  (Y = rho X) into gram_ws
                    CUBLAS_CHECK(cublasDgemm(la.blas, x_is_u ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, trows, chi, xrows,
                        &one, wk.Theta, left_dim, X, xrows, &zero, T, trows));
                    CUBLAS_CHECK(cublasDgemm(la.blas, x_is_u ? CUBLAS_OP_N : CUBLAS_OP_T, CUBLAS_OP_N, xrows, chi, trows,
                        &one, wk.Theta, left_dim, T, trows, &zero, wk.gram_ws, xrows));
                    if(warm && wk.warm_eta > 0.0) {
                        // flow truncation (experimental): Y = X + eta * rho X (X orthonormal, tr rho = 1). Directions with
                        // weight*eta >> 1 rotate to the new optimum, weak ones stay continuous with the previous basis.
                        const double one_ = 1.0;
                        CUBLAS_CHECK(cublasDscal(la.blas, xrows * chi, &wk.warm_eta, wk.gram_ws, 1));
                        CUBLAS_CHECK(cublasDaxpy(la.blas, xrows * chi, &one_, X, 1, wk.gram_ws, 1));
                    }
                    CUSOLVER_CHECK(cusolverDnDgeqrf(la.solver, xrows, chi, wk.gram_ws, xrows, wk.qr_tau,
                        wk.qr_work, wk.qr_lwork, wk.d_info_ws));
                    CUSOLVER_CHECK(cusolverDnDorgqr(la.solver, xrows, chi, chi, wk.gram_ws, xrows, wk.qr_tau,
                        wk.qr_work, wk.qr_lwork, wk.d_info_ws));
                    CUDA_CHECK(cudaMemcpyAsync(X, wk.gram_ws, sizeof(double) * xrows * chi, cudaMemcpyDeviceToDevice));
                };
                // warm_iters = 0: re-orthonormalise the seed only (pure basis reuse, no rho applied)
                auto qr_only = [&](double* X, int xrows) {
                    CUDA_CHECK(cudaMemcpyAsync(wk.gram_ws, X, sizeof(double) * xrows * chi, cudaMemcpyDeviceToDevice));
                    CUSOLVER_CHECK(cusolverDnDgeqrf(la.solver, xrows, chi, wk.gram_ws, xrows, wk.qr_tau,
                        wk.qr_work, wk.qr_lwork, wk.d_info_ws));
                    CUSOLVER_CHECK(cusolverDnDorgqr(la.solver, xrows, chi, chi, wk.gram_ws, xrows, wk.qr_tau,
                        wk.qr_work, wk.qr_lwork, wk.d_info_ws));
                    CUDA_CHECK(cudaMemcpyAsync(X, wk.gram_ws, sizeof(double) * xrows * chi, cudaMemcpyDeviceToDevice));
                };
                if(gram_u) {
                    // U = top-chi eigvecs (orthonormal); W = Theta^T U = V S (n x chi) -> A2 = W^T
                    if(warm) {
                        CUDA_KERNEL_CHECK((seed_left_factor<<<bl, threads>>>(
                            wk.U_ws, psi.A[site].d_data, Dl, d, chi, left_dim)));
                        if(wk.warm_iters == 0 || wk.warm_eta > 0.0 || wk.warm_orth) qr_only(wk.U_ws, left_dim);   // flow needs orthonormal X
                        for(int it = 0; it < wk.warm_iters; ++it) refine_step(wk.U_ws, left_dim, wk.Vt_ws, right_dim, true);
                    } else
                    CUDA_KERNEL_CHECK((copy_cols_reversed<<<cb, threads>>>(wk.U_ws, wk.gram_ws, gdim, gdim, chi)));
                    if(refine) refine_step(wk.U_ws, left_dim, wk.Vt_ws, right_dim, true);
                    CUBLAS_CHECK(cublasDgemm(la.blas, CUBLAS_OP_T, CUBLAS_OP_N, right_dim, chi, left_dim, &one,
                        wk.Theta, left_dim, wk.U_ws, left_dim, &zero, wk.Vt_ws, right_dim));
                } else {
                    // V = top-chi eigvecs (orthonormal); P = Theta V = U S (m x chi) -> A1 = P
                    if(warm) {
                        CUDA_KERNEL_CHECK((seed_right_factor<<<br, threads>>>(
                            wk.Vt_ws, psi.A[site + 1].d_data, chi, d, Dr, right_dim)));
                        if(wk.warm_iters == 0 || wk.warm_eta > 0.0 || wk.warm_orth) qr_only(wk.Vt_ws, right_dim);
                        for(int it = 0; it < wk.warm_iters; ++it) refine_step(wk.Vt_ws, right_dim, wk.U_ws, left_dim, false);
                    } else
                    CUDA_KERNEL_CHECK((copy_cols_reversed<<<cb, threads>>>(wk.Vt_ws, wk.gram_ws, gdim, gdim, chi)));
                    if(refine) refine_step(wk.Vt_ws, right_dim, wk.U_ws, left_dim, false);
                    CUBLAS_CHECK(cublasDgemm(la.blas, CUBLAS_OP_N, CUBLAS_OP_N, left_dim, chi, right_dim, &one,
                        wk.Theta, left_dim, wk.Vt_ws, right_dim, &zero, wk.U_ws, left_dim));
                }
                CUDA_KERNEL_CHECK((build_left_tensor<<<bl, threads>>>(
                    newA1.d_data, wk.U_ws, Dl, d, chi, left_dim)));
                CUDA_KERNEL_CHECK((build_right_tensor_plain<<<br, threads>>>(
                    newA2.d_data, wk.Vt_ws, chi, d, Dr, right_dim)));
                return;
            }
            if(direction == SweepDirection::LeftToRight) {
                CUDA_KERNEL_CHECK((build_left_tensor<<<bl, threads>>>(
                    newA1.d_data, wk.U_ws, Dl, d, chi, left_dim)));
                CUDA_KERNEL_CHECK((build_right_tensor<<<br, threads>>>(
                    newA2.d_data, wk.S_ws, wk.Vt_ws, chi, d, Dr, right_dim)));
            } else {
                CUDA_KERNEL_CHECK((build_left_tensor_with_s<<<bl, threads>>>(
                    newA1.d_data, wk.U_ws, wk.S_ws, Dl, d, chi, left_dim)));
                CUDA_KERNEL_CHECK((build_right_tensor_plain<<<br, threads>>>(
                    newA2.d_data, wk.Vt_ws, chi, d, Dr, right_dim)));
            }
        };
        if(!wk.production_mode) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            do_build();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float bk_ms = 0.f;
            cudaEventElapsedTime(&bk_ms, e0, e1);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            stats.t_reshape_ms += bk_ms;
        } else {
            do_build();
        }
    }

    if(exact_disc) {
        // Theta is normalized, so discarded = 1 - kept weight; exact to rounding, unlike the tail of tiny
        // eigenvalues (Gram) or the missing tail (gesvdr returns only the kept singular values)
        double kept = 0.0;
        if(rsvd) {
            for(int i = 0; i < stats.kept_chi; ++i) kept += wk.sv_host[i] * wk.sv_host[i];
        } else {
            double nrm = 0.0;
            if(gram_u) CUBLAS_CHECK(cublasDnrm2(la.blas, right_dim * stats.kept_chi, wk.Vt_ws, 1, &nrm));
            else       CUBLAS_CHECK(cublasDnrm2(la.blas, left_dim * stats.kept_chi, wk.U_ws, 1, &nrm));
            kept = nrm * nrm;
        }
        stats.discarded_weight = std::max(0.0, 1.0 - kept);
    }

    if(wk.diag_trunc && warm) {
        // exact top-chi subspace of rho vs the warm factor X (gram_ws is free scratch after the build)
        const double one = 1.0, zero = 0.0;
        const int chi = stats.kept_chi;
        CUBLAS_CHECK(cublasDsyrk(la.blas, CUBLAS_FILL_MODE_LOWER, gram_u ? CUBLAS_OP_N : CUBLAS_OP_T,
            gdim, gram_u ? right_dim : left_dim, &one, wk.Theta, left_dim, &zero, wk.gram_ws, gdim));
        CUSOLVER_CHECK(cusolverDnDsyevd(la.solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            gdim, wk.gram_ws, gdim, wk.gram_eval, wk.gram_work, wk.gram_lwork, wk.d_info_ws));
        std::vector<double> lam(gdim);
        CUDA_CHECK(cudaMemcpy(lam.data(), wk.gram_eval, gdim * sizeof(double), cudaMemcpyDeviceToHost));
        double exact_kept = 0.0;
        for(int i = 0; i < chi; ++i) exact_kept += std::max(0.0, lam[gdim - 1 - i]);
        if(wk.diag_ws_n < (size_t)chi * chi) {
            if(wk.diag_ws) cudaFree(wk.diag_ws);
            CUDA_CHECK(cudaMalloc(&wk.diag_ws, sizeof(double) * chi * chi)); wk.diag_ws_n = (size_t)chi * chi;
        }
        const double* X = gram_u ? wk.U_ws : wk.Vt_ws;   // orthonormal warm factor, gdim x chi, ld gdim
        CUBLAS_CHECK(cublasDgemm(la.blas, CUBLAS_OP_T, CUBLAS_OP_N, chi, chi, gdim, &one, X, gdim,
            wk.gram_ws + (size_t)(gdim - chi) * gdim, gdim, &zero, wk.diag_ws, chi));
        double nrm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(la.blas, chi * chi, wk.diag_ws, 1, &nrm));
        const double extra = exact_kept - (1.0 - stats.discarded_weight), sin2 = chi - nrm * nrm;
        wk.dt_extra_sum += extra; wk.dt_extra_max = std::max(wk.dt_extra_max, extra);
        wk.dt_sin2_sum += sin2;   wk.dt_sin2_max = std::max(wk.dt_sin2_max, sin2);   ++wk.dt_n;
    }

    U_tv.release(); S_tv.release(); Vt_tv.release();
    psi.A[site]     = std::move(newA1);
    psi.A[site + 1] = std::move(newA2);
    return stats;
}
