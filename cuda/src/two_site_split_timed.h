#pragma once
#include "mps.h"
#include "mpo.h"
#include "environment.h"
#include "linalg.h"
#include <cusolverDn.h>
#include <curand.h>
#include "heff_cutensor.h"
#include "heff_lowp.h"
#include "heff_struct.h"
#include <array>
#include <functional>
#include <map>
#include <random>
#include <memory>

enum class HeffBackend { Gemm, Cutensor, Struct };

enum class SweepDirection {
    LeftToRight,
    RightToLeft
};

// -----------------------------------------------------------------------
// Per-site timing breakdown — explicit categories, no remainder trick.
// -----------------------------------------------------------------------
struct TwoSiteUpdateStats {
    // Convergence
    double energy          = 0.0;
    double residual_norm   = 0.0;
    double discarded_weight = 0.0;
    int    kept_chi        = 0;
    int    davidson_iters  = 0;   // total H_eff applies this site
    bool   davidson_converged = false;  // final residual <= tolerance (false = restarts exhausted)
    double davidson_tol    = 0.0;       // residual tolerance used for this solve

    // Timing (ms)
    float t_davidson_ms  = 0.0f;  // FULL Davidson solver (GEMMs + syevd + H_eff, GPU event)
    float t_heff_ms      = 0.0f;  // H_eff apply sub-component only (GPU event, inside davidson)
    float t_svd_ms       = 0.0f;  // SVD kernel (GPU)
    float t_sync_ms      = 0.0f;  // blocking D→H syncs (chi readback)
    float t_alloc_ms     = 0.0f;  // GPU malloc/free for newA1/newA2 only (U/S/Vt now in workspace)
    float t_reshape_ms   = 0.0f;  // merge/build_left/build_right kernels
    float t_cpu_ms       = 0.0f;  // CPU-side: chi scan, discarded weight loop
};

// -----------------------------------------------------------------------
// Pre-allocated GPU workspace — avoids malloc/free inside the sweep loop.
// Create once per (D, d) configuration before the sweep, destroy after.
// -----------------------------------------------------------------------
struct SweepWorkspace {
    static constexpr int MAX_SUBSPACE = 160;
    int subspace_cap = MAX_SUBSPACE;  // envck tree: Davidson basis columns actually allocated (<= MAX_SUBSPACE)

    int max_dim;       // D*d*d*D (worst case bond dims equal D)
    int max_left_dim;  // D*d
    int max_right_dim; // d*D
    int max_rank;      // min(max_left_dim, max_right_dim)

    // Davidson workspace — raw device pointers, allocated at max size
    double* projected    = nullptr;   // MAX_SUBSPACE * MAX_SUBSPACE
    double* basis_mat    = nullptr;   // max_dim * MAX_SUBSPACE
    double* image_mat    = nullptr;   // max_dim * MAX_SUBSPACE
    double* diagonal     = nullptr;   // max_dim
    double* seed_v       = nullptr;   // max_dim
    double* current_v    = nullptr;   // max_dim
    // Per-iteration temporaries (reused each inner iter)
    double* basis_v      = nullptr;   // max_dim
    double* image_v      = nullptr;   // max_dim
    double* overlap_v    = nullptr;   // MAX_SUBSPACE
    double* ground_v     = nullptr;   // max_dim
    double* hground_v    = nullptr;   // max_dim
    double* residual_v   = nullptr;   // max_dim
    double* correction_v = nullptr;   // max_dim
    double* projection_v = nullptr;   // max_dim
    // Subspace eigenproblem workspace
    double* packed_v     = nullptr;   // MAX_SUBSPACE * MAX_SUBSPACE (copy for syevd)
    double* eigvec_v     = nullptr;   // MAX_SUBSPACE (coeffs)
    double* eig_vals     = nullptr;   // MAX_SUBSPACE
    double* eig_work     = nullptr;   // syevd work buffer (pre-sized)
    int     eig_lwork    = 0;
    int*    d_info_ws    = nullptr;   // shared device info int

    // Two-site ground state
    double* ground       = nullptr;   // max_dim
    double* Theta        = nullptr;   // max_left_dim * max_right_dim

    // SVD output (pre-allocated at max size, used as Tensor wrappers below)
    // newA1 and newA2 must be real Tensors since psi takes ownership — see update fn.

    // Pre-allocated SVD output buffers — avoids 3 cudaMalloc/cudaFree per site
    double* U_ws  = nullptr;   // max_left_dim  × max_left_dim
    double* S_ws  = nullptr;   // max_rank
    double* Vt_ws = nullptr;   // max_right_dim × max_right_dim

    // cuBLAS Heff scratch: each (D*d)^2 = max_left_dim^2
    double* heff_ML  = nullptr;
    double* heff_MR  = nullptr;
    double* heff_tmp = nullptr;

    // Host-side singular value buffer for chi selection + discarded weight (no extra alloc)
    std::vector<double> sv_host;

    // SVD truncation threshold; if <= 0, keep exactly min(D, rank) singular values (fixed-D mode)
    double svd_threshold = 1e-12;

    // When true: no per-Heff sync, no per-component events — end-to-end wall time only.
    bool production_mode = false;

    // Davidson budget: 0 = unchanged defaults (tol 1e-12 fixed-D, up to 8 restarts, no matvec cap).
    // max_matvec > 0 stops a local solve after that many H_eff applications (returns the current best
    // Ritz vector, flagged not converged unless it met the tolerance); tol_override > 0 replaces the tolerance.
    // Truncation of the two-site Theta (m x n):
    //   Jacobi : cuSOLVER gesvdj, full U/V (v6 default)
    //   Gesvd  : cuSOLVER gesvd (QR algorithm, thin factors) — what GPU ITensor uses
    //   Gram   : density-matrix route (White 1992): eigenvectors of Theta Theta^T (L->R) or Theta^T Theta (R->L)
    //            give the orthonormal factor exactly; the other factor is one GEMM (no division by sigma)
    //   Rsvd       : cuSOLVER Xgesvdr randomized rank-chi SVD (oversampling p = max(8, chi/8), 2 power iterations);
    //                falls back to Gesvd when chi + p exceeds the matrix rank
    //   GramRefine : Gram + one subspace-iteration step (U <- qr(Theta Theta^T U)) to clean the states at the cut
    // For Gram / GramRefine / Rsvd the discarded weight is computed exactly as 1 - (kept weight) after the build.
    //   Warm (experimental) : no eigendecomposition. Seed the kept orthonormal factor with the span of the current
    //                MPS tensor on the updated side, then warm_iters subspace-iteration steps X <- qr(rho X).
    //                Falls back to Gram when the bond dimension changes (schedule step) or chi != old bond dim.
    //                Fixed-D only (thresh <= 0): no singular values are computed.
    enum class SvdMethod { Jacobi, Gesvd, Gram, Rsvd, GramRefine, Warm };
    SvdMethod svd_method = SvdMethod::Jacobi;
    int warm_iters = 1;
    double warm_eta = 0.0;   // > 0: flow truncation X <- qr(X + eta rho X) per step (0 = plain power step rho X)
    bool warm_orth = false;  // orthonormalise the warm seed (QR) before the plain step (isolates the seed effect of warm_eta)
    long long warm_hits = 0, warm_fallbacks = 0;   // counters for the run log
    // Diagnostic (diag=1, slow): after each warm truncation also do the exact eigendecomposition and record
    //   extra = exact kept weight - warm kept weight (>= 0), sin2 = chi - ||X^T U_exact||_F^2 (sum sin^2 principal angles)
    bool   diag_trunc = false;
    double dt_extra_sum = 0, dt_extra_max = 0, dt_sin2_sum = 0, dt_sin2_max = 0;  long long dt_n = 0;
    double* diag_ws = nullptr;  size_t diag_ws_n = 0;
    void*   rsvd_work = nullptr;  size_t rsvd_work_bytes = 0;    // grow-only device workspace
    std::vector<char> rsvd_host_work;
    double* qr_tau = nullptr;     double* qr_work = nullptr;     int qr_lwork = 0;
    double* gram_ws    = nullptr;   // max(m,n)^2 : Gram matrix / eigenvectors, gesvd VT / transposed Theta
    double* gram_eval  = nullptr;   // max(m,n)   : eigenvalues
    double* gram_work  = nullptr;   // syevd workspace for max(m,n)
    int     gram_lwork = 0;
    double* gesvd_work = nullptr;   // gesvd workspace for max(m,n) square
    double* gesvd_rwork = nullptr;
    int     gesvd_lwork = 0;

    // Optional replacement for the Davidson local eigensolver (experimental solvers live in own_experimental/).
    // Must write the normalized two-site ground state (row-major [a,p,q,c], length Dl*d*d*Dr) into `ground`.
    std::function<bool(SweepWorkspace& wk, const MPS& psi, const MPO& H, const Environment& env, LinAlg& la,
                       int site, double& residual, double& energy, int& matvecs, bool& converged, double& tol)>
        local_solver;

    // Reduced-precision proxy: relative noise of size heff_noise added to every H_eff application (0 = off)
    double heff_noise = 0.0;
    // Stochastic Ritz selection (experimental): temperature for mixing in the 2nd Ritz vector at budget exit (0 = off)
    double ritz_T = 0.0;
    std::mt19937_64 ritz_rng{777};
    curandGenerator_t noise_gen = nullptr;  double* noise_buf = nullptr;  size_t noise_n = 0;

    // Optional hook after the local solve (any solver), before truncation: may replace `ground` (normalized,
    // row-major [a,p,q,c]) and must then set `energy` to its Rayleigh quotient. psi still holds the input tensors.
    std::function<void(SweepWorkspace& wk, const MPS& psi, const MPO& H, const Environment& env, LinAlg& la,
                       int site, double& energy)> post_solve;

    // Davidson seed merge + diagonal preconditioner via O(D^3) / factorised contractions instead of the naive
    // per-element kernels (which were ~44% of GPU time at D=1024). Same results; default off (v6 behaviour).
    bool   davidson_fast_kernels = false;

    int    davidson_max_matvec   = 0;
    double davidson_tol_override = 0.0;

    // H_eff implementation used inside Davidson. Cutensor plans are cached per (Dl, Dr, Wl, Wm, Wr).
    HeffBackend heff_backend = HeffBackend::Gemm;
    std::map<std::array<int,5>, std::unique_ptr<HeffCutensor>> cutensor_plans;
    HeffCutensor& cutensor_for(int Dl, int d, int Dr, int Wl, int Wm, int Wr);
    // Reduced-precision H_eff (experimental, cutensor backend only): 0 = FP64 (default), 1 = TF32, 2 = FP32
    int heff_lowp = 0;
    std::map<std::array<int,6>, std::unique_ptr<HeffLowp>> lowp_plans;
    HeffLowp& lowp_for(int Dl, int d, int Dr, int Wl, int Wm, int Wr);
    std::map<std::array<int,4>, std::unique_ptr<HeffStruct>> struct_plans;
    HeffStruct& struct_for(int Dl, int d, int Dr, int Wl, int Wr);

    explicit SweepWorkspace(int D, int d, cusolverDnHandle_t solver, int cap = MAX_SUBSPACE);
    ~SweepWorkspace();

    // No copy
    SweepWorkspace(const SweepWorkspace&) = delete;
    SweepWorkspace& operator=(const SweepWorkspace&) = delete;

private:
    void alloc(double*& ptr, size_t n);
};

// -----------------------------------------------------------------------
// Public entry point — uses workspace to avoid in-loop allocations.
// -----------------------------------------------------------------------
TwoSiteUpdateStats two_site_update_timed(
    MPS& psi,
    const MPO& H,
    Environment& env,
    LinAlg& la,
    SweepWorkspace& wk,
    int site,
    SweepDirection direction
);

// Building blocks for custom local solvers (SweepWorkspace::local_solver):
// out = H_eff(site, site+1) * in with the workspace's configured backend; in/out are length Dl*d*d*Dr.
void sweep_apply_heff(SweepWorkspace& wk, double* out, const double* in, const MPS& psi, const MPO& H,
                      const Environment& env, LinAlg& la, int site);
// theta = A[site] * A[site+1] merged into the two-site layout (row-major [a,p,q,c]).
void sweep_merge_theta(double* theta, const MPS& psi, int site);
