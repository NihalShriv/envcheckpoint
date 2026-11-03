// dmrg_exp.cu — copy of apps/dmrg_run.cu (same driver, same metrics) plus experimental local solvers.
//   solver=davidson (default, unchanged) | cheb (Chebyshev-basis s-step Krylov, own_experimental/src/cheb_solver.cu)
//   cheb_deg=m (H_eff applications per restart = m+1), cheb_restarts=r, cheb_margin=lo margin
//   dav_fast=1: Davidson with O(D^3) seed merge + factorised diagonal (fair baseline for the new solver)
//
// Usage (key=value, any order):
//   bin/dmrg_run model=cyl Lx=16 Ly=4 J2=0.5 sched=64:2,128:2,256:2,512:2,1024:4 heff=gemm
//                ckpt=results/v6/run.ckpt out=results/v6/run.csv max_hours=22 thresh=0 seed=1
//   thresh    : SVD weight threshold; 0 (default) = fixed-D, keep min(D, rank)
//   svd       : jacobi (gesvdj, full U/V; default) | gesvd (QR algorithm, thin) | gram (density-matrix eigen)
//               | rsvd (randomized rank-chi, gesvdr) | gram_refine (gram + one subspace-iteration step)
//               | warm (no eigensolve: seed = current tensor, warm_iters subspace-iteration steps; fixed-D)
//   flow_final: 1 = apply warm_eta only at the final D stage (hybrid: plain warm step before)
//   warm_eta  : > 0 = flow truncation for svd=warm: X <- qr(X + eta rho X) (eta -> inf = plain step)
//   conv      : 1 = convergence-driven stages (sched D:n = max n sweeps), stage_tol, kick_tol, final_tol, conv_min
//   envcache  : 1 = keep environments across half-sweeps (no full rebuild per pass; same results)
//   ritz_T    : T > 0 = stochastic Ritz selection at the Davidson budget exit (mix 2nd Ritz vector, weight
//               U(0,1) exp(-gap/T), random sign), off for the last ritz_clean sweeps (default 3)
//   accel     : 1 = feasibility probe: tensor-wise extrapolation along the last sweep + exact-energy line search
//   lowp      : fp64 (default) | tf32 | fp32 = H_eff precision (cutensor backend), FP64 for the last lowp_clean sweeps
//   heff_noise: eps > 0 = random relative error eps on every H_eff (reduced-precision proxy), off for the last
//               noise_clean sweeps (default 3)
//   diis      : m >= 2 = Pulay/Anderson extrapolation across sweeps with m-1 stored pairs (experimental)
//   diag      : 1 = slow feasibility diagnostics (solver=cheb): sweep-to-sweep alignment of local states and,
//               for svd=warm, distance from the exact truncation (DIAG lines on stderr)
//   dav_maxmv : max H_eff applications per local Davidson solve (0 = unlimited, default solver)
//   dav_tol   : Davidson residual tolerance override (0 = default 1e-12 in fixed-D mode)
//   compress  : 1 = exact SVD compression of the bond-list MPO (YC4: W 23 -> 14), same Hamiltonian to rounding
//   production: 1 = no per-Heff/per-site timing syncs (heff_s, davidson_s, svd_s, env_s read 0); full run time in .done
//   model=cyl : J1-J2 cylinder (j1j2_cylinder_bonds), model=tri : triangular ladder (Ly ignored)
//               model=tric : triangular J1-J2 XC cylinder (own_experimental; Ly >= 4)
//   init      : random (default) | stripe_x | stripe_y (product states);  init_from=<ckpt> : start from a saved MPS
//   sched     : D:sweeps stages, D ramps up; total sweeps = sum of stage sweeps
//   heff      : gemm (per-MPO-channel cuBLAS, v5) | cutensor (generic baseline) | struct (MPO-structure-aware)
//
// Restart: if <ckpt> exists the run resumes after the last completed sweep (appending to <out>).
// Exit codes: 0 = schedule finished (writes <ckpt>.done), 3 = stopped early for the wall-time budget.
//
// CSV columns: model,Lx,Ly,N,W,J2,heff,sweep,D_cap,max_chi,energy,dE,max_disc,sum_disc,
//              sweep_s,heff_s,heff_frac,davidson_s,svd_s,env_s,n_heff,peak_mem_gb,
//              n_solves,n_unconverged,max_residual,davidson_tol   (Davidson convergence per sweep)

#include "mps.h"
#include "mpo.h"
#include "environment.h"
#include "canonical.h"
#include "linalg.h"
#include "two_site_split_timed.h"
#include "energy_eval.h"
#include "norm_eval.h"
#include "checkpoint.h"
#include "cheb_solver.h"
#include "heff_cutensor.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <map>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cuda_runtime.h>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// symdiag=1: <psi| F |psi> / <psi|psi> with F = prod_i sigma^x_i (global spin flip), by transfer matrices:
// E_{i+1} = sum_p A_p^T E_i B_p, B_p = A_{1-p} (flip) or A_p (norm). A[a + Dl*(p + d*c)] -> slice p is Dl x Dr, ld Dl*d.
static double flip_overlap(const MPS& psi, cublasHandle_t h, bool flip)
{
    double *E = nullptr, *T = nullptr, *En = nullptr;
    CUDA_CHECK(cudaMalloc(&E, sizeof(double)));
    const double one = 1.0, zero = 0.0;
    CUDA_CHECK(cudaMemcpy(E, &one, sizeof(double), cudaMemcpyHostToDevice));
    int Dl = 1;
    for (int i = 0; i < psi.N; ++i) {
        const Tensor& A = psi.A[i];
        const int d = A.shape[1], Dr = A.shape[2], ld = Dl * d;
        CUDA_CHECK(cudaMalloc(&T, sizeof(double) * Dl * Dr)); CUDA_CHECK(cudaMalloc(&En, sizeof(double) * Dr * Dr));
        CUDA_CHECK(cudaMemset(En, 0, sizeof(double) * Dr * Dr));
        for (int p = 0; p < d; ++p) {
            const double* Ap = A.d_data + (size_t)p * Dl;
            const double* Bp = A.d_data + (size_t)(flip ? d - 1 - p : p) * Dl;
            CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, Dl, Dr, Dl, &one, E, Dl, Bp, ld, &zero, T, Dl));   // E B_p
            CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, Dr, Dr, Dl, &one, Ap, ld, T, Dl, &one, En, Dr));  // += A_p^T (E B_p)
        }
        CUDA_CHECK(cudaFree(E)); CUDA_CHECK(cudaFree(T)); E = En; Dl = Dr;
    }
    double r = 0; CUDA_CHECK(cudaMemcpy(&r, E, sizeof(double), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaFree(E));
    return r;
}

// diis=m (experimental, feasibility): Pulay/Anderson extrapolation across sweeps, per (bond, direction).
// Keeps the last m-1 (input x, output g) pairs of the local solve (x = merged input theta, g = solver output,
// both normalized and sign-aligned), adds the current pair, minimises ||sum c_i (g_i - x_i)|| with sum c_i = 1,
// mixes g* = sum c_i g_i, and accepts g* only if its Rayleigh quotient (one extra H_eff) is lower than the
// solver's energy. Possible only because warm truncation keeps the bond bases aligned between sweeps.
struct SweepDIIS {
    int m = 3; double cmax = 5.0;
    struct Slot { std::vector<double*> x, g; size_t n = 0; int head = 0; };
    std::map<int, Slot> slots;
    int dir = 0;
    double *xin = nullptr, *mix = nullptr, *hmix = nullptr; size_t cap = 0;
    long n_try = 0, n_acc = 0; double gain_sum = 0;
    static double dot(cublasHandle_t h, size_t n, const double* a, const double* b) {
        double r = 0; CUBLAS_CHECK(cublasDdot(h, n, a, 1, b, 1, &r)); return r;
    }
    void apply(SweepWorkspace& wk, const MPS& psi, const MPO& H, const Environment& env, LinAlg& la,
               int site, double& energy) {
        const size_t n = (size_t)psi.A[site].shape[0] * psi.d * psi.d * psi.A[site + 1].shape[2];
        cublasHandle_t h = la.blas;
        if (cap < n) {
            for (double* p : {xin, mix, hmix}) if (p) cudaFree(p);
            CUDA_CHECK(cudaMalloc(&xin, n * 8)); CUDA_CHECK(cudaMalloc(&mix, n * 8)); CUDA_CHECK(cudaMalloc(&hmix, n * 8));
            cap = n;
        }
        const double* g0 = wk.ground;
        sweep_merge_theta(xin, psi, site);                        // the input state of this local solve
        double nx = std::sqrt(dot(h, n, xin, xin)), sx = (dot(h, n, xin, g0) < 0 ? -1.0 : 1.0) / nx;
        CUBLAS_CHECK(cublasDscal(h, n, &sx, xin, 1));
        Slot& s = slots[2 * site + dir];
        if (s.n != n) { for (double* p : s.x) cudaFree(p); for (double* p : s.g) cudaFree(p); s.x.clear(); s.g.clear(); s.n = n; s.head = 0; }
        const int k = s.x.size();
        if (k >= 1) {
            std::vector<const double*> X{xin}, G{g0}; std::vector<double> sg{1.0};
            for (int j = 0; j < k; ++j) { X.push_back(s.x[j]); G.push_back(s.g[j]); sg.push_back(dot(h, n, s.g[j], g0) < 0 ? -1.0 : 1.0); }
            const int K = k + 1;
            std::vector<double> M(K * K);
            for (int i = 0; i < K; ++i) for (int j = i; j < K; ++j) {
                double v = sg[i] * sg[j] * (dot(h, n, G[i], G[j]) - dot(h, n, G[i], X[j]) - dot(h, n, X[i], G[j]) + dot(h, n, X[i], X[j]));
                M[i * K + j] = M[j * K + i] = v;
            }
            double tr = 0; for (int i = 0; i < K; ++i) tr += M[i * K + i];
            for (int i = 0; i < K; ++i) M[i * K + i] += 1e-12 * tr + 1e-300;
            std::vector<double> c(K, 1.0);                          // solve M c = 1 (Gaussian elimination, K <= 4)
            for (int p = 0; p < K; ++p) {
                int piv = p; for (int r = p + 1; r < K; ++r) if (std::fabs(M[r * K + p]) > std::fabs(M[piv * K + p])) piv = r;
                for (int q = 0; q < K; ++q) std::swap(M[p * K + q], M[piv * K + q]); std::swap(c[p], c[piv]);
                for (int r = p + 1; r < K; ++r) { double f = M[r * K + p] / M[p * K + p];
                    for (int q = p; q < K; ++q) M[r * K + q] -= f * M[p * K + q]; c[r] -= f * c[p]; }
            }
            for (int p = K - 1; p >= 0; --p) { for (int q = p + 1; q < K; ++q) c[p] -= M[p * K + q] * c[q]; c[p] /= M[p * K + p]; }
            double sc = 0, cm = 0; for (double v : c) sc += v; for (double& v : c) { v /= sc; cm = std::max(cm, std::fabs(v)); }
            if (std::isfinite(cm) && cm <= cmax) {
                CUDA_CHECK(cudaMemset(mix, 0, n * 8));
                for (int i = 0; i < K; ++i) { double a = c[i] * sg[i]; CUBLAS_CHECK(cublasDaxpy(h, n, &a, G[i], 1, mix, 1)); }
                double inv = 1.0 / std::sqrt(dot(h, n, mix, mix)); CUBLAS_CHECK(cublasDscal(h, n, &inv, mix, 1));
                sweep_apply_heff(wk, hmix, mix, psi, H, env, la, site);
                const double E = dot(h, n, mix, hmix);
                ++n_try;
                if (E < energy - 1e-14) {
                    CUDA_CHECK(cudaMemcpy(wk.ground, mix, n * 8, cudaMemcpyDeviceToDevice));
                    gain_sum += energy - E; energy = E; ++n_acc;
                }
            }
        }
        // store (input, final output) in the ring of m-1 previous pairs
        if ((int)s.x.size() < m - 1) {
            double *a, *b; CUDA_CHECK(cudaMalloc(&a, n * 8)); CUDA_CHECK(cudaMalloc(&b, n * 8));
            s.x.push_back(a); s.g.push_back(b); s.head = s.x.size() - 1;
        } else s.head = (s.head + 1) % (m - 1);
        CUDA_CHECK(cudaMemcpy(s.x[s.head], xin, n * 8, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(s.g[s.head], wk.ground, n * 8, cudaMemcpyDeviceToDevice));
    }
    void report(int sweep) {
        std::cerr << std::setprecision(4) << "DIIS sweep=" << sweep << " tries=" << n_try << " accepted=" << n_acc
                  << " mean_gain=" << (n_acc ? gain_sum / n_acc : 0.0) << "\n";
        n_try = n_acc = 0; gain_sum = 0;
    }
};

// diag=1 feasibility probe for cross-sweep momentum: per (bond, direction) keep the previous same-direction local
// ground state theta (sign-aligned) and its change delta. Per sweep: drift = 1 - |<theta_k, theta_{k-1}>| and
// cos(delta_k, delta_{k-1}); cos near +1 = updates keep pointing the same way (momentum would help),
// near 0 = basis rotation / noise dominates. Slow (host syncs, ~3 extra vectors per bond); diagnostics only.
struct SweepAlignDiag {
    struct Slot { double *th = nullptr, *dl = nullptr, *tmp = nullptr; size_t n = 0; bool has_dl = false; };
    std::map<int, Slot> slots;
    int dir = 0;
    double drift_sum = 0, drift_max = 0, dnorm_sum = 0, cos_sum = 0;  long n_drift = 0, n_cos = 0, n_cos_pos = 0;
    void record(cublasHandle_t h, int site, const double* g, size_t n) {
        Slot& s = slots[2 * site + dir];
        if (s.n != n) {
            for (double* p : {s.th, s.dl, s.tmp}) if (p) cudaFree(p);
            CUDA_CHECK(cudaMalloc(&s.th, n * sizeof(double))); CUDA_CHECK(cudaMalloc(&s.dl, n * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&s.tmp, n * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(s.th, g, n * sizeof(double), cudaMemcpyDeviceToDevice));
            s.n = n; s.has_dl = false; return;
        }
        double ov = 0, m1 = -1, dn = 0;
        CUBLAS_CHECK(cublasDdot(h, n, g, 1, s.th, 1, &ov));
        double sg = ov < 0 ? -1.0 : 1.0;
        // tmp = sg*g - th  (new delta);  th <- sg*g
        CUDA_CHECK(cudaMemcpy(s.tmp, g, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUBLAS_CHECK(cublasDscal(h, n, &sg, s.tmp, 1));
        CUBLAS_CHECK(cublasDaxpy(h, n, &m1, s.th, 1, s.tmp, 1));
        CUBLAS_CHECK(cublasDnrm2(h, n, s.tmp, 1, &dn));
        const double drift = 1.0 - std::fabs(ov);
        drift_sum += drift; drift_max = std::max(drift_max, drift); dnorm_sum += dn; ++n_drift;
        if (s.has_dl && dn > 1e-10) {
            double c = 0, dp = 0;
            CUBLAS_CHECK(cublasDdot(h, n, s.tmp, 1, s.dl, 1, &c));
            CUBLAS_CHECK(cublasDnrm2(h, n, s.dl, 1, &dp));
            if (dp > 1e-10) { c /= dn * dp; cos_sum += c; ++n_cos; n_cos_pos += c > 0.5; }
        }
        std::swap(s.dl, s.tmp); s.has_dl = dn > 1e-10;
        CUDA_CHECK(cudaMemcpy(s.th, g, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUBLAS_CHECK(cublasDscal(h, n, &sg, s.th, 1));
    }
    void report(int sweep, SweepWorkspace& wk) {
        std::cerr << std::setprecision(4) << "DIAG sweep=" << sweep
                  << " drift_mean=" << (n_drift ? drift_sum / n_drift : NAN) << " drift_max=" << drift_max
                  << " dnorm_mean=" << (n_drift ? dnorm_sum / n_drift : NAN)
                  << " cos_mean=" << (n_cos ? cos_sum / n_cos : NAN)
                  << " cos_gt_half=" << (n_cos ? double(n_cos_pos) / n_cos : NAN) << " n_cos=" << n_cos
                  << " | trunc extra_disc_mean=" << (wk.dt_n ? wk.dt_extra_sum / wk.dt_n : NAN)
                  << " extra_disc_max=" << wk.dt_extra_max
                  << " sin2_mean=" << (wk.dt_n ? wk.dt_sin2_sum / wk.dt_n : NAN) << " sin2_max=" << wk.dt_sin2_max
                  << " n=" << wk.dt_n << "\n";
        drift_sum = drift_max = dnorm_sum = cos_sum = 0; n_drift = n_cos = n_cos_pos = 0;
        wk.dt_extra_sum = wk.dt_extra_max = wk.dt_sin2_sum = wk.dt_sin2_max = 0; wk.dt_n = 0;
    }
};

int main(int argc, char** argv)
{
    const auto t_start = clk::now();   // full run time: init, sweeps, checkpoints, final checks
    std::map<std::string, std::string> arg = {
        {"gpu_cap_gb",""}, {"model","cyl"}, {"Lx","8"}, {"Ly","4"}, {"J2","0.5"}, {"sched","64:2,128:2"}, {"heff","gemm"},
        {"ckpt","results/v6/run.ckpt"}, {"out","results/v6/run.csv"}, {"max_hours","1e9"},
        {"thresh","0"}, {"seed","1"}, {"production","0"}, {"compress","0"},
        {"dav_maxmv","0"}, {"dav_tol","0"}, {"svd","jacobi"},
        {"solver","davidson"}, {"cheb_deg","3"}, {"cheb_restarts","1"}, {"cheb_margin","0.05"}, {"dav_fast","0"}, {"warm_iters","1"}, {"diag","0"}, {"diis","0"}, {"heff_noise","0"}, {"noise_clean","3"}, {"lowp","fp64"}, {"lowp_clean","3"}, {"bondlog","0"}, {"accel","0"}, {"ritz_T","0"}, {"ritz_clean","3"}, {"symdiag","0"}, {"envcache","0"}, {"conv","0"}, {"stage_tol","1e-5"}, {"final_tol","1e-8"}, {"kick_tol","1e-7"}, {"conv_min","2"}, {"warm_eta","0"}, {"warm_orth","0"}, {"flow_final","0"}, {"dumptheta","0"}, {"init","random"}, {"init_from",""}, {"evalonly","0"}, {"envspec","0"}, {"envck","0"}, {"memtrack","0"}, {"envdisk",""}, {"envck2","0"}, {"envdisk_async","0"}, {"memprof","0"}, {"dav_cols",""}, {"heff_chunks","1"}, {"envpin","0"}, {"envpin_prefetch","2"} };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto eq = s.find('=');
        if (eq == std::string::npos || !arg.count(s.substr(0, eq))) { std::cerr << "unknown arg " << s << "\n"; return 2; }
        arg[s.substr(0, eq)] = s.substr(eq + 1);
    }
    const std::string model = arg["model"];
    const int Lx = std::stoi(arg["Lx"]);
    const int Ly = model == "tri" ? 2 : std::stoi(arg["Ly"]);
    const double J2 = std::stod(arg["J2"]);
    const double max_s = std::stod(arg["max_hours"]) * 3600.0;

    // D schedule -> per-sweep cap
    std::vector<int> D_of_sweep;
    { std::stringstream ss(arg["sched"]); std::string tok;
      while (std::getline(ss, tok, ',')) {
          auto c = tok.find(':');
          int D = std::stoi(tok.substr(0, c)), n = std::stoi(tok.substr(c + 1));
          for (int k = 0; k < n; ++k) D_of_sweep.push_back(D);
      } }
    const int n_sweeps = D_of_sweep.size();
    const int D_max = *std::max_element(D_of_sweep.begin(), D_of_sweep.end());

    // model=tric: triangular-lattice J1-J2 XC cylinder (own_experimental). Site (x,y) at x e1 + y e2, e1 = (1,0),
    // e2 = (1/2, sqrt3/2); periodic in y (circumference along e2), open in x; index x*Ly + y.
    // J1 neighbours (1,0), (0,1), (-1,1); J2 neighbours (1,1), (-1,2), (-2,1). Needs Ly >= 4 (no double counting).
    auto tric_bonds = [&]() {
        std::vector<Bond> b;
        auto add = [&](int x, int y, int dx, int dy, double J) {
            const int x2 = x + dx; if (x2 < 0 || x2 >= Lx || J == 0.0) return;
            b.push_back({x * Ly + y, x2 * Ly + ((y + dy) % Ly + Ly) % Ly, J});
        };
        for (int x = 0; x < Lx; ++x) for (int y = 0; y < Ly; ++y) {
            add(x, y, 1, 0, 1.0); add(x, y, 0, 1, 1.0); add(x, y, -1, 1, 1.0);
            add(x, y, 1, 1, J2);  add(x, y, -1, 2, J2); add(x, y, -2, 1, J2);
        }
        for (auto& e : b) if (e.i > e.j) std::swap(e.i, e.j);
        return b;
    };
    if (model == "tric" && Ly < 4) { std::cerr << "model=tric needs Ly >= 4\n"; return 2; }
    std::vector<Bond> bonds = model == "tri" ? tri_ladder_bonds(Lx, 1.0)
                            : model == "tric" ? tric_bonds() : j1j2_cylinder_bonds(Lx, Ly, 1.0, J2);
    const int N = Lx * Ly, W_raw = heisenberg_mpo_width(bonds);
    MPO H(N, 2, W_raw);
    initialize_heisenberg_bonds(H, bonds);
    const int W = arg["compress"] == "1" ? compress_mpo(H) : W_raw;   // exact SVD compression (W_raw -> W)

    LinAlg la;
    std::srand(std::stoi(arg["seed"]));
    MPS psi(N, 2, D_of_sweep[0]);
    RunState st;
    const std::string ckpt = arg["ckpt"];
    bool restarted = load_checkpoint(ckpt, psi, st);
    if (!restarted) {
        // init_from=<ckpt>: start a fresh schedule from another run's saved MPS (e.g. a J2 = 0 ground state)
        // init=stripe_x | stripe_y: product state, spin up where x (resp. y) is even, else down (total Sz = 0 needs even L)
        RunState other;
        const bool loaded = arg["init_from"].size() && load_checkpoint(arg["init_from"], psi, other);
        if (arg["init_from"].size() && !loaded) { std::cerr << "init_from: cannot load " << arg["init_from"] << "\n"; return 2; }
        if (loaded) {
            // the first pass sweeps L->R and builds right environments: the loaded state must be RIGHT-canonical
            // (left-canonicalising it here garbled the first sweep and destroyed the loaded state; fixed in v21)
            right_canonicalize(psi, la);
            std::cerr << "initial state loaded from " << arg["init_from"] << " (right-canonicalised)\n";
        } else if (arg["init"] == "stripe_x" || arg["init"] == "stripe_y") {
            for (int i = 0; i < N; ++i) {
                const int x = i / Ly, y = i % Ly;
                const bool up = ((arg["init"] == "stripe_x" ? x : y) % 2) == 0;
                psi.A[i] = Tensor({1, 2, 1}); psi.A[i].from_host(up ? std::vector<double>{1.0, 0.0} : std::vector<double>{0.0, 1.0});
            }
        } else if (arg["init"] != "random") { std::cerr << "init must be random|stripe_x|stripe_y\n"; return 2; }
        else psi.random_initialize();
        if (!loaded) left_canonicalize(psi, la);   // fresh starts: unchanged v6 flow
    }

    // envck tree: the workspace must also fit a loaded state with a larger bond than the schedule (restart to a
    // smaller D crashed before), and the Davidson basis is sized to what dav_maxmv can use (dav_cols to override).
    int D_ws = D_max;
    for (auto& a : psi.A) D_ws = std::max({D_ws, a.shape[0], a.shape[2]});
    const int dav_mv = std::stoi(arg["dav_maxmv"]);
    const int dav_cols = arg["dav_cols"].size() ? std::stoi(arg["dav_cols"]) : (dav_mv > 0 ? dav_mv + 4 : SweepWorkspace::MAX_SUBSPACE);
    heff_set_chunks(std::stoi(arg["heff_chunks"]));   // envck tree: chunked H_eff intermediates (1 = off)
    SweepWorkspace wk(D_ws, 2, la.solver, dav_cols);
    wk.svd_threshold = std::stod(arg["thresh"]);
    if (arg["svd"] == "jacobi") wk.svd_method = SweepWorkspace::SvdMethod::Jacobi;
    else if (arg["svd"] == "gesvd") wk.svd_method = SweepWorkspace::SvdMethod::Gesvd;
    else if (arg["svd"] == "gram") wk.svd_method = SweepWorkspace::SvdMethod::Gram;
    else if (arg["svd"] == "rsvd") wk.svd_method = SweepWorkspace::SvdMethod::Rsvd;
    else if (arg["svd"] == "gram_refine") wk.svd_method = SweepWorkspace::SvdMethod::GramRefine;
    else if (arg["svd"] == "warm") wk.svd_method = SweepWorkspace::SvdMethod::Warm;
    else { std::cerr << "svd must be jacobi|gesvd|gram|rsvd|gram_refine|warm\n"; return 2; }
    wk.warm_iters = std::stoi(arg["warm_iters"]);
    wk.warm_eta = std::stod(arg["warm_eta"]);
    wk.warm_orth = arg["warm_orth"] == "1";
    if (wk.warm_iters < 0) { std::cerr << "warm_iters must be >= 0 (0 = re-orthonormalise the seed only)\n"; return 2; }
    wk.davidson_max_matvec = std::stoi(arg["dav_maxmv"]);
    wk.davidson_tol_override = std::stod(arg["dav_tol"]);
    wk.production_mode = arg["production"] == "1";   // no per-Heff timing syncs (heff_s/davidson_s/svd_s read 0)
    wk.davidson_fast_kernels = arg["dav_fast"] == "1";
    if (arg["solver"] == "cheb") {
        ChebParams cp;
        cp.degree = std::stoi(arg["cheb_deg"]);
        cp.restarts = std::stoi(arg["cheb_restarts"]);
        cp.lo_margin = std::stod(arg["cheb_margin"]);
        // H_eff eigenvalues lie inside the global spectrum; for H = sum_b J_b S_i.S_j each bond term is in
        // [-3/4, 1/4] |J_b|, so sum_b |J_b| / 4 bounds the spectrum from above.
        for (const auto& b : bonds) cp.hi += std::fabs(b.J) / 4.0;
        cp.hi *= 1.0 + 1e-6;
        wk.local_solver = make_cheb_solver(cp);
        std::cerr << "local solver: Chebyshev s-step Krylov, degree " << cp.degree << " (" << cp.degree + 1
                  << " H_eff/solve), restarts " << cp.restarts << ", spectrum upper bound " << cp.hi << "\n";
    } else if (arg["solver"] != "davidson") { std::cerr << "solver must be davidson|cheb\n"; return 2; }
    if (arg["lowp"] != "fp64" && arg["lowp"] != "tf32" && arg["lowp"] != "fp32") { std::cerr << "lowp must be fp64|tf32|fp32\n"; return 2; }
    // bondlog=1: per-solve log <out>.bonds (sweep, solve index: 0..N-2 = L->R sites, then R->L, residual, energy, disc)
    std::ofstream bondlog; if (arg["bondlog"] == "1") { bondlog.open(arg["out"] + ".bonds"); bondlog << "sweep,idx,residual,energy,disc\n" << std::setprecision(12); }
    const int diis_m = std::stoi(arg["diis"]);
    const bool diis = diis_m >= 2;
    SweepDIIS ds;
    if (diis) {
        ds.m = diis_m;
        wk.post_solve = [&ds](SweepWorkspace& w, const MPS& p, const MPO& h, const Environment& e, LinAlg& l,
                              int site, double& en) { ds.apply(w, p, h, e, l, site, en); };
        std::cerr << "DIIS across sweeps: m=" << diis_m << " (history " << diis_m - 1 << " + current), energy-safeguarded\n";
    }
    const bool diag = arg["diag"] == "1";
    SweepAlignDiag dg;
    // dumptheta=1: write the (normalised, mixed-canonical) two-site ground state at bonds N/2-1 and N/2+3 during
    // every L->R pass to <out>.theta_<site>.bin (overwritten, so the last L->R pass remains):
    // int64 Dl, d, Dr, then Dl*d*d*Dr doubles, row-major [a,p,q,c]. Analysis: scripts/disentangle_proxy.py.
    if (arg["dumptheta"] == "1") {
        if (diis) { std::cerr << "dumptheta and diis both use post_solve\n"; return 2; }
        const std::string pre = arg["out"];
        wk.post_solve = [&dg, pre, N](SweepWorkspace& w, const MPS& p, const MPO&, const Environment&, LinAlg&,
                                       int site, double&) {
            if (dg.dir != 0 || (site != N / 2 - 1 && site != N / 2 + 3)) return;
            const int64_t Dl = p.A[site].shape[0], d = p.d, Dr = p.A[site + 1].shape[2];
            std::vector<double> h((size_t)(Dl * d * d * Dr));
            CUDA_CHECK(cudaMemcpy(h.data(), w.ground, h.size() * sizeof(double), cudaMemcpyDeviceToHost));
            std::ofstream f(pre + ".theta_" + std::to_string(site) + ".bin", std::ios::binary);
            f.write((const char*)&Dl, 8); f.write((const char*)&d, 8); f.write((const char*)&Dr, 8);
            f.write((const char*)h.data(), h.size() * sizeof(double));
        };
    }
    if (diag) {
        if (!wk.local_solver) { std::cerr << "diag=1 needs solver=cheb\n"; return 2; }
        wk.diag_trunc = true;
        auto base = wk.local_solver;
        wk.local_solver = [base, &dg](SweepWorkspace& w, const MPS& p, const MPO& h, const Environment& e, LinAlg& l,
                                      int site, double& res, double& en, int& mv, bool& conv, double& tol) {
            bool r = base(w, p, h, e, l, site, res, en, mv, conv, tol);
            dg.record(l.blas, site, w.ground, (size_t)p.A[site].shape[0] * p.d * p.d * p.A[site + 1].shape[2]);
            return r;
        };
    }
    if (arg["heff"] == "gemm") wk.heff_backend = HeffBackend::Gemm;
    else if (arg["heff"] == "cutensor") wk.heff_backend = HeffBackend::Cutensor;
    else if (arg["heff"] == "struct") wk.heff_backend = HeffBackend::Struct;
    else { std::cerr << "heff must be gemm|cutensor|struct\n"; return 2; }

    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (!arg["gpu_cap_gb"].empty()) {   // emulate a smaller card: hold everything above the cap
        const size_t cap = (size_t)(std::stod(arg["gpu_cap_gb"]) * 1e9);
        if (cap < total_b) {
            void* ballast = nullptr;
            CUDA_CHECK(cudaMalloc(&ballast, total_b - cap));
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            std::cerr << "gpu_cap_gb=" << cap / 1e9 << " ballast_gb=" << (total_b - cap) / 1e9 << " free_gb=" << free_b / 1e9 << "\n";
        }
    }
    std::cerr << model << " Lx=" << Lx << " Ly=" << Ly << " N=" << N << " W=" << W << " (raw " << W_raw << ") bonds=" << bonds.size()
              << " heff=" << arg["heff"] << " sweeps=" << n_sweeps << " D_max=" << D_max
              << (restarted ? " RESTART after sweep " + std::to_string(st.sweeps_done) : std::string(" fresh start"))
              << " gpu_mem_total_gb=" << total_b / 1e9 << "\n";

    const std::string out = arg["out"];
    bool new_file = !std::ifstream(out).good();
    std::ofstream csv(out, std::ios::app);
    csv << std::setprecision(12);
    if (new_file)
        csv << "model,Lx,Ly,N,W,J2,heff,sweep,D_cap,max_chi,energy,dE,max_disc,sum_disc,"
               "sweep_s,heff_s,heff_frac,davidson_s,svd_s,env_s,n_heff,peak_mem_gb,"
               "n_solves,n_unconverged,max_residual,davidson_tol\n";

    CUDA_CHECK(cudaDeviceSynchronize());
    const double init_s = secs(t_start, clk::now());   // CUDA init, MPO/MPS build, canonicalize, workspace
    double ckpt_s = 0.0, sweeps_s = 0.0;
    const auto t_job = clk::now();
    double last_sweep_s = 0.0, E_prev = st.energy;
    const bool accel = arg["accel"] == "1";
    std::vector<Tensor> prevA;            // accel: MPS tensors after the previous final-D sweep
    MPS trial(N, psi.d, 1);               // accel: scratch MPS whose tensors are replaced per trial
    const bool envcache = arg["envcache"] == "1";
    // envck=k: store only every k-th environment block of the far side, rebuild a segment (<= k blocks) when the
    // sweep reaches it. Exact (same contractions); memory ~ N/k + k blocks instead of N; ~1 extra env build/site.
    const int envck = std::stoi(arg["envck"]);
    const bool memtrack = arg["memtrack"] == "1";   // sample device memory after every bond (true per-sweep peak)
    if (envck > 0 && envcache) { fprintf(stderr, "envck needs envcache=0\n"); return 1; }
    // envck2=k2 (with envck=k): two-level (revolve-like) schedule - a rebuilt segment keeps only every k2-th block,
    // the gaps are rebuilt again from those; memory ~ N/k + k/k2 + k2 blocks for ~2 extra env builds per site.
    const int envck2 = std::stoi(arg["envck2"]);
    auto ck_keep = [&](int i) { return envck <= 0 || i % envck == 0 || i == 0 || i == N; };
    // envdisk=<dir> (with envck=k): baseline for comparison - blocks that envck would drop are written to disk
    // (synchronous, like ITensor DiskProjMPO / TeNPy cache) and read back when needed instead of recomputed.
    const std::string envdisk = arg["envdisk"];
    auto disk_path = [&](char side, int i) { return envdisk + "/" + side + std::to_string(i) + ".env"; };
    // envdisk_async=1: stronger baseline - file writes run in background threads (GPU block freed right after the
    // device->host copy) and the next block the sweep will need is prefetched from disk while the current bond runs.
    const bool disk_async = arg["envdisk_async"] == "1";
    using HostBlock = std::pair<std::vector<int>, std::vector<double>>;
    std::map<std::string, std::future<void>> disk_writes;
    std::map<std::string, std::future<HostBlock>> disk_reads;
    auto disk_write_file = [](std::string path, std::vector<int> sh, std::vector<double> h) {
        std::ofstream f(path, std::ios::binary);
        int r = sh.size(); f.write((char*)&r, sizeof r); f.write((char*)sh.data(), r * sizeof(int));
        f.write((char*)h.data(), h.size() * sizeof(double));
        if (!f) throw std::runtime_error("envdisk write failed");
    };
    auto disk_read_file = [](std::string path) {
        std::ifstream f(path, std::ios::binary); int r = 0; f.read((char*)&r, sizeof r);
        std::vector<int> sh(r); f.read((char*)sh.data(), r * sizeof(int));
        size_t n = 1; for (int e : sh) n *= e;
        std::vector<double> h(n); f.read((char*)h.data(), n * sizeof(double));
        if (!f) throw std::runtime_error("envdisk read failed");
        return HostBlock(std::move(sh), std::move(h));
    };
    auto disk_put = [&](Tensor& t, char side, int i) {
        const std::string path = disk_path(side, i);
        std::vector<double> h = t.to_host(); std::vector<int> sh = t.shape;
        t = Tensor();
        if (!disk_async) { disk_write_file(path, sh, h); return; }
        if (disk_writes.count(path)) disk_writes[path].get();
        disk_reads.erase(path);
        disk_writes[path] = std::async(std::launch::async, disk_write_file, path, std::move(sh), std::move(h));
    };
    auto disk_prefetch = [&](char side, int i) {
        const std::string path = disk_path(side, i);
        auto w = disk_writes.find(path);
        if (w == disk_writes.end() || disk_reads.count(path)) return;   // not on disk in this pass, or already queued
        std::shared_future<void> done = w->second.share(); disk_writes.erase(w);
        disk_reads[path] = std::async(std::launch::async, [=]() { done.get(); return disk_read_file(path); });
    };
    auto disk_get = [&](Tensor& t, char side, int i) {
        const std::string path = disk_path(side, i);
        HostBlock b;
        if (disk_async) {
            disk_prefetch(side, i);
            auto r = disk_reads.find(path);
            if (r != disk_reads.end()) { b = r->second.get(); disk_reads.erase(r); }
            else b = disk_read_file(path);
            disk_prefetch(side, side == 'R' ? i + 1 : i - 1);            // the block the sweep needs next
        } else b = disk_read_file(path);
        t = Tensor(b.first); t.from_host(b.second);
    };
    // envpin=1 (with envck=k): strongest offload baseline - blocks that envck would drop are copied to pinned host
    // memory with cudaMemcpyAsync on a separate non-blocking stream (the copy engine runs while the GPU computes),
    // freed on that stream after the copy, and the next `envpin_prefetch` blocks the sweep will need are copied back
    // ahead of time; the compute stream only waits on an event when it actually uses a block. Host RAM ~ all blocks.
    const bool envpin = arg["envpin"] == "1";
    const int pin_pf = std::stoi(arg["envpin_prefetch"]);
    struct PinBlock { double* h = nullptr; size_t cap = 0; std::vector<int> shape; };
    std::map<std::pair<char, int>, PinBlock> pinned;
    std::map<std::pair<char, int>, std::pair<Tensor, cudaEvent_t>> inflight;   // prefetched device copies
    cudaStream_t pstream = nullptr;
    if (envpin) CUDA_CHECK(cudaStreamCreateWithFlags(&pstream, cudaStreamNonBlocking));
    auto new_event = []() { cudaEvent_t e; CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; };
    auto pin_put = [&](Tensor& t, char side, int i) {
        PinBlock& b = pinned[{side, i}];
        const size_t n = (size_t)t.size;
        if (b.cap < n) { if (b.h) CUDA_CHECK(cudaFreeHost(b.h)); CUDA_CHECK(cudaHostAlloc((void**)&b.h, n * sizeof(double), cudaHostAllocDefault)); b.cap = n; }
        b.shape = t.shape;
        cudaEvent_t ready = new_event();
        CUDA_CHECK(cudaEventRecord(ready, 0));                       // block finished on the compute stream
        CUDA_CHECK(cudaStreamWaitEvent(pstream, ready, 0));
        CUDA_CHECK(cudaMemcpyAsync(b.h, t.d_data, n * sizeof(double), cudaMemcpyDeviceToHost, pstream));
        CUDA_CHECK(cudaFreeAsync(t.d_data, pstream));               // freed after the copy, without stalling compute
        t.d_data = nullptr; t.size = 0; t.shape.clear();
        CUDA_CHECK(cudaEventDestroy(ready));
    };
    auto pin_prefetch = [&](char side, int i) {
        auto key = std::make_pair(side, i);
        auto it = pinned.find(key);
        if (it == pinned.end() || it->second.shape.empty() || inflight.count(key)) return;
        Tensor d(it->second.shape);                                  // allocated in stream order on the compute stream
        cudaEvent_t alloc = new_event(), done = new_event();
        CUDA_CHECK(cudaEventRecord(alloc, 0));
        CUDA_CHECK(cudaStreamWaitEvent(pstream, alloc, 0));
        CUDA_CHECK(cudaMemcpyAsync(d.d_data, it->second.h, (size_t)d.size * sizeof(double), cudaMemcpyHostToDevice, pstream));
        CUDA_CHECK(cudaEventRecord(done, pstream));
        CUDA_CHECK(cudaEventDestroy(alloc));
        inflight.emplace(key, std::make_pair(std::move(d), done));
    };
    auto pin_get = [&](Tensor& t, char side, int i) {
        auto key = std::make_pair(side, i);
        pin_prefetch(side, i);                                       // no-op if already in flight
        auto it = inflight.find(key);
        CUDA_CHECK(cudaStreamWaitEvent(0, it->second.second, 0));    // compute waits only for this block
        CUDA_CHECK(cudaEventDestroy(it->second.second));
        t = std::move(it->second.first);
        inflight.erase(it);
        pinned[key].shape.clear();                                   // consumed in this pass
        for (int p = 1; p <= pin_pf; ++p) pin_prefetch(side, side == 'R' ? i + p : i - p);
    };
    auto ck_build_right = [&](Environment& env) {
        env.R[N] = Tensor({1,1,1}); env.R[N].from_host({1.0});
        for (int i = N - 1; i >= 0; --i) {
            env.update_right_site(psi, H, i);
            if (!ck_keep(i + 1)) { if (envpin) pin_put(env.R[i + 1], 'R', i + 1); else if (envdisk.size()) disk_put(env.R[i + 1], 'R', i + 1); else env.R[i + 1] = Tensor(); }
        }
    };
    auto ck_build_left = [&](Environment& env) {
        env.L[0] = Tensor({1,1,1}); env.L[0].from_host({1.0});
        for (int i = 0; i < N; ++i) {
            env.update_left_site(psi, H, i);
            if (!ck_keep(i)) { if (envpin) pin_put(env.L[i], 'L', i); else if (envdisk.size()) disk_put(env.L[i], 'L', i); else env.L[i] = Tensor(); }
        }
    };
    auto ck_need_R = [&](Environment& env, int j) {   // ensure R[j]; rebuild from the next stored block on the right
        if (env.R[j].d_data) return;
        if (envpin) { pin_get(env.R[j], 'R', j); return; }
        if (envdisk.size()) { disk_get(env.R[j], 'R', j); return; }
        int c = j + 1; while (!env.R[c].d_data) ++c;
        for (int i = c - 1; i >= j; --i) {
            env.update_right_site(psi, H, i);
            if (envck2 > 0 && i + 1 < c && (i + 1) % envck2) env.R[i + 1] = Tensor();   // inner level: keep every k2-th
        }
    };
    auto ck_need_L = [&](Environment& env, int j) {
        if (env.L[j].d_data) return;
        if (envpin) { pin_get(env.L[j], 'L', j); return; }
        if (envdisk.size()) { disk_get(env.L[j], 'L', j); return; }
        int c = j - 1; while (!env.L[c].d_data) --c;
        for (int i = c; i < j; ++i) {
            env.update_left_site(psi, H, i);
            if (envck2 > 0 && i > c && i % envck2) env.L[i] = Tensor();
        }
    };
    std::unique_ptr<Environment> cenv;    // envcache: persistent environments across half-sweeps
    bool cache_R = false;                 // envcache: R blocks from the last R->L pass match psi
    // conv=1: convergence-driven schedule. sched entries D:n become stages with at most n sweeps; a stage ends when
    // |dE| per sweep < stage_tol (after >= conv_min sweeps). Kicks (heff_noise / lowp / ritz_T) stay on until, at
    // the final D, |dE| < kick_tol; then clean sweeps until |dE| < final_tol (>= conv_min clean sweeps).
    // While kicks are on, the energy used (and logged) is the exact <psi|H|psi>/<psi|psi> (its cost is counted).
    const bool conv = arg["conv"] == "1";
    std::vector<std::pair<int,int>> stages;
    for (int D : D_of_sweep) { if (stages.empty() || stages.back().first != D) stages.push_back({D, 0}); ++stages.back().second; }
    const double stage_tol = std::stod(arg["stage_tol"]), final_tol = std::stod(arg["final_tol"]), kick_tol = std::stod(arg["kick_tol"]);
    const int conv_min = std::stoi(arg["conv_min"]);
    int stage = 0, stage_cnt = 0, clean_cnt = 0;
    bool kicks = true, conv_done = false;
    double convE_prev = NAN, conv_best = INFINITY;
    int no_improve = 0;

    // envspec=1 (with evalonly): channel rank of the left environments. L[i] (D,D,W) viewed as M (D^2 x W);
    // Gram G = M^T M (W x W). Prints, at sites N/4, N/2, 3N/4, the eigenvalues of G relative to the largest and of the
    // normalised G_xy / sqrt(G_xx G_yy): how many MPO channels the environment really uses (compression feasibility).
    if (arg["envspec"] == "1") {
        left_canonicalize(psi, la);
        Environment ev(N); ev.build_left(psi, H);
        for (int s : {N / 4, N / 2, 3 * N / 4}) {
            const Tensor& Lt = ev.L[s];
            const int D1 = Lt.shape[0], D2 = Lt.shape[1], Wc = Lt.shape[2], R = D1 * D2;
            double* G; CUDA_CHECK(cudaMalloc(&G, sizeof(double) * Wc * Wc));
            const double one = 1.0, zero = 0.0;
            CUBLAS_CHECK(cublasDgemm(la.blas, CUBLAS_OP_T, CUBLAS_OP_N, Wc, Wc, R, &one, Lt.d_data, R, Lt.d_data, R, &zero, G, Wc));
            std::vector<double> g(Wc * Wc); CUDA_CHECK(cudaMemcpy(g.data(), G, sizeof(double) * Wc * Wc, cudaMemcpyDeviceToHost)); cudaFree(G);
            // symmetric eigenvalues via cuSOLVER-free Jacobi on host (W <= ~64)
            auto eig = [&](std::vector<double> a) {
                const int n = Wc; std::vector<double> ev2(n);
                for (int sweep = 0; sweep < 100; ++sweep) {
                    double off = 0; for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) off += a[p*n+q]*a[p*n+q];
                    if (off < 1e-30) break;
                    for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) {
                        if (std::fabs(a[p*n+q]) < 1e-300) continue;
                        double th = 0.5 * std::atan2(2 * a[p*n+q], a[q*n+q] - a[p*n+p]), c = std::cos(th), sn = std::sin(th);
                        for (int k = 0; k < n; ++k) { double akp = a[k*n+p], akq = a[k*n+q]; a[k*n+p] = c*akp - sn*akq; a[k*n+q] = sn*akp + c*akq; }
                        for (int k = 0; k < n; ++k) { double apk = a[p*n+k], aqk = a[q*n+k]; a[p*n+k] = c*apk - sn*aqk; a[q*n+k] = sn*apk + c*aqk; }
                    }
                }
                for (int p = 0; p < n; ++p) ev2[p] = a[p*n+p];
                std::sort(ev2.begin(), ev2.end(), std::greater<double>()); return ev2;
            };
            std::vector<double> gn(g); for (int x = 0; x < Wc; ++x) for (int y = 0; y < Wc; ++y)
                gn[x*Wc+y] = g[x*Wc+y] / std::sqrt(std::max(g[x*Wc+x], 1e-300) * std::max(g[y*Wc+y], 1e-300));
            for (int pass = 0; pass < 2; ++pass) {
                auto e = eig(pass ? gn : g); const double m = std::max(e[0], 1e-300);
                std::cerr << "ENVSPEC site=" << s << " D=" << D1 << " W=" << Wc << (pass ? " normalised" : " raw") << " rank>1e-6: "
                          << std::count_if(e.begin(), e.end(), [&](double v){ return v > 1e-6 * m; }) << " >1e-10: "
                          << std::count_if(e.begin(), e.end(), [&](double v){ return v > 1e-10 * m; }) << " >1e-14: "
                          << std::count_if(e.begin(), e.end(), [&](double v){ return v > 1e-14 * m; }) << "  spectrum:";
                for (double v : e) std::cerr << " " << std::setprecision(2) << v / m;
                std::cerr << "\n";
            }
        }
    }
    const bool evalonly = arg["evalonly"] == "1";   // evalonly=1: no sweeps, only the final exact energy (+ symdiag) of the loaded state
    for (int sw = st.sweeps_done; sw < (evalonly ? st.sweeps_done : n_sweeps + (conv ? 40 : 0)); ++sw) {   // conv: room for the clean phase
        if (last_sweep_s > 0 && secs(t_job, clk::now()) + 1.3 * last_sweep_s > max_s) {
            std::cerr << "wall-time budget: stopping before sweep " << sw + 1 << "\n";
            return 3;
        }
        psi.D = conv ? stages[stage].first : D_of_sweep[sw];
        if (arg["flow_final"] == "1")   // hybrid: flow truncation only at the final bond dimension (plain warm step before)
            wk.warm_eta = (conv ? stage + 1 == (int)stages.size() : psi.D == D_max) ? std::stod(arg["warm_eta"]) : 0.0;
        wk.heff_noise = sw < n_sweeps - std::stoi(arg["noise_clean"]) ? std::stod(arg["heff_noise"]) : 0.0;   // precision proxy
        wk.ritz_T = sw < n_sweeps - std::stoi(arg["ritz_clean"]) ? std::stod(arg["ritz_T"]) : 0.0;   // stochastic Ritz selection
        wk.heff_lowp = sw >= n_sweeps - std::stoi(arg["lowp_clean"]) || arg["lowp"] == "fp64" ? 0 : arg["lowp"] == "tf32" ? 1 : 2;
        if (conv) {   // kicks controlled by the convergence state, not by the sweep index
            wk.heff_noise = kicks ? std::stod(arg["heff_noise"]) : 0.0;
            wk.ritz_T = kicks ? std::stod(arg["ritz_T"]) : 0.0;
            wk.heff_lowp = !kicks || arg["lowp"] == "fp64" ? 0 : arg["lowp"] == "tf32" ? 1 : 2;
        }
        int max_chi = 0, n_heff = 0, n_solves = 0, n_unconv = 0;
        double max_resid = 0.0, dav_tol = 0.0;
        double max_disc = 0, sum_disc = 0, heff_ms = 0, dav_ms = 0, svd_ms = 0, env_s = 0, E = 0;
        size_t peak_used = 0;
        // memprof=1: split of the device memory at the per-sweep peak into MPS, stored environments and the rest
        const bool memprof = arg["memprof"] == "1";
        size_t pk_used = 0, pk_mps = 0, pk_env = 0;
        auto prof = [&](const Environment& env) {
            if (!memprof) return;
            size_t f, t; cudaMemGetInfo(&f, &t);
            if (t - f <= pk_used) return;
            size_t m = 0, e = 0;
            for (auto& a : psi.A) m += (size_t)a.size * sizeof(double);
            for (auto& x : env.L) e += (size_t)x.size * sizeof(double);
            for (auto& x : env.R) e += (size_t)x.size * sizeof(double);
            pk_used = t - f; pk_mps = m; pk_env = e;
        };
        auto track = [&](const TwoSiteUpdateStats& s) {
            if (bondlog) bondlog << sw + 1 << "," << n_solves << "," << s.residual_norm << "," << s.energy << "," << s.discarded_weight << "\n";
            ++n_solves; n_unconv += !s.davidson_converged;
            max_resid = std::max(max_resid, s.residual_norm); dav_tol = s.davidson_tol;
            E = s.energy; max_chi = std::max(max_chi, s.kept_chi);
            max_disc = std::max(max_disc, s.discarded_weight); sum_disc += s.discarded_weight;
            heff_ms += s.t_heff_ms; dav_ms += s.t_davidson_ms; svd_ms += s.t_svd_ms; n_heff += s.davidson_iters;
            if (!wk.production_mode || memtrack) { size_t f, t; cudaMemGetInfo(&f, &t); peak_used = std::max(peak_used, t - f); }
        };

        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = clk::now();
        // envcache=1: one persistent Environment. L->R keeps the L blocks it builds (still exact in the next R->L
        // pass: sites left of the bond are untouched there) and frees R; R->L keeps the R blocks for the next sweep
        // and frees L. Full rebuild only on the first sweep or after psi changed outside the sweep (accel).
        {
            auto te = clk::now();
            std::unique_ptr<Environment> tmp;
            if (envcache && !cenv) cenv = std::make_unique<Environment>(N);
            if (!envcache) tmp = std::make_unique<Environment>(N);
            Environment& env = envcache ? *cenv : *tmp;
            if (envck > 0) ck_build_right(env);
            else if (!envcache || !cache_R) env.build_right(psi, H);
            env.L[0] = Tensor({1,1,1}); env.L[0].from_host({1.0});
            if (!wk.production_mode) { CUDA_CHECK(cudaDeviceSynchronize()); env_s += secs(te, clk::now()); }
            for (int site = 0; site < N - 1; ++site) {
                dg.dir = 0;
                ds.dir = 0;
                if (envck > 0) ck_need_R(env, site + 2);
                track(two_site_update_timed(psi, H, env, la, wk, site, SweepDirection::LeftToRight));
                prof(env);
                env.update_left_site(psi, H, site);
                env.R[site + 2] = Tensor();                            // never read again this pass
                if (!envcache) env.L[site] = Tensor();                // halves peak env memory
            }
        }
        {
            auto te = clk::now();
            std::unique_ptr<Environment> tmp;
            if (!envcache) tmp = std::make_unique<Environment>(N);
            Environment& env = envcache ? *cenv : *tmp;
            if (envck > 0) ck_build_left(env);
            else if (!envcache) env.build_left(psi, H);
            env.R[N] = Tensor({1,1,1}); env.R[N].from_host({1.0});
            if (!wk.production_mode) { CUDA_CHECK(cudaDeviceSynchronize()); env_s += secs(te, clk::now()); }
            for (int site = N - 2; site >= 0; --site) {
                dg.dir = 1;
                ds.dir = 1;
                if (envck > 0) ck_need_L(env, site);
                track(two_site_update_timed(psi, H, env, la, wk, site, SweepDirection::RightToLeft));
                prof(env);
                env.update_right_site(psi, H, site + 1);
                env.L[site] = Tensor();                                // never read again this pass
                if (!envcache) env.R[site + 2] = Tensor();
            }
            cache_R = envcache;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        last_sweep_s = secs(t0, clk::now());
        if (memprof) std::cout << "MEMPROF sweep=" << sw + 1 << " peak_gb=" << pk_used / 1e9 << " mps_gb=" << pk_mps / 1e9
                               << " env_gb=" << pk_env / 1e9 << " other_gb=" << (double)(pk_used - pk_mps - pk_env) / 1e9 << "\n";
        if (wk.production_mode && !memtrack) { size_t f, t; cudaMemGetInfo(&f, &t); peak_used = t - f; }   // sampled once, after the sweep

        // accel=1 (feasibility probe for global acceleration): at the final D, extrapolate every tensor along the
        // last sweep's change, A(b) = A + b (A - A_prev) (bases stay aligned thanks to warm truncation), pick the
        // b in {0, 0.5, 1, 2} with the lowest exact <psi|H|psi>/<psi|psi>, re-canonicalise. Its cost is added to
        // this sweep's time so time-to-energy comparisons stay fair.
        if (accel && psi.D == D_max && sw + 1 < n_sweeps) {
            auto ta = clk::now();
            bool same = (int)prevA.size() == N;
            for (int i = 0; same && i < N; ++i) same = prevA[i].shape == psi.A[i].shape;
            if (same) {
                const double E0 = compute_total_energy(psi, H) / compute_norm(psi);
                double Eb = E0, bb = 0.0;
                std::vector<Tensor> bestA;
                for (double b : {0.5, 1.0, 2.0}) {
                    for (int i = 0; i < N; ++i) {
                        trial.A[i] = Tensor(psi.A[i].shape);
                        trial.A[i].copy_from(psi.A[i]);
                        const double s = 1.0 + b, mb = -b;
                        CUBLAS_CHECK(cublasDscal(la.blas, trial.A[i].size, &s, trial.A[i].d_data, 1));
                        CUBLAS_CHECK(cublasDaxpy(la.blas, trial.A[i].size, &mb, prevA[i].d_data, 1, trial.A[i].d_data, 1));
                    }
                    const double Et = compute_total_energy(trial, H) / compute_norm(trial);
                    if (Et < Eb - 1e-13) { Eb = Et; bb = b; bestA = std::move(trial.A); trial.A.resize(N); }
                }
                if (bb > 0) { psi.A = std::move(bestA); right_canonicalize(psi, la); E = Eb; cache_R = false; }
                std::cerr << std::setprecision(14) << "ACCEL sweep=" << sw + 1 << " E0=" << E0 << " best_b=" << bb
                          << " E=" << Eb << " gain=" << E0 - Eb << "\n";
            }
            prevA.resize(N);
            for (int i = 0; i < N; ++i) { prevA[i] = Tensor(psi.A[i].shape); prevA[i].copy_from(psi.A[i]); }
            CUDA_CHECK(cudaDeviceSynchronize());
            last_sweep_s += secs(ta, clk::now());
        }

        st.sweeps_done = sw + 1;
        st.energy = E;
        auto tc = clk::now();
        save_checkpoint(ckpt, psi, st);   // device->host copy + file write + rename (blocking)
        ckpt_s += secs(tc, clk::now());
        sweeps_s += last_sweep_s;

        if (conv) {
            const bool kicked = wk.heff_noise > 0 || wk.heff_lowp || wk.ritz_T > 0;
            if (kicked) {   // perturbed Ritz values are not trustworthy: use (and log) the exact energy, count its cost
                auto tx = clk::now();
                E = compute_total_energy(psi, H) / compute_norm(psi);
                CUDA_CHECK(cudaDeviceSynchronize());
                last_sweep_s += secs(tx, clk::now());
            }
            const double dEc = std::isnan(convE_prev) ? INFINITY : std::fabs(E - convE_prev);
            convE_prev = E;
            ++stage_cnt;
            const bool last = stage + 1 == (int)stages.size();
            std::string ev;
            if (!last) {
                if ((stage_cnt >= conv_min && dEc < stage_tol) || stage_cnt >= stages[stage].second) {
                    ev = "advance D " + std::to_string(stages[stage].first) + " -> " + std::to_string(stages[stage + 1].first);
                    ++stage; stage_cnt = 0; convE_prev = NAN;
                }
            } else if (kicks && wk.heff_noise == 0 && wk.heff_lowp == 0 && wk.ritz_T == 0) {
                kicks = false;             // nothing to switch off: go straight to the clean phase
            }
            // kicked energies jitter at a noise floor, so |dE| alone may never drop below kick_tol: also switch off
            // on a plateau (no new lowest energy for 2 consecutive sweeps at the final D)
            if (last && kicks) {
                if (E < conv_best - kick_tol) { conv_best = E; no_improve = 0; } else ++no_improve;
            }
            if (last && kicks && ((stage_cnt >= conv_min && (dEc < kick_tol || no_improve >= 2)) || stage_cnt >= stages[stage].second)) {
                kicks = false; clean_cnt = 0; convE_prev = NAN; ev = "kicks off";
            } else if (last && !kicks && !ev.size()) {
                ++clean_cnt;
                if ((clean_cnt >= conv_min && dEc < final_tol) || stage_cnt >= stages[stage].second + 30) {
                    conv_done = true; ev = "converged";
                }
            }
            std::cerr << std::setprecision(14) << "CONV sweep=" << sw + 1 << " D=" << psi.D << " E=" << E << " |dE|="
                      << dEc << (ev.size() ? " -> " + ev : std::string()) << "\n";
        }
        double dE = (sw == 0 && !restarted) ? NAN : E - E_prev;
        E_prev = E;
        csv << model << "," << Lx << "," << Ly << "," << N << "," << W << "," << J2 << "," << arg["heff"] << ","
            << sw + 1 << "," << psi.D << "," << max_chi << "," << E << "," << dE << "," << max_disc << "," << sum_disc << ","
            << last_sweep_s << "," << heff_ms / 1e3 << "," << heff_ms / 1e3 / last_sweep_s << ","
            << dav_ms / 1e3 << "," << svd_ms / 1e3 << "," << env_s << "," << n_heff << "," << peak_used / 1e9 << ","
            << n_solves << "," << n_unconv << "," << max_resid << "," << dav_tol << "\n";
        csv.flush();
        if (diag) dg.report(sw + 1, wk);
        if (diis) ds.report(sw + 1);
        std::cerr << "sweep " << sw + 1 << "/" << n_sweeps << " D=" << psi.D << " chi=" << max_chi
                  << " E=" << std::setprecision(12) << E << " disc=" << max_disc << " t=" << last_sweep_s << "s"
                  << " heff_frac=" << heff_ms / 1e3 / last_sweep_s << " mem=" << peak_used / 1e9 << "GB"
                  << " davidson: " << n_unconv << "/" << n_solves << " above tol " << dav_tol
                  << ", max_resid=" << max_resid << "\n";
        if (conv_done) break;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto tf = clk::now();
    cenv.reset(); wk.cutensor_plans.clear(); wk.lowp_plans.clear(); prevA.clear();   // free sweep memory before the final check (H100 OOM)
    double Emps = compute_total_energy(psi, H);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double energy_s = secs(tf, clk::now());
    tf = clk::now();
    double nrm = compute_norm(psi);
    if (arg["symdiag"] == "1") {
        const double nn = flip_overlap(psi, la.blas, false), ff = flip_overlap(psi, la.blas, true);
        std::cerr << std::setprecision(14) << "SYMDIAG <psi|psi>=" << nn << " (compute_norm " << nrm << ")"
                  << " <psi|F|psi>/<psi|psi>=" << ff / nn << "\n";
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double norm_s = secs(tf, clk::now());
    std::cerr << "final: davidson_E=" << std::setprecision(14) << st.energy << " <psi|H|psi>/<psi|psi>=" << Emps / nrm
              << " norm=" << nrm << "\n";
    double total_s = secs(t_start, clk::now());
    std::cerr << "total_run_s=" << total_s << "\n";
    if (arg["svd"] == "warm") std::cerr << "warm_hits=" << wk.warm_hits << " warm_fallbacks=" << wk.warm_fallbacks << "\n";
    // .timing: wall-time breakdown of the whole run; other_s = per-sweep CSV/log I/O, budget checks, teardown
    std::ofstream(ckpt + ".timing") << std::setprecision(8)
        << "init_s " << init_s << "\nsweeps_s " << sweeps_s << "\ncheckpoint_s " << ckpt_s
        << "\nfinal_energy_s " << energy_s << "\nfinal_norm_s " << norm_s << "\ntotal_s " << total_s
        << "\nother_s " << total_s - init_s - sweeps_s - ckpt_s - energy_s - norm_s << "\n";
    // .done: davidson_E  <psi|H|psi>/<psi|psi>  norm  total_run_s
    std::ofstream(ckpt + ".done") << std::setprecision(14) << st.energy << " " << Emps / nrm << " " << nrm << " " << total_s << "\n";
    return 0;
}
