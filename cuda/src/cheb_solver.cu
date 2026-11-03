#include "cheb_solver.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {
// Cyclic Jacobi eigensolver for a small symmetric n x n matrix (column-major). Eigenvalues ascending,
// eigenvectors in the columns of V.
void jacobi_eig(int n, std::vector<double> A, std::vector<double>& w, std::vector<double>& V)
{
    V.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) V[i + n * i] = 1.0;
    auto at = [&](int i, int j) -> double& { return A[i + (size_t)n * j]; };
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0, tot = 0;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { tot += at(i, j) * at(i, j); if (i != j) off += at(i, j) * at(i, j); }
        if (off <= 1e-30 * tot || off == 0.0) break;
        for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) {
            double apq = at(p, q);
            if (std::fabs(apq) < 1e-300) continue;
            double theta = (at(q, q) - at(p, p)) / (2 * apq);
            double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
            double c = 1 / std::sqrt(t * t + 1), s = t * c;
            for (int k = 0; k < n; ++k) {          // A <- A J
                double akp = at(k, p), akq = at(k, q);
                at(k, p) = c * akp - s * akq; at(k, q) = s * akp + c * akq;
            }
            for (int k = 0; k < n; ++k) {          // A <- J^T A
                double apk = at(p, k), aqk = at(q, k);
                at(p, k) = c * apk - s * aqk; at(q, k) = s * apk + c * aqk;
            }
            for (int k = 0; k < n; ++k) {          // V <- V J
                double vkp = V[k + n * p], vkq = V[k + n * q];
                V[k + n * p] = c * vkp - s * vkq; V[k + n * q] = s * vkp + c * vkq;
            }
        }
    }
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return at(a, a) < at(b, b); });
    std::vector<double> V2((size_t)n * n);
    w.resize(n);
    for (int j = 0; j < n; ++j) {
        w[j] = at(idx[j], idx[j]);
        for (int i = 0; i < n; ++i) V2[i + n * j] = V[i + n * idx[j]];
    }
    V.swap(V2);
}

struct ChebState {
    ChebParams p;
    double E_prev = NAN;
    double* d_small = nullptr;      // [G | P] (2 n^2) then c (n)
    ~ChebState() { if (d_small) cudaFree(d_small); }
};
}

decltype(SweepWorkspace::local_solver) make_cheb_solver(const ChebParams& params)
{
    auto st = std::make_shared<ChebState>();
    st->p = params;
    const int n = params.degree + 1;
    CUDA_CHECK(cudaMalloc(&st->d_small, sizeof(double) * (2 * n * n + n)));

    return [st, n](SweepWorkspace& wk, const MPS& psi, const MPO& H, const Environment& env, LinAlg& la,
                   int site, double& residual, double& energy, int& matvecs, bool& converged, double& tol) -> bool {
        const ChebParams& p = st->p;
        const int Dl = psi.A[site].shape[0], d = psi.d, Dr = psi.A[site + 1].shape[2];
        const int dim = Dl * d * d * Dr;
        double* X  = wk.basis_mat;   // dim x n, column k = x_k
        double* HX = wk.image_mat;   // dim x n, column k = H x_k
        auto col  = [&](double* M, int k) { return M + (size_t)dim * k; };
        double* dG = st->d_small;
        double* dP = st->d_small + n * n;
        double* dc = st->d_small + 2 * n * n;
        const double one = 1.0, zero = 0.0;

        sweep_merge_theta(col(X, 0), psi, site);   // warm start
        double theta = 0.0, res = INFINITY;
        matvecs = 0;
        for (int r = 0; r < p.restarts; ++r) {
            if (r > 0) CUDA_CHECK(cudaMemcpyAsync(col(X, 0), wk.ground, sizeof(double) * dim, cudaMemcpyDeviceToDevice));
            const double hi = p.hi;
            const double lo = std::isnan(st->E_prev) ? -3.0 * p.hi : st->E_prev - p.lo_margin * (hi - st->E_prev);
            const double c = 0.5 * (hi + lo), e = 0.5 * (hi - lo);

            // Chebyshev-basis Krylov vectors: only H_eff applications + axpy/scal, no inner products
            for (int k = 0; k < n; ++k) {
                sweep_apply_heff(wk, col(HX, k), col(X, k), psi, H, env, la, site);
                ++matvecs;
                if (k + 1 == n) break;
                double* xn = col(X, k + 1);
                CUDA_CHECK(cudaMemcpyAsync(xn, col(HX, k), sizeof(double) * dim, cudaMemcpyDeviceToDevice));
                const double a = (k == 0 ? 1.0 : 2.0) / e, b = -a * c, m1 = -1.0;
                CUBLAS_CHECK(cublasDscal(la.blas, dim, &a, xn, 1));
                CUBLAS_CHECK(cublasDaxpy(la.blas, dim, &b, col(X, k), 1, xn, 1));
                if (k > 0) CUBLAS_CHECK(cublasDaxpy(la.blas, dim, &m1, col(X, k - 1), 1, xn, 1));
            }

            // projected problem: G = X^T X, P = X^T H X  -> one host copy (the only sync before the residual)
            CUBLAS_CHECK(cublasDgemm(la.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, dim, &one, X, dim, X, dim, &zero, dG, n));
            CUBLAS_CHECK(cublasDgemm(la.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, dim, &one, X, dim, HX, dim, &zero, dP, n));
            std::vector<double> GP(2 * n * n);
            CUDA_CHECK(cudaMemcpy(GP.data(), dG, sizeof(double) * 2 * n * n, cudaMemcpyDeviceToHost));
            std::vector<double> G(GP.begin(), GP.begin() + n * n), P(GP.begin() + n * n, GP.end());
            for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
                double s = 0.5 * (P[i + n * j] + P[j + n * i]); P[i + n * j] = P[j + n * i] = s;
            }

            // canonical orthogonalisation: keep directions with Gram eigenvalue > 1e-13 * max
            std::vector<double> gs, U;
            jacobi_eig(n, G, gs, U);
            double gmax = gs.back();
            std::vector<int> keep;
            for (int i = 0; i < n; ++i) if (gs[i] > 1e-13 * gmax) keep.push_back(i);
            const int k = keep.size();
            std::vector<double> T((size_t)n * k);            // T = U_keep s^{-1/2}
            for (int j = 0; j < k; ++j) for (int i = 0; i < n; ++i)
                T[i + n * j] = U[i + n * keep[j]] / std::sqrt(gs[keep[j]]);
            std::vector<double> Pt((size_t)k * k, 0.0);       // Pt = T^T P T
            for (int a2 = 0; a2 < k; ++a2) for (int b2 = 0; b2 < k; ++b2) {
                double s = 0;
                for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) s += T[i + n * a2] * P[i + n * j] * T[j + n * b2];
                Pt[a2 + k * b2] = s;
            }
            std::vector<double> pw, Y;
            jacobi_eig(k, Pt, pw, Y);
            theta = pw[0];
            std::vector<double> cvec(n, 0.0);                 // c = T y0  (so c^T G c = 1)
            for (int i = 0; i < n; ++i) for (int j = 0; j < k; ++j) cvec[i] += T[i + n * j] * Y[j];

            // ground = X c (normalized by construction); residual = || HX c - theta X c ||
            CUDA_CHECK(cudaMemcpyAsync(dc, cvec.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
            CUBLAS_CHECK(cublasDgemv(la.blas, CUBLAS_OP_N, dim, n, &one, X, dim, dc, 1, &zero, wk.ground, 1));
            CUBLAS_CHECK(cublasDgemv(la.blas, CUBLAS_OP_N, dim, n, &one, HX, dim, dc, 1, &zero, wk.residual_v, 1));
            const double mt = -theta;
            CUBLAS_CHECK(cublasDaxpy(la.blas, dim, &mt, wk.ground, 1, wk.residual_v, 1));
            CUBLAS_CHECK(cublasDnrm2(la.blas, dim, wk.residual_v, 1, &res));
            st->E_prev = theta;
        }
        residual = res;
        energy = theta;
        tol = p.tol;
        converged = res <= p.tol;
        return std::isfinite(theta) && std::isfinite(res);
    };
}
