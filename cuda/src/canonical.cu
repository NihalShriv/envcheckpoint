#include "canonical.h"
#include <algorithm>
#include <cmath>

namespace {
__global__ void transpose_matrix(
    double* out,
    const double* in,
    int rows,
    int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= rows * cols)
        return;

    int row = idx % rows;
    int col = idx / rows;
    out[col + cols * row] = in[row + rows * col];
}

void transpose_tensor(Tensor& out, const Tensor& in, int rows, int cols)
{
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    transpose_matrix<<<blocks,threads>>>(out.d_data, in.d_data, rows, cols);
}

void normalize_tensor_if_needed(Tensor& A, LinAlg& la)
{
    double norm = la.nrm2(A);
    if(norm > 1e-12 && std::isfinite(norm))
        la.scal(A, 1.0 / norm);
}
} // namespace

void left_canonicalize(MPS& psi, LinAlg& la)
{
    for(int site = 0; site < psi.N - 1; ++site)
    {
        Tensor& A = psi.A[site];

        int Dl = A.shape[0];
        int d  = A.shape[1];
        int Dr = A.shape[2];
        int m = Dl * d;
        int n = Dr;

        Tensor M({m, n});
        M.copy_from(A);

        Tensor Q({m, n});
        Tensor R({n, n});
        la.qr(M, Q, R, m, n);

        psi.A[site] = std::move(Q);
        psi.A[site].reshape({Dl, d, n});

        Tensor& next = psi.A[site + 1];
        int d2  = next.shape[1];
        int Dr2 = next.shape[2];

        Tensor next_mat({n, d2 * Dr2});
        next_mat.copy_from(next);

        Tensor updated({n, d2 * Dr2});
        la.gemm(R, next_mat, updated, n, n, d2 * Dr2);

        psi.A[site + 1] = std::move(updated);
        psi.A[site + 1].reshape({n, d2, Dr2});
    }

    normalize_tensor_if_needed(psi.A[psi.N - 1], la);
}

void right_canonicalize(MPS& psi, LinAlg& la)
{
    for(int site = psi.N - 1; site > 0; --site)
    {
        Tensor& A = psi.A[site];

        int Dl = A.shape[0];
        int d  = A.shape[1];
        int Dr = A.shape[2];
        int m = Dl;
        int n = d * Dr;

        Tensor AT({n, m});
        transpose_tensor(AT, A, m, n);

        Tensor QT({n, m});
        Tensor RT({m, m});
        la.qr(AT, QT, RT, n, m);

        Tensor Q({m, n});
        transpose_tensor(Q, QT, n, m);

        psi.A[site] = std::move(Q);
        psi.A[site].reshape({m, d, Dr});

        Tensor LT({m, m});
        transpose_tensor(LT, RT, m, m);

        Tensor& prev = psi.A[site - 1];
        int Dl2 = prev.shape[0];
        int d2  = prev.shape[1];
        int Dr2 = prev.shape[2];

        Tensor prev_mat({Dl2 * d2, Dr2});
        prev_mat.copy_from(prev);

        Tensor updated({Dl2 * d2, m});
        la.gemm(prev_mat, LT, updated, Dl2 * d2, Dr2, m);

        psi.A[site - 1] = std::move(updated);
        psi.A[site - 1].reshape({Dl2, d2, m});
    }

    normalize_tensor_if_needed(psi.A[0], la);
}
