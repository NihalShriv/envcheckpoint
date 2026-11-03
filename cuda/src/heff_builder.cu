#include "heff_builder.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
__global__ void apply_two_site_heff_kernel(
    double* theta_out,
    const double* theta_in,
    const double* L,
    const double* R,
    const double* W1,
    const double* W2,
    int Dl,
    int d,
    int Dr,
    int W1l,
    int W1r,
    int W2r,
    int Ld0,
    int Ld1,
    int Rd0,
    int Rd1,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dim)
        return;

    int a = idx / (d * d * Dr);
    int tmp = idx % (d * d * Dr);
    int p = tmp / (d * Dr);
    tmp %= (d * Dr);
    int q = tmp / Dr;
    int c = tmp % Dr;

    double sum = 0.0;

    for(int b = 0; b < Dl; ++b)
    for(int r = 0; r < d; ++r)
    for(int s = 0; s < d; ++s)
    for(int d2 = 0; d2 < Dr; ++d2)
    {
        double theta = theta_in[((b * d + r) * d + s) * Dr + d2];
        double local_sum = 0.0;

        for(int x = 0; x < W1l; ++x)
        for(int y = 0; y < W1r; ++y)
        for(int z = 0; z < W2r; ++z)
        {
            double Lval = L[layout::env_idx(a, b, x, Ld0, Ld1)];
            double Rval = R[layout::env_idx(c, d2, z, Rd0, Rd1)];
            double W1val = W1[layout::mpo_idx(x, y, p, r, W1l, W1r, d)];
            double W2val = W2[layout::mpo_idx(y, z, q, s, W1r, W2r, d)];
            local_sum += Lval * W1val * W2val * Rval;
        }

        sum += local_sum * theta;
    }

    theta_out[idx] = sum;
}
} // namespace

void apply_two_site_heff(
    Tensor& theta_out,
    const Tensor& theta_in,
    const MPS& psi,
    const MPO& H,
    const Environment& env,
    int site)
{
    const Tensor& A1 = psi.A[site];
    const Tensor& A2 = psi.A[site+1];

    const Tensor& W1 = H.W[site];
    const Tensor& W2 = H.W[site+1];

    const Tensor& L = env.L[site];
    const Tensor& R = env.R[site+2];

    int Dl = A1.shape[0];
    int d  = psi.d;
    int Dr = A2.shape[2];

    int dim = Dl * d * d * Dr;
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;

    apply_two_site_heff_kernel<<<blocks,threads>>>(
        theta_out.d_data,
        theta_in.d_data,
        L.d_data,
        R.d_data,
        W1.d_data,
        W2.d_data,
        Dl,
        d,
        Dr,
        W1.shape[0],
        W1.shape[1],
        W2.shape[1],
        L.shape[0],
        L.shape[1],
        R.shape[0],
        R.shape[1],
        dim);
}

// -----------------------------------------------------------------------
// cuBLAS GEMM-based Heff
// -----------------------------------------------------------------------
namespace {

// ML_y[a*d+p, b*d+r] (column-major) = Σ_x L[a,b,x] * W1[x,y,p,r]
// Each thread handles one (ap, br) element.
__global__ void build_ML_y(double* ML, const double* L, const double* W1,
                            int y, int Dl, int d, int W1l, int W1r)
{
    int ap = blockIdx.x * blockDim.x + threadIdx.x;
    int br = blockIdx.y * blockDim.y + threadIdx.y;
    int left_dim = Dl * d;
    if(ap >= left_dim || br >= left_dim) return;
    int a = ap / d, p = ap % d;
    int b = br / d, r = br % d;
    double sum = 0.0;
    for(int x = 0; x < W1l; ++x)
        sum += L[a + Dl * (b + Dl * x)]
             * W1[x + W1l * (y + W1r * (p + d * r))];
    ML[ap + left_dim * br] = sum;
}

// MR_y[q*Dr+c, s*Dr+d2] (column-major) = Σ_z W2[y,z,q,s] * R[c,d2,z]
__global__ void build_MR_y(double* MR, const double* W2, const double* R,
                            int y, int Dr, int d, int W1r, int W2r)
{
    int qc  = blockIdx.x * blockDim.x + threadIdx.x;
    int sd2 = blockIdx.y * blockDim.y + threadIdx.y;
    int right_dim = d * Dr;
    if(qc >= right_dim || sd2 >= right_dim) return;
    int q = qc / Dr, c = qc % Dr;
    int s = sd2 / Dr, d2 = sd2 % Dr;
    double sum = 0.0;
    for(int z = 0; z < W2r; ++z)
        sum += W2[y + W1r * (z + W2r * (q + d * s))]
             * R[c + Dr * (d2 + Dr * z)];
    MR[qc + right_dim * sd2] = sum;
}

} // anonymous namespace

void apply_two_site_heff_gemm(
    Tensor& theta_out,
    const Tensor& theta_in,
    const MPS& psi,
    const MPO& H,
    const Environment& env,
    int site,
    cublasHandle_t blas,
    double* ML_buf,
    double* MR_buf,
    double* tmp_buf)
{
    const Tensor& W1 = H.W[site];
    const Tensor& W2 = H.W[site + 1];
    const Tensor& L  = env.L[site];
    const Tensor& R  = env.R[site + 2];

    int Dl        = psi.A[site].shape[0];
    int d         = psi.d;
    int Dr        = psi.A[site + 1].shape[2];
    int W1l       = W1.shape[0];
    int W1r       = W1.shape[1];   // = W2.shape[0], loop bound for y
    int W2r       = W2.shape[1];
    int left_dim  = Dl * d;
    int right_dim = d * Dr;

    // T_in/T_out stored row-major (left_dim × right_dim) = col-major (right_dim × left_dim).
    // Accumulate into T_out; zero first so beta=1 works for all y iterations.
    cudaMemset(theta_out.d_data, 0, sizeof(double) * left_dim * right_dim);

    const double one  = 1.0;
    const double zero = 0.0;

    dim3 threads2d(16, 16);
    dim3 ml_blocks((left_dim  + 15) / 16, (left_dim  + 15) / 16);
    dim3 mr_blocks((right_dim + 15) / 16, (right_dim + 15) / 16);

    for(int y = 0; y < W1r; ++y)
    {
        build_ML_y<<<ml_blocks, threads2d>>>(
            ML_buf, L.d_data, W1.d_data, y, Dl, d, W1l, W1r);
        build_MR_y<<<mr_blocks, threads2d>>>(
            MR_buf, W2.d_data, R.d_data, y, Dr, d, W1r, W2r);

        // tmp_cm (right_dim × left_dim) = MR_y (right_dim × right_dim) @ T_in_cm (right_dim × left_dim)
        cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    right_dim, left_dim, right_dim,
                    &one,
                    MR_buf,          right_dim,
                    theta_in.d_data, right_dim,
                    &zero,
                    tmp_buf,         right_dim);

        // T_out_cm (right_dim × left_dim) += tmp_cm @ ML_y^T (left_dim × left_dim)
        cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T,
                    right_dim, left_dim, left_dim,
                    &one,
                    tmp_buf,          right_dim,
                    ML_buf,           left_dim,
                    &one,
                    theta_out.d_data, right_dim);
    }
}

bool validate_two_site_heff_apply(
    const MPS& psi,
    const MPO& H,
    const Environment& env,
    int site,
    double tolerance)
{
    const Tensor& A1 = psi.A[site];
    const Tensor& A2 = psi.A[site+1];
    int Dl = A1.shape[0];
    int d = psi.d;
    int Dm = A1.shape[2];
    int Dr = A2.shape[2];
    int dim = Dl * d * d * Dr;

    if(dim > 4096)
        return true;

    std::vector<double> hA1 = A1.to_host();
    std::vector<double> hA2 = A2.to_host();
    std::vector<double> hL = env.L[site].to_host();
    std::vector<double> hR = env.R[site+2].to_host();
    std::vector<double> hW1 = H.W[site].to_host();
    std::vector<double> hW2 = H.W[site+1].to_host();

    std::vector<double> hTheta(dim, 0.0);
    for(int a = 0; a < Dl; ++a)
    for(int p = 0; p < d; ++p)
    for(int q = 0; q < d; ++q)
    for(int c = 0; c < Dr; ++c)
    {
        double sum = 0.0;
        for(int m = 0; m < Dm; ++m)
        {
            double left = hA1[layout::mps_idx(a, p, m, Dl, d)];
            double right = hA2[layout::mps_idx(m, q, c, Dm, d)];
            sum += left * right;
        }
        hTheta[((a * d + p) * d + q) * Dr + c] = sum;
    }

    Tensor theta_in({dim});
    Tensor theta_out({dim});
    theta_in.from_host(hTheta);
    apply_two_site_heff(theta_out, theta_in, psi, H, env, site);
    auto gpu = theta_out.to_host();

    int W1l = H.W[site].shape[0];
    int W1r = H.W[site].shape[1];
    int W2r = H.W[site+1].shape[1];
    int Ld0 = env.L[site].shape[0];
    int Ld1 = env.L[site].shape[1];
    int Rd0 = env.R[site+2].shape[0];
    int Rd1 = env.R[site+2].shape[1];

    for(int a = 0; a < Dl; ++a)
    for(int p = 0; p < d; ++p)
    for(int q = 0; q < d; ++q)
    for(int c = 0; c < Dr; ++c)
    {
        int out_idx = ((a * d + p) * d + q) * Dr + c;
        double reference = 0.0;
        for(int b = 0; b < Dl; ++b)
        for(int r = 0; r < d; ++r)
        for(int s = 0; s < d; ++s)
        for(int d2 = 0; d2 < Dr; ++d2)
        {
            double theta = hTheta[((b * d + r) * d + s) * Dr + d2];
            double local_sum = 0.0;
            for(int x = 0; x < W1l; ++x)
            for(int y = 0; y < W1r; ++y)
            for(int z = 0; z < W2r; ++z)
            {
                double Lval = hL[layout::env_idx(a, b, x, Ld0, Ld1)];
                double Rval = hR[layout::env_idx(c, d2, z, Rd0, Rd1)];
                double W1val = hW1[layout::mpo_idx(x, y, p, r, W1l, W1r, d)];
                double W2val = hW2[layout::mpo_idx(y, z, q, s, W1r, W2r, d)];
                local_sum += Lval * W1val * W2val * Rval;
            }
            reference += local_sum * theta;
        }
        if(std::fabs(reference - gpu[out_idx]) > tolerance)
        {
            std::cout << "Two-site apply mismatch on site " << site
                      << ": ref=" << reference
                      << " gpu=" << gpu[out_idx] << std::endl;
            return false;
        }
    }

    std::vector<double> x(dim), y(dim);
    for(int i = 0; i < dim; ++i)
    {
        x[i] = std::sin(0.37 * (i + 1));
        y[i] = std::cos(0.19 * (i + 1));
    }

    Tensor x_dev({dim});
    Tensor y_dev({dim});
    Tensor hx_dev({dim});
    Tensor hy_dev({dim});
    x_dev.from_host(x);
    y_dev.from_host(y);
    apply_two_site_heff(hx_dev, x_dev, psi, H, env, site);
    apply_two_site_heff(hy_dev, y_dev, psi, H, env, site);
    auto hx = hx_dev.to_host();
    auto hy = hy_dev.to_host();

    double xHy = 0.0;
    double yHx = 0.0;
    for(int i = 0; i < dim; ++i)
    {
        xHy += x[i] * hy[i];
        yHx += y[i] * hx[i];
    }
    if(std::fabs(xHy - yHx) > 100.0 * tolerance)
    {
        std::cout << "Hermiticity check failed on site " << site
                  << ": <x|Hy>=" << xHy
                  << " <y|Hx>=" << yHx << std::endl;
        return false;
    }

    return true;
}
