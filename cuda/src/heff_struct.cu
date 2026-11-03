#include "heff_struct.h"
#include "mpo.h"
#include <vector>

namespace {
// T3[q,p,a,d2,z] = sum over nonzeros (x,r,s,c) of slice (z,p,q):  c * T1[a,d2,s,r,x]
// blockIdx.y = output slice (q fastest, then p, then z); threads run over e = a + Dl*d2 (coalesced T1 reads).
__global__ void weff_apply(double* T3, const double* T1, const int* off, const int* idx, const double* val,
                           int Dl, int Dr, int d)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int DD = Dl * Dr;
    if (e >= DD) return;
    int slice = blockIdx.y;               // = q + d*(p + d*z)
    double sum = 0.0;
    for (int k = off[slice]; k < off[slice + 1]; ++k)
        sum += val[k] * T1[e + (size_t)DD * idx[k]];   // idx = s + d*(r + d*x)
    int q = slice % d, p = (slice / d) % d, z = slice / (d * d);
    int a = e % Dl, d2 = e / Dl;
    T3[q + d * (p + d * (a + (size_t)Dl * (d2 + (size_t)Dr * z)))] = sum;
}
}

HeffStruct::HeffStruct(int Dl, int d, int Dr, int Wl, int Wr) : Dl_(Dl), d_(d), Dr_(Dr), Wl_(Wl), Wr_(Wr)
{
    CUDA_CHECK(cudaMalloc(&T1_, sizeof(double) * (size_t)Dl * Dr * d * d * Wl));
    CUDA_CHECK(cudaMalloc(&T3_, sizeof(double) * (size_t)d * d * Dl * Dr * Wr));
}

HeffStruct::~HeffStruct()
{
    cudaFree(T1_); cudaFree(T3_);
    for (auto& kv : cache_) { cudaFree(kv.second.off); cudaFree(kv.second.idx); cudaFree(kv.second.val); }
}

const HeffStruct::Sparse& HeffStruct::sparse_for(const Tensor& W1, const Tensor& W2)
{
    auto key = std::make_pair((const double*)W1.d_data, (const double*)W2.d_data);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;

    const int d = d_, Wl = W1.shape[0], Wm = W1.shape[1], Wr = W2.shape[1];
    auto w1 = W1.to_host(), w2 = W2.to_host();
    // Weff[x,z,p,q,r,s] = sum_y W1[x,y,p,r] W2[y,z,q,s]; group nonzeros by output slice (z,p,q)
    int n_slices = Wr * d * d;
    std::vector<std::vector<std::pair<int,double>>> by_slice(n_slices);
    for (int x = 0; x < Wl; ++x) for (int z = 0; z < Wr; ++z)
        for (int p = 0; p < d; ++p) for (int q = 0; q < d; ++q)
            for (int r = 0; r < d; ++r) for (int s = 0; s < d; ++s) {
                double c = 0.0;
                for (int y = 0; y < Wm; ++y)
                    c += w1[layout::mpo_idx(x, y, p, r, Wl, Wm, d)] * w2[layout::mpo_idx(y, z, q, s, Wm, Wr, d)];
                if (c != 0.0) by_slice[q + d * (p + d * z)].push_back({s + d * (r + d * x), c});
            }
    std::vector<int> off(n_slices + 1, 0), idx;
    std::vector<double> val;
    for (int sl = 0; sl < n_slices; ++sl) {
        for (auto& e : by_slice[sl]) { idx.push_back(e.first); val.push_back(e.second); }
        off[sl + 1] = idx.size();
    }
    Sparse sp;
    sp.nnz = idx.size();
    CUDA_CHECK(cudaMalloc(&sp.off, sizeof(int) * off.size()));
    CUDA_CHECK(cudaMalloc(&sp.idx, sizeof(int) * std::max<size_t>(1, idx.size())));
    CUDA_CHECK(cudaMalloc(&sp.val, sizeof(double) * std::max<size_t>(1, val.size())));
    CUDA_CHECK(cudaMemcpy(sp.off, off.data(), sizeof(int) * off.size(), cudaMemcpyHostToDevice));
    if (sp.nnz) {
        CUDA_CHECK(cudaMemcpy(sp.idx, idx.data(), sizeof(int) * idx.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sp.val, val.data(), sizeof(double) * val.size(), cudaMemcpyHostToDevice));
    }
    return cache_.emplace(key, sp).first->second;
}

void HeffStruct::apply(Tensor& out, const Tensor& theta_in, const Tensor& L,
                       const Tensor& W1, const Tensor& W2, const Tensor& R, cublasHandle_t blas)
{
    const int Dl = Dl_, Dr = Dr_, d = d_, Wl = Wl_, Wr = Wr_;
    const Sparse& sp = sparse_for(W1, W2);
    const double one = 1.0, zero = 0.0;
    const int K = Dr * d * d;   // theta viewed column-major as (d2,s,r) x b

    // T1_x (a x K) = L_x (a x b) * theta^T (b x K), batched over x
    CUBLAS_CHECK(cublasDgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_T, Dl, K, Dl,
        &one, L.d_data, Dl, (long long)Dl * Dl, theta_in.d_data, K, 0LL,
        &zero, T1_, Dl, (long long)Dl * K, Wl));

    dim3 grid((Dl * Dr + 255) / 256, Wr * d * d);
    CUDA_KERNEL_CHECK((weff_apply<<<grid, 256>>>(T3_, T1_, sp.off, sp.idx, sp.val, Dl, Dr, d)));

    // out (c x (q,p,a)) = R (c x (d2,z)) * T3^T ((d2,z) x (q,p,a))
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, Dr, d * d * Dl, Dr * Wr,
        &one, R.d_data, Dr, T3_, d * d * Dl, &zero, out.d_data, Dr));

    nnz_ = sp.nnz;
    flops_ = 2.0 * Dl * K * Dl * Wl + 2.0 * sp.nnz * Dl * Dr + 2.0 * Dr * (double)d * d * Dl * Dr * Wr;
}
