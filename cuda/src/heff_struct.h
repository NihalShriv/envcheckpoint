#pragma once
#include "tensor.h"
#include <cublas_v2.h>
#include <map>
#include <utility>

// MPO-structure-aware two-site Heff (layout-specialised, sparsity-exploiting), same FLOP-optimal order as the
// generic contraction but with the MPO step done only over nonzero blocks:
//   T1[a,d2,s,r,x]   = sum_b  L[a,b,x] theta[b,r,s,d2]                 one strided-batched GEMM over x
//   T3[q,p,a,d2,z]   = sum_{x,r,s} Weff[x,z,p,q,r,s] T1[a,d2,s,r,x]    fused kernel over nonzeros of
//                      Weff = sum_y W1[x,y,p,r] W2[y,z,q,s]  (precomputed once per (W1,W2) pair)
//   out[c,q,p,a]     = sum_{d2,z} R[c,d2,z] T3[q,p,a,d2,z]              one GEMM, output already in final layout
// No permutation kernels: T1/T3 layouts are chosen so both GEMMs read/write contiguous operands.
// Note: the nonzero list is cached by the W1/W2 device pointers; MPO values must not change in place.
class HeffStruct {
public:
    HeffStruct(int Dl, int d, int Dr, int Wl, int Wr);
    ~HeffStruct();
    HeffStruct(const HeffStruct&) = delete;
    HeffStruct& operator=(const HeffStruct&) = delete;

    void apply(Tensor& out, const Tensor& theta_in, const Tensor& L,
               const Tensor& W1, const Tensor& W2, const Tensor& R, cublasHandle_t blas);
    double flops() const { return flops_; }   // valid after the first apply()
    int nnz() const { return nnz_; }

private:
    struct Sparse { int* off = nullptr; int* idx = nullptr; double* val = nullptr; int nnz = 0; };
    const Sparse& sparse_for(const Tensor& W1, const Tensor& W2);

    int Dl_, d_, Dr_, Wl_, Wr_;
    double *T1_ = nullptr, *T3_ = nullptr;
    std::map<std::pair<const double*, const double*>, Sparse> cache_;
    double flops_ = 0;
    int nnz_ = 0;
};
