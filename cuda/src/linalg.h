#pragma once
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "tensor.h"

class LinAlg {
public:
    cublasHandle_t blas;
    cusolverDnHandle_t solver;

    // Pre-allocated solver workspace — activated by calling prewarm_svd / prewarm_syevd.
    // When set, svd() and syevd() reuse these buffers instead of malloc/free per call.
    // Existing code that never calls prewarm continues to work identically.
    int*    d_info     = nullptr;
    double* svd_work   = nullptr;
    int     svd_lwork  = 0;
    double* eig_work   = nullptr;
    int     eig_lwork  = 0;

    LinAlg();
    ~LinAlg();

    // Pre-allocate workspace for the largest matrix sizes you will encounter.
    // Safe to call once before the sweep loop; idempotent if called again with
    // the same or smaller sizes.
    void prewarm_svd(int m, int n);
    void prewarm_syevd(int dim);

    void syevd(Tensor& A, Tensor& eigvals, int dim);
    void svd(Tensor& A, Tensor& U, Tensor& S, Tensor& V,
             int m, int n);
    void gemm(const Tensor& A,
              const Tensor& B,
              Tensor& C,
              int m, int k, int n);
    void qr(Tensor& A, Tensor& Q, Tensor& R,
            int m, int n);
    double nrm2(const Tensor& A);
    double dot(const Tensor& A, const Tensor& B);
    void scal(Tensor& A, double alpha);
    void axpy(double alpha, const Tensor& X, Tensor& Y);
    void copy(const Tensor& X, Tensor& Y);
};