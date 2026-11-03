#pragma once
#include "mps.h"
#include "mpo.h"
#include "environment.h"
#include <cublas_v2.h>

void apply_two_site_heff(
    Tensor& theta_out,
    const Tensor& theta_in,
    const MPS& psi,
    const MPO& H,
    const Environment& env,
    int site
);

// cuBLAS GEMM-based Heff: O(W·D^3·d^2) vs the original O(D^4·W^3).
// ML_buf/MR_buf/tmp_buf must each be pre-allocated to at least (D*d)^2 doubles.
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
    double* tmp_buf
);

bool validate_two_site_heff_apply(
    const MPS& psi,
    const MPO& H,
    const Environment& env,
    int site,
    double tolerance
);