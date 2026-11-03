#pragma once
#include "tensor.h"
#include <cutensor.h>

// Reduced-precision two-site Heff (experimental): the same four contractions as HeffCutensor, on float copies of
// the operands, with compute type TF32 (tensor cores, ~10-bit mantissa products, FP32 accumulate) or plain FP32.
// Inputs/outputs stay double: operands are converted on the fly each call (cheap next to the contractions).
class HeffLowp {
public:
    HeffLowp(int Dl, int d, int Dr, int Wl, int Wm, int Wr, bool tf32);
    ~HeffLowp();
    HeffLowp(const HeffLowp&) = delete;
    HeffLowp& operator=(const HeffLowp&) = delete;

    void apply(double* out, const double* theta_in, const Tensor& L,
               const Tensor& W1, const Tensor& W2, const Tensor& R);

private:
    cutensorHandle_t handle_ = nullptr;
    cutensorPlan_t plan_[4] = {};
    cutensorOperationDescriptor_t op_[4] = {};
    cutensorTensorDescriptor_t desc_[9] = {};
    size_t n_[9] = {};                                   // element counts of the 9 operands
    float* f_[9] = {};                                   // float buffers (inputs, intermediates, output)
    void* work_ = nullptr;
    uint64_t work_size_ = 0;
};
