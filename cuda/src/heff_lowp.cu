#include "heff_lowp.h"
#include <vector>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <string>

#define CUTENSOR_CHECK(expr) do { cutensorStatus_t s_ = (expr); if (s_ != CUTENSOR_STATUS_SUCCESS) \
    throw std::runtime_error(std::string("cuTENSOR error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                             " for " #expr ": " + cutensorGetErrorString(s_)); } while (0)

namespace {
enum : int32_t { A='a', B='b', C='c', D2='d', P='p', Q='q', Rr='r', S='s', X='x', Y='y', Z='z' };
__global__ void d2f(float* o, const double* i, size_t n)
{ size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (k < n) o[k] = (float)i[k]; }
__global__ void f2d(double* o, const float* i, size_t n)
{ size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (k < n) o[k] = (double)i[k]; }
void to_f(float* o, const double* i, size_t n) { d2f<<<(unsigned)((n + 255) / 256), 256>>>(o, i, n); }
}

// Same modes and contraction order as HeffCutensor (see heff_cutensor.h).
HeffLowp::HeffLowp(int Dl, int d, int Dr, int Wl, int Wm, int Wr, bool tf32)
{
    std::map<int32_t, int64_t> ext = { {A,Dl},{B,Dl},{C,Dr},{D2,Dr},{P,d},{Q,d},{Rr,d},{S,d},{X,Wl},{Y,Wm},{Z,Wr} };
    const std::vector<std::vector<int32_t>> modes = {
        {A, B, X}, {D2, S, Rr, B}, {A, X, Rr, S, D2}, {X, Y, P, Rr}, {A, Y, P, S, D2},
        {Y, Z, Q, S}, {A, P, Z, Q, D2}, {C, D2, Z}, {C, Q, P, A} };
    CUTENSOR_CHECK(cutensorCreate(&handle_));
    for (int i = 0; i < 9; ++i) {
        std::vector<int64_t> e; size_t n = 1;
        for (int32_t m : modes[i]) { e.push_back(ext[m]); n *= ext[m]; }
        n_[i] = n;
        CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle_, &desc_[i], (uint32_t)e.size(), e.data(),
                                                      nullptr, CUTENSOR_R_32F, 128));
        CUDA_CHECK(cudaMalloc(&f_[i], n * sizeof(float)));
    }
    const int ops[4][3] = { {0,1,2}, {2,3,4}, {4,5,6}, {6,7,8} };
    cutensorPlanPreference_t pref;
    CUTENSOR_CHECK(cutensorCreatePlanPreference(handle_, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    for (int k = 0; k < 4; ++k) {
        int a = ops[k][0], b = ops[k][1], c = ops[k][2];
        CUTENSOR_CHECK(cutensorCreateContraction(handle_, &op_[k],
            desc_[a], modes[a].data(), CUTENSOR_OP_IDENTITY,
            desc_[b], modes[b].data(), CUTENSOR_OP_IDENTITY,
            desc_[c], modes[c].data(), CUTENSOR_OP_IDENTITY,
            desc_[c], modes[c].data(), tf32 ? CUTENSOR_COMPUTE_DESC_TF32 : CUTENSOR_COMPUTE_DESC_32F));
        uint64_t ws = 0;
        CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(handle_, op_[k], pref, CUTENSOR_WORKSPACE_DEFAULT, &ws));
        CUTENSOR_CHECK(cutensorCreatePlan(handle_, &plan_[k], op_[k], pref, ws));
        work_size_ = std::max(work_size_, ws);
    }
    CUTENSOR_CHECK(cutensorDestroyPlanPreference(pref));
    if (work_size_) CUDA_CHECK(cudaMalloc(&work_, work_size_));
}

HeffLowp::~HeffLowp()
{
    for (auto p : plan_) if (p) cutensorDestroyPlan(p);
    for (auto o : op_) if (o) cutensorDestroyOperationDescriptor(o);
    for (auto d : desc_) if (d) cutensorDestroyTensorDescriptor(d);
    for (auto f : f_) cudaFree(f);
    cudaFree(work_);
    if (handle_) cutensorDestroy(handle_);
}

void HeffLowp::apply(double* out, const double* theta_in, const Tensor& L,
                     const Tensor& W1, const Tensor& W2, const Tensor& R)
{
    const float one = 1.0f, zero = 0.0f;
    to_f(f_[0], L.d_data, n_[0]); to_f(f_[1], theta_in, n_[1]);
    to_f(f_[3], W1.d_data, n_[3]); to_f(f_[5], W2.d_data, n_[5]); to_f(f_[7], R.d_data, n_[7]);
    CUTENSOR_CHECK(cutensorContract(handle_, plan_[0], &one, f_[0], f_[1], &zero, f_[2], f_[2], work_, work_size_, 0));
    CUTENSOR_CHECK(cutensorContract(handle_, plan_[1], &one, f_[2], f_[3], &zero, f_[4], f_[4], work_, work_size_, 0));
    CUTENSOR_CHECK(cutensorContract(handle_, plan_[2], &one, f_[4], f_[5], &zero, f_[6], f_[6], work_, work_size_, 0));
    CUTENSOR_CHECK(cutensorContract(handle_, plan_[3], &one, f_[6], f_[7], &zero, f_[8], f_[8], work_, work_size_, 0));
    f2d<<<(unsigned)((n_[8] + 255) / 256), 256>>>(out, f_[8], n_[8]);
}
