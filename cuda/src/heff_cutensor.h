#pragma once
#include "tensor.h"
#include <cutensor.h>
#include <vector>

// Generic (MPO-structure-unaware) two-site Heff via cuTENSOR 2.x, standard DMRG contraction order:
//   T1[a,x,r,s,d2] = L[a,b,x]     * theta[b,r,s,d2]
//   T2[a,y,p,s,d2] = T1           * W1[x,y,p,r]
//   T3[a,p,z,q,d2] = T2           * W2[y,z,q,s]
//   out[a,p,q,c]   = T3           * R[c,d2,z]
// Plans and intermediates are built once per shape (constructor); apply() only launches contractions.
// envck tree: heff_set_chunks(c > 1) splits the first two contractions over chunks of the MPO index x and the
// last two over chunks of z (T1, T3 held only per chunk, T2 accumulated over x chunks, out over z chunks).
// Same flops, ~2/3 less intermediate memory for large c; results differ from c = 1 only by summation order.
void heff_set_chunks(int c);

class HeffCutensor {
public:
    HeffCutensor(int Dl, int d, int Dr, int Wl, int Wm, int Wr);
    ~HeffCutensor();
    HeffCutensor(const HeffCutensor&) = delete;
    HeffCutensor& operator=(const HeffCutensor&) = delete;

    void apply(Tensor& out, const Tensor& theta_in, const Tensor& L,
               const Tensor& W1, const Tensor& W2, const Tensor& R);
    double flops() const { return flops_; }

private:
    cutensorHandle_t handle_ = nullptr;
    cutensorPlan_t plan_[4] = {};
    cutensorOperationDescriptor_t op_[4] = {};
    cutensorTensorDescriptor_t desc_[9] = {};
    double *T1_ = nullptr, *T2_ = nullptr, *T3_ = nullptr;
    void* work_ = nullptr;
    uint64_t work_size_ = 0;
    double flops_ = 0;
    // chunked mode
    struct CPlan { cutensorPlan_t plan = nullptr; cutensorOperationDescriptor_t op = nullptr; uint64_t ws = 0; };
    int chunks_ = 1, Dl_, d_, Dr_, Wl_, Wm_, Wr_;
    std::vector<CPlan> cplans_;            // [2][4]: {full chunk, last chunk} x op 0..3
    std::vector<cutensorTensorDescriptor_t> cdescs_;
    CPlan make(const std::vector<int32_t>& mA, const std::vector<int64_t>& eA, const std::vector<int64_t>& sA,
               const std::vector<int32_t>& mB, const std::vector<int64_t>& eB, const std::vector<int64_t>& sB,
               const std::vector<int32_t>& mD, const std::vector<int64_t>& eD);
    void apply_chunked(Tensor& out, const Tensor& theta_in, const Tensor& L,
                       const Tensor& W1, const Tensor& W2, const Tensor& R);
};
