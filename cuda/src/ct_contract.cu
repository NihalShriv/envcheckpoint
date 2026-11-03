#include "ct_contract.h"
#include "tensor.h"
#include <cutensor.h>
#include <map>
#include <string>

#define CT_CHECK(expr) do { cutensorStatus_t s_ = (expr); if (s_ != CUTENSOR_STATUS_SUCCESS) \
    throw std::runtime_error(std::string("cuTENSOR error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                             " for " #expr ": " + cutensorGetErrorString(s_)); } while (0)

namespace {
struct Plan { cutensorPlan_t plan; uint64_t ws; };

cutensorHandle_t handle() {
    static cutensorHandle_t h = [] { cutensorHandle_t x; CT_CHECK(cutensorCreate(&x)); return x; }();
    return h;
}

std::string key(const std::vector<int32_t>& m, const std::vector<int64_t>& e) {
    std::string k;
    for (size_t i = 0; i < m.size(); ++i) k += char(m[i]) + std::to_string(e[i]) + ",";
    return k + "|";
}

cutensorTensorDescriptor_t desc(const std::vector<int64_t>& e) {
    cutensorTensorDescriptor_t d;
    CT_CHECK(cutensorCreateTensorDescriptor(handle(), &d, (uint32_t)e.size(), e.data(), nullptr, CUTENSOR_R_64F, 128));
    return d;
}

// ponytail: plans and descriptors live for the whole process (shapes repeat once D saturates)
std::map<std::string, Plan>& plans() { static std::map<std::string, Plan> p; return p; }

void* workspace(uint64_t n) {
    static void* w = nullptr; static uint64_t cap = 0;
    if (n > cap) { if (w) cudaFree(w); CUDA_CHECK(cudaMalloc(&w, n)); cap = n; }
    return w;
}
}

double* ct_scratch(int slot, size_t n)
{
    static std::map<int, std::pair<double*, size_t>> bufs;
    auto& b = bufs[slot];
    if (n > b.second) { if (b.first) cudaFree(b.first); CUDA_CHECK(cudaMalloc(&b.first, n * sizeof(double))); b.second = n; }
    return b.first;
}

void ct_contract(const std::vector<int32_t>& mA, const std::vector<int64_t>& eA, const double* A,
                 const std::vector<int32_t>& mB, const std::vector<int64_t>& eB, const double* B,
                 const std::vector<int32_t>& mD, const std::vector<int64_t>& eD, double* D)
{
    std::string k = key(mA, eA) + key(mB, eB) + key(mD, eD);
    auto it = plans().find(k);
    if (it == plans().end()) {
        cutensorTensorDescriptor_t dA = desc(eA), dB = desc(eB), dD = desc(eD);
        cutensorOperationDescriptor_t op;
        CT_CHECK(cutensorCreateContraction(handle(), &op, dA, mA.data(), CUTENSOR_OP_IDENTITY,
                                           dB, mB.data(), CUTENSOR_OP_IDENTITY,
                                           dD, mD.data(), CUTENSOR_OP_IDENTITY,
                                           dD, mD.data(), CUTENSOR_COMPUTE_DESC_64F));
        cutensorPlanPreference_t pref;
        CT_CHECK(cutensorCreatePlanPreference(handle(), &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
        Plan p{};
        CT_CHECK(cutensorEstimateWorkspaceSize(handle(), op, pref, CUTENSOR_WORKSPACE_DEFAULT, &p.ws));
        CT_CHECK(cutensorCreatePlan(handle(), &p.plan, op, pref, p.ws));
        CT_CHECK(cutensorDestroyPlanPreference(pref));
        it = plans().emplace(k, p).first;
    }
    const double one = 1.0, zero = 0.0;
    CT_CHECK(cutensorContract(handle(), it->second.plan, &one, A, B, &zero, D, D,
                              workspace(it->second.ws), it->second.ws, 0));
}
