#pragma once
#include <cstdint>
#include <vector>

// D[mD] = A[mA] * B[mB] (double, beta = 0) via cuTENSOR 2.x on the default stream.
// Modes are single chars, listed fastest-varying first; extents in the same order.
// Plans are cached per (modes, extents); workspace grows on demand.
void ct_contract(const std::vector<int32_t>& mA, const std::vector<int64_t>& eA, const double* A,
                 const std::vector<int32_t>& mB, const std::vector<int64_t>& eB, const double* B,
                 const std::vector<int32_t>& mD, const std::vector<int64_t>& eD, double* D);

// Device scratch buffer `slot` with at least n doubles (grow-only, reused across calls).
double* ct_scratch(int slot, size_t n);
