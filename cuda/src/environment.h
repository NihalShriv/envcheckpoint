#pragma once
#include "mps.h"
#include "mpo.h"

class Environment {
public:
    std::vector<Tensor> L;
    std::vector<Tensor> R;
    bool naive = false;   // reference O(D^4) kernels (testing only); default = cuTENSOR O(D^3 W)

    Environment(int N);

    void build_left(const MPS& psi,
                    const MPO& H);
    void update_left_site(const MPS& psi,
                          const MPO& H,
                          int site);

    void build_right(const MPS& psi,
                     const MPO& H);
    void update_right_site(const MPS& psi,
                           const MPO& H,
                           int site);

    bool is_well_conditioned(double max_abs_threshold) const;
};