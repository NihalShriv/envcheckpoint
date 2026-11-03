#include "energy_eval.h"

double compute_total_energy(
    MPS& psi,
    MPO& H)
{
    int N = psi.N;

    // envck tree: stream the left environments (keep only the current block) so the final energy needs O(1)
    // environment blocks instead of N. Same contractions as build_left.
    Environment env(N);
    env.L[0] = Tensor({1,1,1});
    env.L[0].from_host({1.0});
    for (int i = 0; i < N; i++) { env.update_left_site(psi, H, i); env.L[i] = Tensor(); }

    return env.L[N].item();
}
