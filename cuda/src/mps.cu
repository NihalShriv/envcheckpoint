#include "mps.h"
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace {
int capped_bond_dim(int left_sites, int right_sites, int local_dim, int max_dim)
{
    int steps = std::min(left_sites, right_sites);
    int dim = 1;

    for(int i = 0; i < steps; i++)
    {
        if(dim >= max_dim)
            return max_dim;

        if(dim > max_dim / local_dim)
            return max_dim;

        dim *= local_dim;
    }

    return std::min(dim, max_dim);
}
} // namespace

MPS::MPS(int N_, int d_, int D_)
    : N(N_), d(d_), D(D_)
{
    A.reserve(N);

    for(int i=0;i<N;i++)
    {
        int Dl = capped_bond_dim(i, N-i, d, D);
        int Dr = capped_bond_dim(i+1, N-i-1, d, D);

        A.emplace_back(std::vector<int>{Dl,d,Dr});
    }
}

void MPS::random_initialize()
{
    for(int i=0;i<N;i++)
    {
        int size = A[i].size;
        std::vector<double> h(size);

        double norm = 0.0f;

        for(int j=0;j<size;j++)
        {
            h[j] = static_cast<double>(rand())/RAND_MAX;
            norm += h[j]*h[j];
        }

        norm = std::sqrt(norm);

        for(int j=0;j<size;j++)
            h[j] /= norm;

        A[i].from_host(h);
    }
}