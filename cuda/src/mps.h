#pragma once
#include <vector>
#include "tensor.h"

class MPS {
public:
    int N;   // number of sites
    int d;   // physical dimension
    int D;   // bond dimension

    std::vector<Tensor> A; // tensors (Dl, d, Dr)

    MPS(int N_, int d_, int D_);
    void random_initialize();
};