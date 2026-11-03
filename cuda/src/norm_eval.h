#pragma once
#include "mps.h"

double compute_norm(MPS& psi);
// Reference O(D^4) implementation, testing only.
double compute_norm_naive(MPS& psi);
