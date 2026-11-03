#pragma once

#include "mps.h"
#include "mpo.h"
#include "environment.h"

double compute_total_energy(
    MPS& psi,
    MPO& H
);
