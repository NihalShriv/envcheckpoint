#pragma once
#include "mps.h"
#include "linalg.h"

void left_canonicalize(MPS& psi, LinAlg& la);
void right_canonicalize(MPS& psi, LinAlg& la);