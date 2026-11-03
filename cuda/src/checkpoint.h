#pragma once
#include "mps.h"
#include <string>

// Binary MPS checkpoint. Written to <path>.tmp then renamed, so a crash mid-write never corrupts <path>.
// Environments are not stored: they are rebuilt from the MPS on restart (one O(N D^3 W) pass).
struct RunState {
    int    sweeps_done = 0;   // completed full (L->R + R->L) sweeps
    double energy      = 0.0; // energy after the last completed sweep
};

void save_checkpoint(const std::string& path, const MPS& psi, const RunState& st);
// Returns false if <path> does not exist. Throws on a corrupt / mismatched file.
bool load_checkpoint(const std::string& path, MPS& psi, RunState& st);
