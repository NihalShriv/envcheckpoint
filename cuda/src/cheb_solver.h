#pragma once
#include "two_site_split_timed.h"

// Sync-light local eigensolver for two-site DMRG: s-step Krylov in a Chebyshev basis + Rayleigh-Ritz.
//
// Per local solve (and per restart):
//   x_0 = theta (warm start), x_1 = (H x_0 - c x_0)/e, x_{k+1} = 2(H x_k - c x_k)/e - x_{k-1}   (k < m)
//   i.e. Chebyshev polynomials of H mapped from [lo, hi] to [-1, 1]. span{x_0..x_m} = Krylov K_{m+1}(H, x_0),
//   so Rayleigh-Ritz on it is as accurate as (m+1)-vector Lanczos, but needs NO inner products while building
//   the basis (only axpy/scal) -> no device-to-host syncs inside the loop.
//   Then: G = X^T X, P = X^T HX (two GEMMs), ONE device->host copy, tiny generalised eigenproblem on the host
//   (canonical orthogonalisation drops near-dependent directions), ground = X c, exact residual ||HXc - th Xc||.
//   H_eff applications per solve: m + 1.
// Spectral interval: hi = upper bound of the global spectrum (supplied, e.g. sum|J|/4 for Heisenberg; H_eff
// eigenvalues lie inside the global spectrum), lo = previous Ritz value minus a margin. The interval only
// affects the conditioning of the basis, not which subspace is spanned.
struct ChebParams {
    int    degree   = 3;       // m: basis size m+1, H_eff applications m+1 per restart
    int    restarts = 1;       // outer restarts (ground of one restart seeds the next)
    double hi       = 0.0;     // spectral upper bound (required, > 0)
    double lo_margin = 0.05;   // lo = E_prev - lo_margin * (hi - E_prev)
    double tol      = 1e-12;   // residual tolerance, for the convergence flag only (fixed work, no early exit)
};

// Returns a SweepWorkspace::local_solver implementing the method above (keeps E_prev between calls).
decltype(SweepWorkspace::local_solver) make_cheb_solver(const ChebParams& p);
