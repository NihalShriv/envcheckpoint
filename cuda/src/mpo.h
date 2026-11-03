#pragma once
#include <vector>
#include "tensor.h"
#include <string>

namespace layout {
// Global tensor layout helpers used by MPO, environments, and Heff apply.
inline __host__ __device__ int mpo_idx(int wl, int wr, int p, int q, int Wl, int Wr, int d)
{
    return wl + Wl * (wr + Wr * (p + d * q));
}

inline __host__ __device__ int env_idx(int left, int right, int mpo, int left_dim, int right_dim)
{
    return left + left_dim * (right + right_dim * mpo);
}

inline __host__ __device__ int mps_idx(int left, int phys, int right, int Dl, int d)
{
    return left + Dl * (phys + d * right);
}
}

class MPO {
public:
    int N;
    int d;
    int Wdim;

    std::vector<Tensor> W; // tensors (Wl, Wr, d, d)

    MPO(int N_, int d_, int Wdim_);

    void initialize_ising(double Jzz, double h);
};
void initialize_tri_ladder(MPO& H, double J);

// ---- Generic spin-1/2 Heisenberg MPO from an explicit bond list: H = sum_b J_b S_i . S_j ----
// Channels: 0 = start, 1 + 3*(t-1) + op = operator op (Sz, S+, S-) emitted t sites ago, W-1 = done.
// The step-t channel entering site j carries the operator from exactly site j-t, so the closing
// coefficient J(j-t, j) is exact for any bond list (no lattice-specific gating needed).
struct Bond { int i, j; double J; };
int  heisenberg_mpo_width(const std::vector<Bond>& bonds);      // 2 + 3 * max|i-j|
void initialize_heisenberg_bonds(MPO& H, const std::vector<Bond>& bonds);

// Lx x Ly square-lattice J1-J2 cylinder (open in x, periodic in y if Ly >= 3).
// MPS order: column-major snake-free, site = x*Ly + y.
std::vector<Bond> j1j2_cylinder_bonds(int Lx, int Ly, double J1, double J2);
// 2-leg triangular ladder, same geometry as initialize_tri_ladder (site 2k = top, 2k+1 = bottom).
std::vector<Bond> tri_ladder_bonds(int Lx, double J);

// Exact SVD compression of an MPO, treating (p,q) as the physical leg: a left-to-right then right-to-left
// sweep of SVDs on every bond, dropping singular values below rel_tol * (largest at that bond).
// Leaves the operator unchanged to rounding (rel_tol = 1e-12 removes only linearly dependent channels);
// per-site bond dimensions shrink (e.g. YC4 J1-J2: 23 -> 14). Returns the new maximum bond dimension.
int compress_mpo(MPO& H, double rel_tol = 1e-12);

// Bond file (vacancy lattices etc.):  lines "N <n>", "site <k> <x> <y>" (MPS order k = 0..n-1, original
// lattice coordinates kept for distances), "bond <i> <j> <J>"; '#' starts a comment.
struct SiteXY { double x = 0, y = 0; };
std::vector<Bond> read_bond_file(const std::string& path, int& N, std::vector<SiteXY>& xy);
