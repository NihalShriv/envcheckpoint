#include "mpo.h"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cusolverDn.h>
#include <fstream>
#include <sstream>

MPO::MPO(int N_, int d_, int Wdim_)
    : N(N_), d(d_), Wdim(Wdim_)
{
    W.reserve(N);

    for(int i = 0; i < N; ++i)
    {
        int Wl = (i == 0) ? 1 : Wdim;
        int Wr = (i == N - 1) ? 1 : Wdim;
        W.emplace_back(std::vector<int>{Wl, Wr, d, d});
    }
}

void MPO::initialize_ising(double Jzz, double h)
{
    // OBC TFIM MPO with bond dimension 3.
    // States: 0=start, 1=Z-emitted (waiting to close ZZ), 2=done.
    // Every site including the last must apply -h*X on the start channel (0->0/2).

    for(int site = 0; site < N; ++site)
    {
        int Wl = W[site].shape[0];
        int Wr = W[site].shape[1];
        std::vector<double> hW(Wl * Wr * d * d, 0.0);

        auto z_value = [&](int p) {
            return (p == 0) ? 1.0 : -1.0;
        };

        auto x_value = [&](int p, int q) {
            return (p != q) ? 1.0 : 0.0;
        };

        if(site == 0)
        {
            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(0, 0, p, p, Wl, Wr, d)] = 1.0;

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(0, 1, p, p, Wl, Wr, d)] = z_value(p);

            for(int p = 0; p < d; ++p)
                for(int q = 0; q < d; ++q)
                    hW[layout::mpo_idx(0, 2, p, q, Wl, Wr, d)] = -h * x_value(p, q);
        }
        else if(site == N - 1)
        {
            // field on last site: channel 0 closes with -h*X
            for(int p = 0; p < d; ++p)
                for(int q = 0; q < d; ++q)
                    hW[layout::mpo_idx(0, 0, p, q, Wl, Wr, d)] = -h * x_value(p, q);

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(1, 0, p, p, Wl, Wr, d)] = Jzz * z_value(p);

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(2, 0, p, p, Wl, Wr, d)] = 1.0;
        }
        else
        {
            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(0, 0, p, p, Wl, Wr, d)] = 1.0;

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(0, 1, p, p, Wl, Wr, d)] = z_value(p);

            for(int p = 0; p < d; ++p)
                for(int q = 0; q < d; ++q)
                    hW[layout::mpo_idx(0, 2, p, q, Wl, Wr, d)] = -h * x_value(p, q);

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(1, 2, p, p, Wl, Wr, d)] = Jzz * z_value(p);

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(2, 2, p, p, Wl, Wr, d)] = 1.0;
        }

        W[site].from_host(hW);
    }
}

// 2-leg isotropic triangular Heisenberg ladder, W=11. Site 2k=top, 2k+1=bottom of rung k.
// Bonds: rung/anti-diag (dist 1), legs (dist 2), diagonal (2k,2k+3) (dist 3, even origin only).
void initialize_tri_ladder(MPO& H, double J)
{
    const int d = 2;
    for (int site = 0; site < H.N; ++site) {
        int Wl = H.W[site].shape[0];
        int Wr = H.W[site].shape[1];
        std::vector<double> hW(Wl * Wr * d * d, 0.0);
        auto at  = [&](int wl, int wr, int p, int q) -> double& {
            return hW[layout::mpo_idx(wl, wr, p, q, Wl, Wr, d)];
        };
        auto setI  = [&](int wl, int wr){ at(wl,wr,0,0)+=1.; at(wl,wr,1,1)+=1.; };
        auto setSz = [&](int wl, int wr, double c){ at(wl,wr,0,0)+=.5*c; at(wl,wr,1,1)-=.5*c; };
        auto setSp = [&](int wl, int wr, double c){ at(wl,wr,0,1)+=c; };
        auto setSm = [&](int wl, int wr, double c){ at(wl,wr,1,0)+=c; };

        if (site == 0) {
            setI(0,0);
            setSz(0,1,1.); setSp(0,2,1.); setSm(0,3,1.);
        } else if (site == H.N - 1) {
            setSz(1,0,J);   setSm(2,0,J*.5); setSp(3,0,J*.5);
            setSz(4,0,J);   setSm(5,0,J*.5); setSp(6,0,J*.5);
            setSz(7,0,J);   setSm(8,0,J*.5); setSp(9,0,J*.5);
            setI(10,0);
        } else {
            setI(0,0);
            setSz(0,1,1.); setSp(0,2,1.); setSm(0,3,1.);
            setSz(1,10,J); setSm(2,10,J*.5); setSp(3,10,J*.5);
            setI(1,4); setI(2,5); setI(3,6);
            setSz(4,10,J); setSm(5,10,J*.5); setSp(6,10,J*.5);
            // (2k+1, 2k+4) is not a ladder bond: 3NN propagation only on even sites
            if (site % 2 == 0) { setI(4,7); setI(5,8); setI(6,9); }
            setSz(7,10,J); setSm(8,10,J*.5); setSp(9,10,J*.5);
            setI(10,10);
        }
        H.W[site].from_host(hW);
    }
}

int heisenberg_mpo_width(const std::vector<Bond>& bonds)
{
    int kmax = 1;
    for (const auto& b : bonds) kmax = std::max(kmax, std::abs(b.j - b.i));
    return 2 + 3 * kmax;
}

void initialize_heisenberg_bonds(MPO& H, const std::vector<Bond>& bonds)
{
    const int d = 2, N = H.N, W = H.Wdim;
    const int kmax = (W - 2) / 3, done = W - 1;
    if (W != heisenberg_mpo_width(bonds) || H.d != 2)
        throw std::runtime_error("initialize_heisenberg_bonds: MPO width must equal heisenberg_mpo_width(bonds)");

    // J[(i,j)] with i<j, duplicates accumulated
    std::map<std::pair<int,int>, double> J;
    for (const auto& b : bonds) {
        if (b.i == b.j || b.i < 0 || b.j < 0 || b.i >= N || b.j >= N)
            throw std::runtime_error("initialize_heisenberg_bonds: bad bond");
        J[{std::min(b.i, b.j), std::max(b.i, b.j)}] += b.J;
    }
    auto ch = [](int t, int op) { return 1 + 3 * (t - 1) + op; };

    for (int s = 0; s < N; ++s) {
        int Wl = H.W[s].shape[0], Wr = H.W[s].shape[1];
        // boundary tensors keep only the start row (s == 0) / done column (s == N-1)
        auto li = [&](int st) { return s == 0 ? (st == 0 ? 0 : -1) : st; };
        auto ri = [&](int st) { return s == N - 1 ? (st == done ? 0 : -1) : st; };
        std::vector<double> hW((size_t)Wl * Wr * d * d, 0.0);
        auto add = [&](int a, int b, int p, int q, double c) {
            int l = li(a), r = ri(b);
            if (l >= 0 && r >= 0 && c != 0.0) hW[layout::mpo_idx(l, r, p, q, Wl, Wr, d)] += c;
        };
        auto I  = [&](int a, int b)           { add(a,b,0,0,1.0); add(a,b,1,1,1.0); };
        auto Sz = [&](int a, int b, double c) { add(a,b,0,0,0.5*c); add(a,b,1,1,-0.5*c); };
        auto Sp = [&](int a, int b, double c) { add(a,b,0,1,c); };
        auto Sm = [&](int a, int b, double c) { add(a,b,1,0,c); };

        I(0, 0);
        I(done, done);
        Sz(0, ch(1,0), 1.0); Sp(0, ch(1,1), 1.0); Sm(0, ch(1,2), 1.0);
        for (int t = 1; t < kmax; ++t)
            for (int op = 0; op < 3; ++op) I(ch(t,op), ch(t+1,op));
        for (int t = 1; t <= kmax && s - t >= 0; ++t) {
            auto it = J.find({s - t, s});
            if (it == J.end()) continue;
            double c = it->second;
            Sz(ch(t,0), done, c);          // Sz_i Sz_j
            Sm(ch(t,1), done, 0.5 * c);    // S+_i S-_j / 2
            Sp(ch(t,2), done, 0.5 * c);    // S-_i S+_j / 2
        }
        H.W[s].from_host(hW);
    }
}

std::vector<Bond> j1j2_cylinder_bonds(int Lx, int Ly, double J1, double J2)
{
    std::vector<Bond> b;
    auto site = [&](int x, int y) { return x * Ly + ((y % Ly) + Ly) % Ly; };
    bool pbc = Ly >= 3;
    for (int x = 0; x < Lx; ++x)
        for (int y = 0; y < Ly; ++y) {
            if (y + 1 < Ly || pbc) b.push_back({site(x,y), site(x,y+1), J1});      // vertical
            if (x + 1 < Lx) {
                b.push_back({site(x,y), site(x+1,y), J1});                           // horizontal
                if (y + 1 < Ly || pbc) b.push_back({site(x,y), site(x+1,y+1), J2});  // diag up
                if (y - 1 >= 0 || pbc) b.push_back({site(x,y), site(x+1,y-1), J2}); // diag down
            }
        }
    return b;
}

std::vector<Bond> tri_ladder_bonds(int Lx, double J)
{
    std::vector<Bond> b;
    for (int k = 0; k < Lx; ++k) b.push_back({2*k, 2*k+1, J});
    for (int k = 0; k + 1 < Lx; ++k) {
        b.push_back({2*k,   2*k+2, J});
        b.push_back({2*k+1, 2*k+3, J});
        b.push_back({2*k+1, 2*k+2, J});
        b.push_back({2*k,   2*k+3, J});
    }
    return b;
}

namespace {
// Thin SVD of a col-major m x n host matrix via cuSOLVER gesvd (transposing when m < n).
// Returns U (m x k), S (k), Vt (k x n), k = min(m, n); all col-major host arrays.
void host_svd(int m, int n, const std::vector<double>& M,
              std::vector<double>& U, std::vector<double>& S, std::vector<double>& Vt)
{
    static cusolverDnHandle_t h = [] { cusolverDnHandle_t x; CUSOLVER_CHECK(cusolverDnCreate(&x)); return x; }();
    const bool tr = m < n;
    const int mm = tr ? n : m, nn = tr ? m : n, k = nn;
    std::vector<double> A((size_t)mm * nn);
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j)
        (tr ? A[j + (size_t)mm * i] : A[i + (size_t)mm * j]) = M[i + (size_t)m * j];
    double *dA, *dS, *dU, *dVt, *dw, *drw; int *info, lw = 0;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * 8)); CUDA_CHECK(cudaMalloc(&dS, k * 8));
    CUDA_CHECK(cudaMalloc(&dU, (size_t)mm * k * 8)); CUDA_CHECK(cudaMalloc(&dVt, (size_t)k * nn * 8));
    CUDA_CHECK(cudaMalloc(&drw, k * 8)); CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(h, mm, nn, &lw));
    CUDA_CHECK(cudaMalloc(&dw, (size_t)lw * 8));
    CUSOLVER_CHECK(cusolverDnDgesvd(h, 'S', 'S', mm, nn, dA, mm, dS, dU, mm, dVt, k, dw, lw, drw, info));
    std::vector<double> u((size_t)mm * k), vt((size_t)k * nn);
    S.assign(k, 0.0);
    CUDA_CHECK(cudaMemcpy(u.data(), dU, u.size() * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vt.data(), dVt, vt.size() * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(S.data(), dS, k * 8, cudaMemcpyDeviceToHost));
    int hinfo; CUDA_CHECK(cudaMemcpy(&hinfo, info, sizeof(int), cudaMemcpyDeviceToHost));
    for (double* p : {dA, dS, dU, dVt, dw, drw}) cudaFree(p);
    cudaFree(info);
    if (hinfo != 0) throw std::runtime_error("compress_mpo: gesvd failed");
    U.assign((size_t)m * k, 0.0); Vt.assign((size_t)k * n, 0.0);
    if (!tr) { U = u; Vt = vt; }
    else {   // A^T = u S vt  =>  A = vt^T S u^T
        for (int i = 0; i < m; ++i) for (int j = 0; j < k; ++j) U[i + (size_t)m * j] = vt[j + (size_t)k * i];
        for (int j = 0; j < k; ++j) for (int c = 0; c < n; ++c) Vt[j + (size_t)k * c] = u[c + (size_t)n * j];
    }
}
}

int compress_mpo(MPO& H, double rel_tol)
{
    const int N = H.N, d = H.d, dd = d * d;
    std::vector<std::vector<double>> w(N);
    std::vector<int> Wl(N), Wr(N);
    for (int s = 0; s < N; ++s) { w[s] = H.W[s].to_host(); Wl[s] = H.W[s].shape[0]; Wr[s] = H.W[s].shape[1]; }
    auto keep = [&](const std::vector<double>& S) {
        int k = 0; while (k < (int)S.size() && S[k] > rel_tol * S[0]) ++k; return std::max(k, 1); };

    // left-to-right: W[s] as ((wl,p,q) x wr) -> U ; carry diag(S) Vt into W[s+1]
    for (int s = 0; s + 1 < N; ++s) {
        int m = Wl[s] * dd, n = Wr[s];
        std::vector<double> M((size_t)m * n);
        for (int a = 0; a < Wl[s]; ++a) for (int b = 0; b < n; ++b) for (int pq = 0; pq < dd; ++pq)
            M[(a + Wl[s] * pq) + (size_t)m * b] = w[s][a + Wl[s] * (b + n * pq)];
        std::vector<double> U, S, Vt; host_svd(m, n, M, U, S, Vt);
        int k = keep(S), K = (int)S.size();
        std::vector<double> nw((size_t)Wl[s] * k * dd);
        for (int a = 0; a < Wl[s]; ++a) for (int j = 0; j < k; ++j) for (int pq = 0; pq < dd; ++pq)
            nw[a + Wl[s] * (j + k * pq)] = U[(a + Wl[s] * pq) + (size_t)m * j];
        std::vector<double> nxt((size_t)k * Wr[s + 1] * dd, 0.0);
        for (int j = 0; j < k; ++j) for (int b = 0; b < n; ++b) {
            double c = S[j] * Vt[j + (size_t)K * b];
            if (c == 0.0) continue;
            for (int e = 0; e < Wr[s + 1]; ++e) for (int pq = 0; pq < dd; ++pq)
                nxt[j + k * (e + Wr[s + 1] * pq)] += c * w[s + 1][b + Wl[s + 1] * (e + Wr[s + 1] * pq)];
        }
        w[s] = nw; Wr[s] = k; w[s + 1] = nxt; Wl[s + 1] = k;
    }
    // right-to-left: W[s] as (wl x (wr,p,q)) -> Vt ; carry U diag(S) into W[s-1]
    for (int s = N - 1; s > 0; --s) {
        int m = Wl[s], n = Wr[s] * dd;
        std::vector<double> U, S, Vt; host_svd(m, n, w[s], U, S, Vt);   // layout already (wl) x (wr,p,q)
        int k = keep(S), K = (int)S.size();
        std::vector<double> nw((size_t)k * n);
        for (int j = 0; j < k; ++j) for (int c = 0; c < n; ++c) nw[j + (size_t)k * c] = Vt[j + (size_t)K * c];
        std::vector<double> prv((size_t)Wl[s - 1] * k * dd, 0.0);
        for (int a = 0; a < Wl[s - 1]; ++a) for (int b = 0; b < m; ++b) for (int pq = 0; pq < dd; ++pq) {
            double x = w[s - 1][a + Wl[s - 1] * (b + m * pq)];
            if (x == 0.0) continue;
            for (int j = 0; j < k; ++j) prv[a + Wl[s - 1] * (j + k * pq)] += x * U[b + (size_t)m * j] * S[j];
        }
        w[s] = nw; Wl[s] = k; w[s - 1] = prv; Wr[s - 1] = k;
    }
    int wmax = 1;
    for (int s = 0; s < N; ++s) {
        H.W[s] = Tensor({Wl[s], Wr[s], d, d});
        H.W[s].from_host(w[s]);
        wmax = std::max({wmax, Wl[s], Wr[s]});
    }
    H.Wdim = wmax;
    return wmax;
}

std::vector<Bond> read_bond_file(const std::string& path, int& N, std::vector<SiteXY>& xy)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("read_bond_file: cannot open " + path);
    std::vector<Bond> b;
    N = -1;
    std::string line;
    while (std::getline(f, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line.resize(h);
        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key)) continue;
        if (key == "N") { ss >> N; xy.assign(N, SiteXY{}); }
        else if (key == "site") { int k; SiteXY p; ss >> k >> p.x >> p.y; if (k < 0 || k >= N) throw std::runtime_error("read_bond_file: bad site"); xy[k] = p; }
        else if (key == "bond") { Bond e; ss >> e.i >> e.j >> e.J; b.push_back(e); }
        else throw std::runtime_error("read_bond_file: unknown key " + key);
    }
    if (N <= 0 || b.empty()) throw std::runtime_error("read_bond_file: missing N or bonds in " + path);
    for (const auto& e : b)
        if (e.i < 0 || e.j < 0 || e.i >= N || e.j >= N || e.i == e.j) throw std::runtime_error("read_bond_file: bad bond");
    return b;
}
