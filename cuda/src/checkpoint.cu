#include "checkpoint.h"
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
const uint64_t MAGIC = 0x31544b43474d5244ull;  // "DRMGCKT1"

template <class T> void put(std::ofstream& f, const T& v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> T get(std::ifstream& f) {
    T v; f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error("checkpoint: truncated file");
    return v;
}
}

void save_checkpoint(const std::string& path, const MPS& psi, const RunState& st)
{
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("checkpoint: cannot open " + tmp);
        put(f, MAGIC);
        put<int32_t>(f, psi.N); put<int32_t>(f, psi.d); put<int32_t>(f, psi.D);
        put<int32_t>(f, st.sweeps_done); put<double>(f, st.energy);
        for (const auto& A : psi.A) {
            put<int32_t>(f, (int32_t)A.shape.size());
            for (int s : A.shape) put<int32_t>(f, s);
            std::vector<double> h = A.to_host();
            f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(double));
        }
        put(f, MAGIC);  // trailer: detects truncated writes on load
        f.flush();
        if (!f) throw std::runtime_error("checkpoint: write failed for " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("checkpoint: rename failed " + tmp + " -> " + path);
}

bool load_checkpoint(const std::string& path, MPS& psi, RunState& st)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    if (get<uint64_t>(f) != MAGIC) throw std::runtime_error("checkpoint: bad magic in " + path);
    int N = get<int32_t>(f), d = get<int32_t>(f);
    get<int32_t>(f);  // D stored for information; the run's D schedule decides the cap
    if (N != psi.N || d != psi.d) throw std::runtime_error("checkpoint: N/d mismatch in " + path);
    st.sweeps_done = get<int32_t>(f);
    st.energy = get<double>(f);
    for (int i = 0; i < N; ++i) {
        int rank = get<int32_t>(f);
        std::vector<int> shape(rank);
        for (auto& s : shape) s = get<int32_t>(f);
        Tensor A(shape);
        std::vector<double> h(A.size);
        f.read(reinterpret_cast<char*>(h.data()), h.size() * sizeof(double));
        if (!f) throw std::runtime_error("checkpoint: truncated tensor data in " + path);
        A.from_host(h);
        psi.A[i] = std::move(A);
    }
    if (get<uint64_t>(f) != MAGIC) throw std::runtime_error("checkpoint: bad trailer in " + path);
    return true;
}
