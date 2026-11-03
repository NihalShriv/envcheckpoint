#include "environment.h"
#include "ct_contract.h"
#include <cmath>

namespace {
__global__ void left_env_kernel(
    double* out,
    const double* L,
    const double* A,
    const double* W,
    int Dl,
    int d,
    int Dr,
    int Wl,
    int Wr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Dr * Dr * Wr)
        return;

    int c = idx % Dr;
    int tmp = idx / Dr;
    int d2 = tmp % Dr;
    int y = tmp / Dr;

    double sum = 0.0;
    for(int a=0;a<Dl;a++)
    for(int b=0;b<Dl;b++)
    for(int x=0;x<Wl;x++)
    for(int p=0;p<d;p++)
    for(int q=0;q<d;q++)
    {
        sum += L[layout::env_idx(a, b, x, Dl, Dl)]
             * A[layout::mps_idx(a, p, c, Dl, d)]
             * W[layout::mpo_idx(x, y, p, q, Wl, Wr, d)]
             * A[layout::mps_idx(b, q, d2, Dl, d)];
    }

    out[idx] = sum;
}

__global__ void right_env_kernel(
    double* out,
    const double* R,
    const double* A,
    const double* W,
    int Dl,
    int d,
    int Dr,
    int Wl,
    int Wr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Dl * Dl * Wl)
        return;

    int a = idx % Dl;
    int tmp = idx / Dl;
    int b = tmp % Dl;
    int x = tmp / Dl;

    double sum = 0.0;
    for(int c=0;c<Dr;c++)
    for(int d2=0;d2<Dr;d2++)
    for(int y=0;y<Wr;y++)
    for(int p=0;p<d;p++)
    for(int q=0;q<d;q++)
    {
        sum += R[layout::env_idx(c, d2, y, Dr, Dr)]
             * A[layout::mps_idx(a, p, c, Dl, d)]
             * W[layout::mpo_idx(x, y, p, q, Wl, Wr, d)]
             * A[layout::mps_idx(b, q, d2, Dl, d)];
    }

    out[idx] = sum;
}
} // namespace

Environment::Environment(int N)
{
    L.resize(N+1);
    R.resize(N+1);
}

void Environment::build_left(
    const MPS& psi,
    const MPO& H)
{
    int N = psi.N;
    L[0] = Tensor({1,1,1});
    L[0].from_host({1.0});

    for(int i=0;i<N;i++)
        update_left_site(psi, H, i);
}

// newL[c,d2,y] = sum L[a,b,x] A[a,p,c] W[x,y,p,q] A[b,q,d2], as three O(D^3 W) contractions.
// Layouts (fastest first): L,R {left,right,mpo}; A {left,phys,right}; W {wl,wr,p,q}.
void Environment::update_left_site(
    const MPS& psi,
    const MPO& H,
    int i)
{
    const Tensor& A = psi.A[i];
    const Tensor& W = H.W[i];
    int Dl = A.shape[0];
    int d  = A.shape[1];
    int Dr = A.shape[2];
    int Wl = W.shape[0];
    int Wr = W.shape[1];

    Tensor newL({Dr,Dr,Wr});
    if(naive) {
        int total = Dr * Dr * Wr;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        left_env_kernel<<<blocks,threads>>>(newL.d_data, L[i].d_data, A.d_data, W.d_data, Dl, d, Dr, Wl, Wr);
    } else {
        double* T1 = ct_scratch(0, (size_t)Dl * Wl * d * Dr);
        double* T2 = ct_scratch(1, (size_t)Dl * Wr * d * Dr);
        ct_contract({'a','b','x'}, {Dl,Dl,Wl}, L[i].d_data, {'a','p','c'}, {Dl,d,Dr}, A.d_data,
                    {'b','x','p','c'}, {Dl,Wl,d,Dr}, T1);
        ct_contract({'b','x','p','c'}, {Dl,Wl,d,Dr}, T1, {'x','y','p','q'}, {Wl,Wr,d,d}, W.d_data,
                    {'b','y','q','c'}, {Dl,Wr,d,Dr}, T2);
        ct_contract({'b','y','q','c'}, {Dl,Wr,d,Dr}, T2, {'b','q','e'}, {Dl,d,Dr}, A.d_data,
                    {'c','e','y'}, {Dr,Dr,Wr}, newL.d_data);
    }
    L[i+1] = std::move(newL);
}

void Environment::build_right(
    const MPS& psi,
    const MPO& H)
{
    int N = psi.N;
    R[N] = Tensor({1,1,1});
    R[N].from_host({1.0});

    for(int i=N-1;i>=0;i--)
        update_right_site(psi, H, i);
}

// newR[a,b,x] = sum R[c,d2,y] A[a,p,c] W[x,y,p,q] A[b,q,d2]
void Environment::update_right_site(
    const MPS& psi,
    const MPO& H,
    int i)
{
    const Tensor& A = psi.A[i];
    const Tensor& W = H.W[i];
    int Dl = A.shape[0];
    int d  = A.shape[1];
    int Dr = A.shape[2];
    int Wl = W.shape[0];
    int Wr = W.shape[1];

    Tensor newR({Dl,Dl,Wl});
    if(naive) {
        int total = Dl * Dl * Wl;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        right_env_kernel<<<blocks,threads>>>(newR.d_data, R[i+1].d_data, A.d_data, W.d_data, Dl, d, Dr, Wl, Wr);
    } else {
        double* T1 = ct_scratch(0, (size_t)Dl * d * Dr * Wr);
        double* T2 = ct_scratch(1, (size_t)Dl * Wl * d * Dr);
        ct_contract({'a','p','c'}, {Dl,d,Dr}, A.d_data, {'c','e','y'}, {Dr,Dr,Wr}, R[i+1].d_data,
                    {'a','p','e','y'}, {Dl,d,Dr,Wr}, T1);
        ct_contract({'a','p','e','y'}, {Dl,d,Dr,Wr}, T1, {'x','y','p','q'}, {Wl,Wr,d,d}, W.d_data,
                    {'a','x','q','e'}, {Dl,Wl,d,Dr}, T2);
        ct_contract({'a','x','q','e'}, {Dl,Wl,d,Dr}, T2, {'b','q','e'}, {Dl,d,Dr}, A.d_data,
                    {'a','b','x'}, {Dl,Dl,Wl}, newR.d_data);
    }
    R[i] = std::move(newR);
}
bool Environment::is_well_conditioned(double max_abs_threshold) const
{
    auto finite_and_bounded = [&](const Tensor& T) {
        if(T.size == 0)
            return true;
        auto host = T.to_host();
        for(double value : host)
        {
            if(!std::isfinite(value) || std::fabs(value) > max_abs_threshold)
                return false;
        }
        return true;
    };

    for(const Tensor& tensor : L)
    {
        if(!finite_and_bounded(tensor))
            return false;
    }

    for(const Tensor& tensor : R)
    {
        if(!finite_and_bounded(tensor))
            return false;
    }

    return true;
}
