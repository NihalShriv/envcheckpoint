#pragma once

#include <vector>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line)
{
    if(status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) + " for " + expr + ": " + cudaGetErrorString(status));
}

inline void check_cublas(cublasStatus_t status, const char* expr, const char* file, int line)
{
    if(status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS error at ") + file + ":" + std::to_string(line) + " for " + expr + ": code=" + std::to_string(static_cast<int>(status)));
}

inline void check_cusolver(cusolverStatus_t status, const char* expr, const char* file, int line)
{
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuSOLVER error at ") + file + ":" + std::to_string(line) + " for " + expr + ": code=" + std::to_string(static_cast<int>(status)));
}

inline void check_last_kernel(const char* expr, const char* file, int line)
{
    check_cuda(cudaGetLastError(), expr, file, line);
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr, __FILE__, __LINE__)
#define CUSOLVER_CHECK(expr) check_cusolver((expr), #expr, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK(expr) do { expr; check_last_kernel(#expr, __FILE__, __LINE__); } while(0)

class Tensor {
public:
    double* d_data = nullptr;
    std::vector<int> shape;
    int size = 0;

    Tensor() = default;
    Tensor(const std::vector<int>& shape_);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void reshape(const std::vector<int>& new_shape);
    void zero();
    void copy_from(const Tensor& other);
    void from_host(const std::vector<double>& h);
    std::vector<double> to_host() const;
    double item() const;
};
