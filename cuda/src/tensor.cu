#include "tensor.h"
#include <iostream>

Tensor::Tensor(const std::vector<int>& shape_)
    : shape(shape_)
{
    size = 1;
    for(int s : shape)
        size *= s;

    // envck tree (v7): stream-ordered allocation on the legacy default stream. Plain cudaFree synchronises the whole
    // device, which would serialise any copy running on another stream (pinned async offload); cudaFreeAsync does not.
    cudaMallocAsync((void**)&d_data, size*sizeof(double), 0);
}

Tensor::~Tensor()
{
    if(d_data)
        cudaFreeAsync(d_data, 0);
}

Tensor::Tensor(Tensor&& other) noexcept
{
    d_data = other.d_data;
    shape  = std::move(other.shape);
    size   = other.size;
    other.d_data = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if(this != &other)
    {
        if(d_data)
            cudaFreeAsync(d_data, 0);

        d_data = other.d_data;
        shape  = std::move(other.shape);
        size   = other.size;
        other.d_data = nullptr;
    }
    return *this;
}

void Tensor::reshape(const std::vector<int>& new_shape)
{
    int new_size = 1;
    for(int s : new_shape)
        new_size *= s;

    if(new_size != size)
    {
        std::cerr << "Reshape size mismatch\n";
        exit(1);
    }

    shape = new_shape;
}

void Tensor::zero()
{
    cudaMemset(d_data, 0, size*sizeof(double));
}

void Tensor::copy_from(const Tensor& other)
{
    if(size != other.size)
    {
        std::cerr << "Tensor copy size mismatch\n";
        exit(1);
    }

    cudaMemcpy(d_data, other.d_data,
               size*sizeof(double),
               cudaMemcpyDeviceToDevice);
}

void Tensor::from_host(const std::vector<double>& h)
{
    cudaMemcpy(d_data, h.data(),
               size*sizeof(double),
               cudaMemcpyHostToDevice);
}

std::vector<double> Tensor::to_host() const
{
    std::vector<double> h(size);
    cudaMemcpy(h.data(), d_data,
               size*sizeof(double),
               cudaMemcpyDeviceToHost);
    return h;
}

double Tensor::item() const
{
    double value = 0.0;
    cudaMemcpy(&value, d_data,
               sizeof(double),
               cudaMemcpyDeviceToHost);
    return value;
}