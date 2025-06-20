#include "vector_add.h"

#include <cstdio>

#include <cuda_runtime.h>

template<typename T>
static __global__ void add_vectors_kernel(const T *a, const T *b, T *c, size_t size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] + b[i];
}

template<typename T>
void add_vectors_(const T *a, const T *b, T *c, size_t size)
{
    const size_t mem_size = size * sizeof(T);

    T *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;

    cudaMalloc(&dev_a, mem_size);
    cudaMalloc(&dev_b, mem_size);
    cudaMalloc(&dev_c, mem_size);

    cudaMemcpy(dev_a, a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, mem_size, cudaMemcpyHostToDevice);

    add_vectors_kernel<<<(size + 255) / 256, 256>>>(dev_a, dev_b, dev_c, size);

    cudaMemcpy(c, dev_c, mem_size, cudaMemcpyDeviceToHost);
}

void add_vectors(const int *a, const int *b, int *c, size_t size)
{
    add_vectors_(a, b, c, size);
}
