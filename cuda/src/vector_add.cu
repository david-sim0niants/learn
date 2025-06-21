#include "vector_add.h"

#include <cstdio>

#include <cuda_runtime.h>

template<typename T>
__global__ void add_vectors_kernel(const T *a, const T *b, T *c, size_t size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] + b[i];
}

template<typename T>
inline void add_vectors_impl(const T *dev_a, const T *dev_b, T *dev_c, size_t size)
{
    add_vectors_kernel<<<(size + 255) / 256, 256>>>(dev_a, dev_b, dev_c, size);
    cudaDeviceSynchronize();
}

void add_vectors(const int *dev_a, const int *dev_b, int *dev_c, size_t size)
{
    add_vectors_impl(dev_a, dev_b, dev_c, size);
}

void add_vectors(const float *dev_a, const float *dev_b, float *dev_c, size_t size)
{
    add_vectors_impl(dev_a, dev_b, dev_c, size);
}

void add_vectors(const double *dev_a, const double *dev_b, double *dev_c, size_t size)
{
    add_vectors_impl(dev_a, dev_b, dev_c, size);
}
