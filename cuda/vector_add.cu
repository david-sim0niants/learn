#include "vector_add.h"
#include "cuda_error.h"
#include "device_buffer.h"

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
void add_vectors_impl(const T *a, const T *b, T *c, size_t size)
{
    DeviceBuffer dev_a(a, size), dev_b(b, size), dev_c(c, size);
    add_vectors_kernel<<<(size + 255) / 256, 256>>>(dev_a.data(), dev_b.data(), dev_c.data(), size);
    cuda_assert();
    dev_c.load_to(c);
}

void add_vectors(const int *a, const int *b, int *c, size_t size)
{
    add_vectors_impl(a, b, c, size);
}
