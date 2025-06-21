#pragma once

#include <stdexcept>

#include <cuda_runtime.h>

template<typename ErrorType = std::runtime_error>
inline void cuda_assert(cudaError_t err = cudaGetLastError())
{
    if (err != cudaSuccess)
        throw ErrorType(cudaGetErrorString(err));
}
