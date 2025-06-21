#include "vector_add.h"
#include "device_buffer.h"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

template<typename T>
static void bm_vector_add(benchmark::State& state)
{
    const std::size_t size = state.range(0);

    std::vector<T> a(size, T(1)), b(size, T(2));

    DeviceBuffer dev_a(a.data(), size), dev_b(b.data(), size);
    DeviceBuffer<T> dev_c(size);

    for (auto _ : state)
    {
        add_vectors(dev_a.data(), dev_b.data(), dev_c.data(), size);
    }

    std::vector<T> c(size);
    dev_c.load_to(c.data());
}

BENCHMARK_TEMPLATE(bm_vector_add, int)->RangeMultiplier(2)->Range(1 << 10, 1 << 24);
BENCHMARK_TEMPLATE(bm_vector_add, float)->RangeMultiplier(2)->Range(1 << 10, 1 << 24);
BENCHMARK_TEMPLATE(bm_vector_add, double)->RangeMultiplier(2)->Range(1 << 10, 1 << 24);
