#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "prng.h"

PRNG prng (42);

template<typename T>
std::vector<T> gen_vector(int size)
{
    T range = std::sqrt(T(100000) / size);

    std::vector<T> vec(size);
    for (int i = 0; i < size; ++i)
        vec[i] = prng.gen_uniform<T>(-range, +range);
    return vec;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
        return EXIT_FAILURE;

    int vec_size = std::stoi(argv[1]);
    if (vec_size < 0)
        return EXIT_FAILURE;

    using Scalar = float;

    auto A = gen_vector<Scalar>(vec_size);
    auto B = gen_vector<Scalar>(vec_size);

    Scalar c = 0.0F;

    for (int i = 0; i < vec_size; ++i) {
        c += A[i] * B[i];
    }

    std::cout << c << std::endl;
    return EXIT_SUCCESS;
}
