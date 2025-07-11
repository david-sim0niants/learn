#include <iostream>
#include <immintrin.h>

__attribute__((noinline)) void do_permute(float res[8])
{
    volatile __m256 vec = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);

    __m256 permuted = _mm256_permute_ps(vec, 0b00011011);

    _mm256_storeu_ps(res, permuted);
}

int main()
{
    float res[8];
    do_permute(res);

    for (int i = 0; i < 8; ++i)
        std::cout << res[i] << std::endl;

    return 0;
}
