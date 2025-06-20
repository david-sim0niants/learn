#include "vector_add.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

int main()
{
    const int size = 1 << 20;

    std::vector<int> a(size), b(size), c(size, 0);

    for (int i = 0; i < size; ++i) {
        a[i] = rand();
        b[i] = rand();
    }

    add_vectors(a.data(), b.data(), c.data(), size);

    for (int i = 0; i < size; ++i)
        assert(c[i] == a[i] + b[i]);

    std::cerr << "PASSED\n";

    return EXIT_SUCCESS;
}
