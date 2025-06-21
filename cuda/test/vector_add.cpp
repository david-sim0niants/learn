#include "vector_add.h"
#include "device_buffer.h"

#include <gtest/gtest.h>

template<typename T>
class VectorAddTest : public ::testing::Test {
protected:
    void SetUp()
    {
        srand(42);
    }
};

using TypeParams = ::testing::Types<int, float, double>;
TYPED_TEST_SUITE(VectorAddTest, TypeParams);

TYPED_TEST(VectorAddTest, BasicCheck)
{
    const size_t size = 1 << 20;

    std::vector<TypeParam> a(size), b(size), c(size, 0);

    auto max_num = std::numeric_limits<TypeParam>::max() / 2;

    for (size_t i = 0; i < size; ++i) {
        a[i] = max_num * ((double)rand() / (double)RAND_MAX);
        b[i] = max_num * ((double)rand() / (double)RAND_MAX);
    }

    DeviceBuffer dev_a(a.data(), size), dev_b(b.data(), size), dev_c(c.data(), size);

    add_vectors(dev_a.data(), dev_b.data(), dev_c.data(), size);

    dev_c.load_to(c.data());

    for (int i = 0; i < size; ++i)
        ASSERT_EQ(c[i], a[i] + b[i]);
}
