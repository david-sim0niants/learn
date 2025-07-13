#pragma once

#include <concepts>
#include <random>

class PRNG {
public:
    using SeedType = std::mt19937::result_type;

    PRNG() : generator(std::random_device{}()) {}

    explicit PRNG(SeedType seed) : generator(seed) {}

    template<std::integral Integral = int>
    inline Integral gen_uniform(Integral min = std::numeric_limits<Integral>::min(),
                    Integral max = std::numeric_limits<Integral>::max())
    {
        return std::uniform_int_distribution<Integral>{min, max}(generator);
    }

    template<std::integral Integral = int>
    inline Integral operator()(
            Integral min = std::numeric_limits<Integral>::min(),
            Integral max = std::numeric_limits<Integral>::max())
    {
        return gen_uniform<Integral>(min, max);
    }

    template<std::floating_point Real = double>
    inline Real gen_uniform(
            Real min = std::numeric_limits<Real>::min(),
            Real max = std::numeric_limits<Real>::max())
    {
        return std::uniform_real_distribution<Real>{min, max}(generator);
    }

    template<std::floating_point Real = double>
    inline Real operator()(
            Real min = std::numeric_limits<Real>::min(),
            Real max = std::numeric_limits<Real>::max())
    {
        return gen_uniform(min, max);
    }

    inline std::mt19937 get_generator()
    {
        return generator;
    }

private:
    std::mt19937 generator;
};
