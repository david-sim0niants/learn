#pragma once

#include <cstddef>

void add_vectors(const int *dev_a, const int *dev_b, int *dev_c, size_t size);
void add_vectors(const float *dev_a, const float *dev_b, float *dev_c, size_t size);
void add_vectors(const double *dev_a, const double *dev_b, double *dev_c, size_t size);
