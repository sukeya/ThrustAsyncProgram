#include "double.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  auto floats = thrust::host_vector<float>();
  auto ints   = thrust::host_vector<int>();

  std::size_t size = 10000;

  floats.reserve(size);
  ints.reserve(size);

  for (std::size_t i = 0; i < size; ++i) {
    floats.push_back(i);
    ints.push_back(i);
  }

  auto event = Double(floats, ints);

  event.wait();

  for (std::size_t i = 0; i < size; ++i) {
    assert(std::abs(floats[i] - i * i) < 1e-5);
    assert(ints[i] == static_cast<int>(i * i));
  }
  std::cout << "Success!\n";

  return 0;
}