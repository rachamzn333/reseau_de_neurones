#pragma once
#include <vector>
#include <cstdint>

using Tensor = std::vector<float>;   // 1-D flat array
using Label = uint8_t;
using Images = std::vector<Tensor>;
using Labels = std::vector<Label>;

constexpr int IMG_SIZE = 28;
constexpr int NUM_CLASSES = 10;
