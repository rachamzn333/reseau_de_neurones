#pragma once
#include "layers.h"
#include <random>

class DenseNN
{
public:
    explicit DenseNN(float lr, std::mt19937& g);

    // NOTE:  ► no "const" on these ◄
    Tensor forward(const Tensor& img);
    float  train_one(const Tensor& img, Label y);
    int    predict(const Tensor& img);

private:
    Dense layer1_;    // Input layer (IMG_SIZE*IMG_SIZE -> 256)
    ReLU  relu1_;     // First activation
    Dense layer2_;    // Hidden layer (256 -> 128)
    ReLU  relu2_;     // Second activation
    Dense layer3_;    // Hidden layer (128 -> 64)
    ReLU  relu3_;     // Third activation
    Dense layer4_;    // Output layer (64 -> 10)
    float lr_;        // Learning rate
};