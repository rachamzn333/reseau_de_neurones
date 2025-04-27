#pragma once
#include "layers.h"
#include <random>

class CNN
{
public:
    explicit CNN(float lr, std::mt19937& g);

    // NOTE:  ► no “const” on these ◄
    Tensor forward(const Tensor& img);
    float  train_one(const Tensor& img, Label y);
    int    predict(const Tensor& img);

private:
    ConvLayer conv_;
    ReLU      relu_;
    MaxPool   pool_;
    Dense     fc_;
    float     lr_;
};
