// denseNN.cpp  – implementations only
#include <iostream>

#include "denseNN.h"
#include <algorithm>
#include <cmath>

/* ───────── constructor ───────── */
DenseNN::DenseNN(float lr, std::mt19937& g)
    : layer1_(IMG_SIZE* IMG_SIZE, 256, g),
    relu1_{},
    layer2_(256, 128, g),
    relu2_{},
    layer3_(128, 64, g),
    relu3_{},
    layer4_(64, NUM_CLASSES, g),
    lr_(lr)
{}

/* ───────── forward pass ───────── */
Tensor DenseNN::forward(const Tensor& x) {
    return layer4_.forward(
        relu3_.forward(
            layer3_.forward(
                relu2_.forward(
                    layer2_.forward(
                        relu1_.forward(
                            layer1_.forward(x)))))));
}

/* ───────── single-sample training step ───────── */
float DenseNN::train_one(const Tensor& x, Label y)
{
    auto logits = forward(x);

    // soft-max
    float maxv = *std::max_element(logits.begin(), logits.end());
    Tensor p(logits.size());
    float sum = 0.f;
    for (size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp(logits[i] - maxv);
        sum += p[i];
    }
    for (float& v : p) v /= sum;

    float loss = -std::log(std::max(1e-7f, p[y]));

    // gradient
    Tensor d_logits(p.size());
    for (size_t i = 0; i < p.size(); ++i)
        d_logits[i] = p[i] - (i == static_cast<size_t>(y) ? 1.f : 0.f);

    auto d_layer4 = layer4_.backward(d_logits, lr_);
    auto d_relu3 = relu3_.backward(d_layer4);
    auto d_layer3 = layer3_.backward(d_relu3, lr_);
    auto d_relu2 = relu2_.backward(d_layer3);
    auto d_layer2 = layer2_.backward(d_relu2, lr_);
    auto d_relu1 = relu1_.backward(d_layer2);
    layer1_.backward(d_relu1, lr_);

    return loss;
}

/* ───────── inference ───────── */
int DenseNN::predict(const Tensor& x)
{
    const Tensor y = forward(x);
    return static_cast<int>(std::distance(
        y.begin(), std::max_element(y.begin(), y.end())));
}