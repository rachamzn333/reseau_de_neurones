// cnn.cpp  – implementations only
#include <iostream>

#include "cnn.h"
#include <algorithm>
#include <cmath>

/* ───────── constructor ───────── */
CNN::CNN(float lr, std::mt19937& g)
    : conv_(1, 8, 3, g),
    relu_{},
    pool_{},
    fc_(8 * 14 * 14, 10, g),
    lr_(lr)
{}

/* ───────── forward pass ───────── */
Tensor CNN::forward(const Tensor& x) {
    return fc_.forward(
        pool_.forward(
            relu_.forward(
                conv_.forward(x))));
}

/* ───────── single-sample training step ───────── */
float CNN::train_one(const Tensor& x, Label y)
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

    auto d_fc = fc_.backward(d_logits, lr_);
    auto d_pool = pool_.backward(d_fc);
    auto d_relu = relu_.backward(d_pool);
    conv_.backward(d_relu, lr_);

    return loss;
}

/* ───────── inference ───────── */
int CNN::predict(const Tensor& x)
{
    const Tensor y = forward(x);
    return static_cast<int>(std::distance(
        y.begin(), std::max_element(y.begin(), y.end())));
}
