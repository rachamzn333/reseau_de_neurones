#pragma once
#include "tensor.h"
#include <random>
#include <vector>

/* ───────── Convolution (3×3, pad=1) ────────────────────────────── */
class ConvLayer {
public:
    ConvLayer(int inC, int outC, int k, std::mt19937& g);

    Tensor forward (const Tensor& in);
    Tensor backward(const Tensor& grad);               // ← lr retiré
    void   apply_gradients(int batch_sz, float lr);    // ← nouveau

private:
    int inC_, outC_, k_;
    Tensor W_, b_,             // poids
           dW_, db_,           // gradients instantanés
           gW_, gb_,           // cumul mini-lot
           cache_;             // entrée mémorisée

    int idx(int c, int y, int x, int C, int H, int W) const;
};

/* ───────── ReLU ────────────────────────────────────────────────── */
class ReLU {
public:
    Tensor forward (const Tensor& in);
    Tensor backward(const Tensor& grad);
    void   apply_gradients(int, float) {}              // stub vide
private:
    Tensor cache_;
};

/* ───────── 2×2 MaxPool ─────────────────────────────────────────── */
class MaxPool {
public:
    Tensor forward (const Tensor& in);
    Tensor backward(const Tensor& grad);
    void   apply_gradients(int, float) {}              // stub vide
private:
    int C_, H_, W_;
    std::vector<int> argmax_;
    int idx(int c, int y, int x, int C, int H, int W) const;
};

/* ───────── Fully-connected ─────────────────────────────────────── */
class Dense {
public:
    Dense(int inD, int outD, std::mt19937& g);

    Tensor forward (const Tensor& in);
    Tensor backward(const Tensor& grad);               // ← lr retiré
    void   apply_gradients(int batch_sz, float lr);    // ← nouveau

private:
    int inD_, outD_;
    Tensor W_, b_,
           dW_, db_,
           gW_, gb_,
           cache_;
};

