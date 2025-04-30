#pragma once
#include "layers.h"
#include "tensor.h"           // définit Tensor, Images, Labels, Label
#include <random>
#include <vector>

class CNN
{
public:
    explicit CNN(float lr, std::mt19937& g);

    /* --- API --- */
    Tensor forward    (const Tensor& img);                       // inference
    float  train_one  (const Tensor& img, Label y);              // 1 image : accumule grad
    float  train_batch(const Images& X, const Labels& Y,         // applique grad 1×/lot
                       const std::vector<int>& batch_idx,
                       int batch_sz);

    int    predict(const Tensor& img);

private:
    ConvLayer conv_;
    ReLU      relu_;
    MaxPool   pool_;
    Dense     fc_;
    float     lr_;               // taux d’apprentissage courant
};
