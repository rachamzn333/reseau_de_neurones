#pragma once
#include "denseNN.h"
#include "tensor.h"

void train_epoch_loop(DenseNN& net,
    const Images& Xtr, const Labels& Ytr,
    const Images& Xte, const Labels& Yte,
    int epochs);