#pragma once
#include "cnn.h"
#include "tensor.h"

void train_epoch_loop(CNN& net,
    const Images& Xtr, const Labels& Ytr,
    const Images& Xte, const Labels& Yte,
    int epochs);

