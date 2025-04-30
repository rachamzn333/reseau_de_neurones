#pragma once
#include "cnn.h"
#include "tensor.h"

/*  Entraîne le réseau ‟net” pendant `epochs` époques
 *  en utilisant un mini-lot de taille `batch_size`.
 */
void train_epoch_loop(CNN&  net,
                      const Images&  Xtr, const Labels&  Ytr,
                      const Images&  Xte, const Labels&  Yte,
                      int  epochs,
                      int  batch_size);
