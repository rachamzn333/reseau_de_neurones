#include "training.h"
#include <algorithm>
#include <numeric>
#include <iostream>

void train_epoch_loop(DenseNN& net,
    const Images& Xtr, const Labels& Ytr,
    const Images& Xte, const Labels& Yte,
    int epochs)
{
    std::vector<int> idx(Xtr.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 gen(42);

    for (int ep = 1; ep <= epochs; ++ep) {               
        std::shuffle(idx.begin(), idx.end(), gen);
        double loss_sum = 0;

        
        for (int i : idx) loss_sum += net.train_one(Xtr[i], Ytr[i]);

        int correct = 0;
        for (size_t i = 0; i < Xte.size(); ++i)
            if (net.predict(Xte[i]) == Yte[i]) ++correct;

        std::cout << "Epoch " << ep
            << "  loss=" << loss_sum / idx.size()
            << "  test_acc=" << (100.0 * correct / Xte.size()) << "%\n";
    }
}