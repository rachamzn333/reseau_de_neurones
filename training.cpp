#include "training.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

void train_epoch_loop(CNN& net,
                      const Images&  Xtr, const Labels&  Ytr,
                      const Images&  Xte, const Labels&  Yte,
                      int  epochs,
                      int  batch_size)
{
    /* --- préparation --- */
    std::vector<int> idx(Xtr.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 gen(42);

    for (int ep = 1; ep <= epochs; ++ep) {            /* PARALLEL_CANDIDATE_OpenMP */
        auto t0 = std::chrono::steady_clock::now();   // départ chrono

        std::shuffle(idx.begin(), idx.end(), gen);
        double loss_sum = 0.0;

        /* ---- boucle mini-lots ---- */
        for (std::size_t pos = 0; pos < idx.size(); pos += batch_size) {
            std::size_t end = std::min(pos + batch_size, idx.size());

            /* indices du lot courant */
            std::vector<int> batch_idx(idx.begin() + pos, idx.begin() + end);

            /* entraîne et récupère la perte moyenne du lot          *
             * (=> on la re-multiplie par sa taille pour avoir       *
             *    la somme des pertes individuelles).                */
            loss_sum += net.train_batch(Xtr, Ytr,
                                        batch_idx,
                                        static_cast<int>(batch_idx.size()))
                         * static_cast<double>(batch_idx.size());
        }

        /* ---- évaluation jeu de test ---- */
        int correct = 0;
        for (std::size_t i = 0; i < Xte.size(); ++i)
            if (net.predict(Xte[i]) == Yte[i]) ++correct;

        auto t1 = std::chrono::steady_clock::now();   // arrêt chrono
        double elapsed_s =
            std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Epoch "   << ep
                  << "  loss="      << loss_sum / static_cast<double>(idx.size())
                  << "  test_acc="  << (100.0 * correct / Xte.size()) << '%'
                  << "  time="      << elapsed_s << " s\n";
    }
}
