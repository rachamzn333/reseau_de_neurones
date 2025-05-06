#include <iostream>
#include "mnist_loader.h"
#include "denseNN.h"
#include "training.h"
#include <random>

// chargement des données
constexpr const char* TRAIN_IMAGES = "../Version_Fully_Connected/data/train-images-idx3-ubyte";
constexpr const char* TRAIN_LABELS = "../Version_Fully_Connected/data/train-labels-idx1-ubyte";
constexpr const char* TEST_IMAGES  = "../Version_Fully_Connected/data/t10k-images-idx3-ubyte";
constexpr const char* TEST_LABELS  = "../Version_Fully_Connected/data/t10k-labels-idx1-ubyte";


constexpr int   EPOCHS = 6;
constexpr float LR = 0.01f;

int main() {
    try {
        Images Xtr = load_images(TRAIN_IMAGES);
        Labels Ytr = load_labels(TRAIN_LABELS);
        Images Xte = load_images(TEST_IMAGES);
        Labels Yte = load_labels(TEST_LABELS);

        std::mt19937 gen(42);
        DenseNN net(LR, gen);
        train_epoch_loop(net, Xtr, Ytr, Xte, Yte, EPOCHS);
    }
    catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
