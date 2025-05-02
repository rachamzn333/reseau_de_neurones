#include <iostream>
#include "mnist_loader.h"
#include "cnn.h"
#include "training.h"
#include <random>

constexpr const char* TRAIN_IMAGES = "D:\\mnist\\train-images.idx3-ubyte";
constexpr const char* TRAIN_LABELS = "D:\\mnist\\train-labels.idx1-ubyte";
constexpr const char* TEST_IMAGES = "D:\\mnist\\t10k-images.idx3-ubyte";
constexpr const char* TEST_LABELS = "D:\\mnist\\t10k-labels.idx1-ubyte";

constexpr int   EPOCHS = 6;
constexpr float LR = 0.01f;
constexpr int    BATCH_SIZE = 32;   // taille du mini-lot


int main() {
    try {
        Images Xtr = load_images(TRAIN_IMAGES);
        Labels Ytr = load_labels(TRAIN_LABELS);
        Images Xte = load_images(TEST_IMAGES);
        Labels Yte = load_labels(TEST_LABELS);

        std::mt19937 gen(42);
        CNN net(LR, gen);
        train_epoch_loop(net, Xtr, Ytr, Xte, Yte, EPOCHS, BATCH_SIZE);

    }
    catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
