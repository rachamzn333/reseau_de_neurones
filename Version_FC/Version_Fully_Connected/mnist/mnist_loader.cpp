#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>

static uint32_t swap_endian(uint32_t v) {
    return  (v >> 24) |
        ((v >> 8) & 0xFF00) |
        ((v << 8) & 0xFF0000) |
        (v << 24);
}
//load images 
Images load_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    uint32_t m, n, r, c;
    f.read((char*)&m, 4); f.read((char*)&n, 4);
    f.read((char*)&r, 4); f.read((char*)&c, 4);
    m = swap_endian(m); n = swap_endian(n);
    r = swap_endian(r); c = swap_endian(c);
    if (m != 2051 || r != 28 || c != 28) throw std::runtime_error("bad image file");

    Images imgs(n, Tensor(r * c));
    for (uint32_t i = 0; i < n; ++i) {
        std::vector<unsigned char> buf(r * c);
        f.read((char*)buf.data(), r * c);
        for (size_t j = 0; j < buf.size(); ++j) imgs[i][j] = buf[j] / 255.0f;
    }
    return imgs;
}
Labels load_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    uint32_t m, n; f.read((char*)&m, 4); f.read((char*)&n, 4);
    m = swap_endian(m); n = swap_endian(n);
    if (m != 2049) throw std::runtime_error("bad label file");
    Labels lbl(n); f.read((char*)lbl.data(), n);
    return lbl;
}
