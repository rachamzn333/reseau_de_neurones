#include "layers.h"
#include <algorithm>
#include <cmath>
#include <numeric>

/* ───────── ConvLayer ─────────────────────────────────────────── */
ConvLayer::ConvLayer(int inC, int outC, int k, std::mt19937& g)
    : inC_(inC), outC_(outC), k_(k)
{
    std::uniform_real_distribution<float> D(-0.05f, 0.05f);
    W_.resize(outC_ * inC_ * k_ * k_);  b_.resize(outC_);
    for (float& w : W_) w = D(g);

    dW_.resize(W_.size()); db_.resize(b_.size());
    gW_.assign(W_.size(), 0.f); gb_.assign(b_.size(), 0.f);
}

int ConvLayer::idx(int c, int y, int x, int C, int H, int W) const {
    return c * H * W + y * W + x;
}

Tensor ConvLayer::forward(const Tensor& in) {
    cache_ = in;
    const int H = IMG_SIZE;
    Tensor out(outC_ * H * H);

    for (int oc = 0; oc < outC_; ++oc) {                 /* PARALLEL_CANDIDATE_OpenMP */
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < H; ++x) {
                float sum = b_[oc];
                for (int ic = 0; ic < inC_; ++ic)
                    for (int ky = -1; ky <= 1; ++ky)
                        for (int kx = -1; kx <= 1; ++kx) {
                            int iy = y + ky, ix = x + kx;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= H) continue;
                            int wi = (((oc * inC_ + ic) * k_ + (ky + 1)) * k_ + (kx + 1));
                            sum += in[idx(ic, iy, ix, inC_, H, H)] * W_[wi];
                        }
                out[idx(oc, y, x, outC_, H, H)] = sum;
            }
    }
    return out;
}

Tensor ConvLayer::backward(const Tensor& g) {           // ← plus de lr
    const int H = IMG_SIZE;
    std::fill(dW_.begin(), dW_.end(), 0.f);
    std::fill(db_.begin(), db_.end(), 0.f);
    Tensor dx(cache_.size(), 0.f);

    for (int oc = 0; oc < outC_; ++oc) {                 /* PARALLEL_CANDIDATE_OpenMP */
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < H; ++x) {
                float grad = g[idx(oc, y, x, outC_, H, H)];
                db_[oc] += grad;
                for (int ic = 0; ic < inC_; ++ic)
                    for (int ky = -1; ky <= 1; ++ky)
                        for (int kx = -1; kx <= 1; ++kx) {
                            int iy = y + ky, ix = x + kx;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= H) continue;
                            int wi = (((oc * inC_ + ic) * k_ + (ky + 1)) * k_ + (kx + 1));
                            dW_[wi] += cache_[idx(ic, iy, ix, inC_, H, H)] * grad;
                            dx[idx(ic, iy, ix, inC_, H, H)] += W_[wi] * grad;
                        }
            }
    }
    /* cumul pour le mini-lot */
    std::transform(gW_.begin(), gW_.end(), dW_.begin(), gW_.begin(), std::plus<float>());
    std::transform(gb_.begin(), gb_.end(), db_.begin(), gb_.begin(), std::plus<float>());
    return dx;
}

void ConvLayer::apply_gradients(int batch_sz, float lr)
{
    const float inv = lr / batch_sz;
    for (size_t i = 0; i < W_.size(); ++i) {
        W_[i] -= inv * gW_[i];
        gW_[i] = 0.f;
    }
    for (size_t i = 0; i < b_.size(); ++i) {
        b_[i] -= inv * gb_[i];
        gb_[i] = 0.f;
    }
}

/* ───────── ReLU ─────────────────────────────────────────────── */
Tensor ReLU::forward(const Tensor& in) {
    cache_ = in;
    Tensor y(in.size());
    for (size_t i = 0; i < in.size(); ++i) y[i] = in[i] > 0 ? in[i] : 0;
    return y;
}

Tensor ReLU::backward(const Tensor& g) {
    Tensor dx(g.size());
    for (size_t i = 0; i < g.size(); ++i) dx[i] = cache_[i] > 0 ? g[i] : 0;
    return dx;
}

/* ───────── MaxPool ─────────────────────────────────────────── */
int MaxPool::idx(int c, int y, int x, int C, int H, int W) const {
    return c * H * W + y * W + x;
}

Tensor MaxPool::forward(const Tensor& in) {
    C_ = static_cast<int>(in.size()) / (IMG_SIZE * IMG_SIZE);
    H_ = IMG_SIZE / 2; W_ = H_;
    Tensor out(C_ * H_ * W_);
    argmax_.clear(); argmax_.reserve(out.size());

    for (int c = 0; c < C_; ++c)                              /* PARALLEL_CANDIDATE_OpenMP */
        for (int y = 0; y < H_; ++y)
            for (int x = 0; x < W_; ++x) {
                float best = -1e9f; int best_i = 0;
                for (int py = 0; py < 2; ++py)
                    for (int px = 0; px < 2; ++px) {
                        int iy = y * 2 + py, ix = x * 2 + px;
                        int i = idx(c, iy, ix, C_, IMG_SIZE, IMG_SIZE);
                        if (in[i] > best) { best = in[i]; best_i = i; }
                    }
                out[idx(c, y, x, C_, H_, W_)] = best;
                argmax_.push_back(best_i);
            }
    return out;
}

Tensor MaxPool::backward(const Tensor& g) {
    Tensor dx(C_ * IMG_SIZE * IMG_SIZE, 0.f);
    for (size_t i = 0; i < argmax_.size(); ++i) dx[argmax_[i]] = g[i];
    return dx;
}

/* ───────── Dense ───────────────────────────────────────────── */
Dense::Dense(int inD, int outD, std::mt19937& g)
    : inD_(inD), outD_(outD)
{
    std::uniform_real_distribution<float> D(-0.05f, 0.05f);
    W_.resize(inD_ * outD_); for (float& w : W_) w = D(g);
    b_.resize(outD_);

    dW_.resize(W_.size()); db_.resize(b_.size());
    gW_.assign(W_.size(), 0.f); gb_.assign(b_.size(), 0.f);
}

Tensor Dense::forward(const Tensor& in) {
    cache_ = in;
    Tensor y(outD_);
    for (int o = 0; o < outD_; ++o) {                       /* PARALLEL_CANDIDATE_OpenMP */
        float s = b_[o];
        for (int i = 0; i < inD_; ++i) s += in[i] * W_[o * inD_ + i];
        y[o] = s;
    }
    return y;
}

Tensor Dense::backward(const Tensor& g) {                  // ← plus de lr
    std::fill(dW_.begin(), dW_.end(), 0.f);
    std::fill(db_.begin(), db_.end(), 0.f);
    Tensor dx(inD_, 0.f);

    for (int o = 0; o < outD_; ++o) {                       /* PARALLEL_CANDIDATE_OpenMP */
        db_[o] += g[o];
        for (int i = 0; i < inD_; ++i) {
            dW_[o * inD_ + i] += cache_[i] * g[o];
            dx[i] += W_[o * inD_ + i] * g[o];
        }
    }
    /* cumul du mini-lot */
    std::transform(gW_.begin(), gW_.end(), dW_.begin(), gW_.begin(), std::plus<float>());
    std::transform(gb_.begin(), gb_.end(), db_.begin(), gb_.begin(), std::plus<float>());
    return dx;
}

void Dense::apply_gradients(int batch_sz, float lr)
{
    const float inv = lr / batch_sz;
    for (size_t i = 0; i < W_.size(); ++i) {
        W_[i] -= inv * gW_[i];
        gW_[i] = 0.f;
    }
    for (size_t i = 0; i < b_.size(); ++i) {
        b_[i] -= inv * gb_[i];
        gb_[i] = 0.f;
    }
}
