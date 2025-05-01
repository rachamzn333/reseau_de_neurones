#include "layers.h"
#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef _OPENMP          // ajout : en-tête OpenMP
  #include <omp.h>
#endif

/* ───────── ConvLayer ─────────────────────────────────────────── */
ConvLayer::ConvLayer(int inC, int outC, int k, std::mt19937& g)
    : inC_(inC), outC_(outC), k_(k)
{
    std::uniform_real_distribution<float> D(-0.05f, 0.05f);
    W_.resize(outC_ * inC_ * k_ * k_);
    b_.resize(outC_);
    for (float& w : W_) w = D(g);

    dW_.resize(W_.size()); db_.resize(b_.size());
    gW_.assign(W_.size(), 0.f); gb_.assign(b_.size(), 0.f);
}

int ConvLayer::idx(int c, int y, int x, int C, int H, int W) const
{
    return c * H * W + y * W + x;
}

/* ---------- forward (parallélisé) ---------- */
Tensor ConvLayer::forward(const Tensor& in)
{
    cache_ = in;
    const int H = IMG_SIZE;
    Tensor out(outC_ * H * H);

#pragma omp parallel for collapse(2) schedule(static)
    for (int oc = 0; oc < outC_; ++oc) {
        for (int y = 0; y < H; ++y) {
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
    }
    return out;
}

/* ---------- backward (toujours séquentiel) ---------- */
Tensor ConvLayer::backward(const Tensor& g)            /* PARALLEL_CANDIDATE_OpenMP */
{
    const int H = IMG_SIZE;
    std::fill(dW_.begin(), dW_.end(), 0.f);
    std::fill(db_.begin(), db_.end(), 0.f);
    Tensor dx(cache_.size(), 0.f);

    for (int oc = 0; oc < outC_; ++oc) {               /* PARALLEL_CANDIDATE_OpenMP */
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
    std::transform(gW_.begin(), gW_.end(), dW_.begin(),
                   gW_.begin(), std::plus<float>());
    std::transform(gb_.begin(), gb_.end(), db_.begin(),
                   gb_.begin(), std::plus<float>());
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
Tensor ReLU::forward(const Tensor& in)
{
    cache_ = in;
    Tensor y(in.size());

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < in.size(); ++i)
        y[i] = in[i] > 0.f ? in[i] : 0.f;

    return y;
}

Tensor ReLU::backward(const Tensor& g)                 /* PARALLEL_CANDIDATE_OpenMP */
{
    Tensor dx(g.size());
    for (std::size_t i = 0; i < g.size(); ++i)
        dx[i] = cache_[i] > 0.f ? g[i] : 0.f;
    return dx;
}

/* ───────── MaxPool ─────────────────────────────────────────── */
int MaxPool::idx(int c, int y, int x, int C, int H, int W) const
{
    return c * H * W + y * W + x;
}

Tensor MaxPool::forward(const Tensor& in)
{
    C_ = static_cast<int>(in.size()) / (IMG_SIZE * IMG_SIZE);
    H_ = IMG_SIZE / 2;
    W_ = H_;

    Tensor out(C_ * H_ * W_);

    /* --- on pré-alloue argmax_ pour éviter les push_back concurrents --- */
    argmax_.assign(out.size(), 0);

    /* Chaque couple (c, y, x) est indépendant ; on peut donc
       paralléliser les trois boucles imbriquées. */
#pragma omp parallel for collapse(3) schedule(static)
    for (int c = 0; c < C_; ++c)
        for (int y = 0; y < H_; ++y)
            for (int x = 0; x < W_; ++x)
            {
                float best   = -1e9f;
                int   best_i = 0;

                /* balayage 2 × 2 */
                for (int py = 0; py < 2; ++py)
                    for (int px = 0; px < 2; ++px) {
                        int iy = y * 2 + py,
                            ix = x * 2 + px;
                        int i = idx(c, iy, ix, C_, IMG_SIZE, IMG_SIZE);
                        if (in[i] > best) { best = in[i]; best_i = i; }
                    }

                std::size_t out_idx = idx(c, y, x, C_, H_, W_);
                out[out_idx]   = best;
                argmax_[out_idx] = best_i;          // accès unique, thread-safe
            }

    return out;
}


Tensor MaxPool::backward(const Tensor& g)              /* PARALLEL_CANDIDATE_OpenMP */
{
    Tensor dx(C_ * IMG_SIZE * IMG_SIZE, 0.f);
    for (std::size_t i = 0; i < argmax_.size(); ++i)
        dx[argmax_[i]] = g[i];
    return dx;
}

/* ───────── Dense ───────────────────────────────────────────── */
Dense::Dense(int inD, int outD, std::mt19937& g)
    : inD_(inD), outD_(outD)
{
    std::uniform_real_distribution<float> D(-0.05f, 0.05f);
    W_.resize(inD_ * outD_);
    for (float& w : W_) w = D(g);
    b_.resize(outD_);

    dW_.resize(W_.size()); db_.resize(b_.size());
    gW_.assign(W_.size(), 0.f); gb_.assign(b_.size(), 0.f);
}

/* ---------- forward (parallélisé) ---------- */
Tensor Dense::forward(const Tensor& in)
{
    cache_ = in;
    Tensor y(outD_);

#pragma omp parallel for schedule(static)
    for (int o = 0; o < outD_; ++o) {
        float s = b_[o];
        for (int i = 0; i < inD_; ++i)
            s += in[i] * W_[o * inD_ + i];
        y[o] = s;
    }
    return y;
}

/* ---------- backward (séquentiel) ---------- */
Tensor Dense::backward(const Tensor& g)                /* PARALLEL_CANDIDATE_OpenMP */
{
    std::fill(dW_.begin(), dW_.end(), 0.f);
    std::fill(db_.begin(), db_.end(), 0.f);
    Tensor dx(inD_, 0.f);

    for (int o = 0; o < outD_; ++o) {                   /* PARALLEL_CANDIDATE_OpenMP */
        db_[o] += g[o];
        for (int i = 0; i < inD_; ++i) {
            dW_[o * inD_ + i] += cache_[i] * g[o];
            dx[i] += W_[o * inD_ + i] * g[o];
        }
    }
    std::transform(gW_.begin(), gW_.end(), dW_.begin(),
                   gW_.begin(), std::plus<float>());
    std::transform(gb_.begin(), gb_.end(), db_.begin(),
                   gb_.begin(), std::plus<float>());
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
