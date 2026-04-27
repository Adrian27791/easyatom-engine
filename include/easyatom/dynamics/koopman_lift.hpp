// =============================================================================
// easyatom/dynamics/koopman_lift.hpp  --  L28
//
// Lift de Koopman DIAGONAL POR MODO sobre estados de Hilbert + extrapolacion.
//
// Diferente a easyatom/dynamics/koopman.hpp (que es EDMD afin sobre R^d).
// Este L28 trabaja sobre easyatom::hilbert::State, proyectando a una base
// anchor ortonormalizada y ajustando un operador K diagonal por modo:
//
//     c_{t,m} = <e_m | s_t>
//     c_{t+1,m} ≈ k_m * c_{t,m}
//     k_m      = sum_t conj(c_{t,m}) * c_{t+1,m}  /  (sum_t |c_{t,m}|² + lambda)
//
// Sin BLAS, sin inversa matricial: O(T*M).
// =============================================================================

#ifndef EASYATOM_DYNAMICS_KOOPMAN_LIFT_HPP
#define EASYATOM_DYNAMICS_KOOPMAN_LIFT_HPP

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "easyatom/denoise/entropy.hpp"
#include "easyatom/hilbert/state.hpp"

namespace easyatom::dynamics::lift {

using easyatom::hilbert::Complex;
using easyatom::hilbert::State;
using easyatom::hilbert::inner;

struct KoopmanModel {
    std::vector<State>   anchors;
    std::vector<Complex> k_diag;
    std::size_t          M = 0;
};

[[nodiscard]] inline KoopmanModel koopman_lift(
    const std::vector<State>& seq,
    const std::vector<State>& anchors,
    double                    lambda = 1e-6) {
    if (seq.size() < 2)
        throw std::invalid_argument("koopman_lift: seq necesita >= 2 estados.");
    if (anchors.empty())
        throw std::invalid_argument("koopman_lift: anchors vacios.");
    if (lambda < 0.0)
        throw std::invalid_argument("koopman_lift: lambda < 0.");

    const std::size_t D = seq.front().dim();
    for (const auto& s : seq) {
        if (s.dim() != D)
            throw std::invalid_argument("koopman_lift: dims heterogeneas.");
    }

    KoopmanModel K;
    K.anchors = easyatom::denoise::gram_schmidt(anchors);
    K.M       = K.anchors.size();
    K.k_diag.assign(K.M, Complex{0.0, 0.0});

    const std::size_t T = seq.size();
    std::vector<std::vector<Complex>> C(T, std::vector<Complex>(K.M));
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t m = 0; m < K.M; ++m) {
            C[t][m] = inner(K.anchors[m], seq[t]);
        }
    }

    for (std::size_t m = 0; m < K.M; ++m) {
        Complex num{0.0, 0.0};
        double  den = lambda;
        for (std::size_t t = 0; t + 1 < T; ++t) {
            num += std::conj(C[t][m]) * C[t + 1][m];
            den += std::norm(C[t][m]);
        }
        K.k_diag[m] = num / den;
    }
    return K;
}

[[nodiscard]] inline State predict_next(
    const KoopmanModel& K, const State& s) {
    if (K.M == 0 || K.anchors.empty())
        throw std::invalid_argument("predict_next: modelo vacio.");
    if (s.dim() != K.anchors.front().dim())
        throw std::invalid_argument("predict_next: dim incompatible.");

    const std::size_t D = s.dim();
    std::vector<Complex> coefs(K.M);
    for (std::size_t m = 0; m < K.M; ++m) {
        coefs[m] = inner(K.anchors[m], s) * K.k_diag[m];
    }
    State out(D);
    for (std::size_t m = 0; m < K.M; ++m) {
        const auto& a_e = K.anchors[m].amplitudes();
        const auto  cm  = coefs[m];
        for (std::size_t i = 0; i < D; ++i)
            out[i] = out[i] + cm * a_e[i];
    }
    return out;
}

struct CrossValidation {
    double      mean_residual2 = 0.0;
    std::size_t holdout_count  = 0;
};

[[nodiscard]] inline CrossValidation cv_tail(
    const std::vector<State>& seq,
    const std::vector<State>& anchors,
    std::size_t               holdout,
    double                    lambda = 1e-6) {
    if (holdout == 0)
        throw std::invalid_argument("cv_tail: holdout = 0.");
    if (seq.size() < holdout + 2)
        throw std::invalid_argument(
            "cv_tail: seq demasiado corto para holdout requerido.");

    const std::size_t cut = seq.size() - holdout;
    std::vector<State> train(seq.begin(), seq.begin() + cut);
    auto K = koopman_lift(train, anchors, lambda);

    double acc = 0.0;
    for (std::size_t t = cut; t < seq.size(); ++t) {
        const State pred = predict_next(K, seq[t - 1]);
        const auto& a_p  = pred.amplitudes();
        const auto& a_g  = seq[t].amplitudes();
        for (std::size_t i = 0; i < pred.dim(); ++i)
            acc += std::norm(a_p[i] - a_g[i]);
    }
    CrossValidation cv;
    cv.holdout_count  = holdout;
    cv.mean_residual2 = acc / static_cast<double>(holdout);
    return cv;
}

}  // namespace easyatom::dynamics::lift

#endif  // EASYATOM_DYNAMICS_KOOPMAN_LIFT_HPP
