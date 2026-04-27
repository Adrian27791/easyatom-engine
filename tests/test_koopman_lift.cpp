// =============================================================================
// tests/test_koopman_lift.cpp  --  L28
// =============================================================================

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "test_framework.hpp"
#include "easyatom/dynamics/koopman_lift.hpp"
#include "easyatom/hilbert/state.hpp"

using easyatom::dynamics::lift::CrossValidation;
using easyatom::dynamics::lift::cv_tail;
using easyatom::dynamics::lift::koopman_lift;
using easyatom::dynamics::lift::KoopmanModel;
using easyatom::dynamics::lift::predict_next;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;

// Construye una secuencia donde s_t = sum_m (k_m^t * c0_m) * |m>.
static std::vector<State> make_seq(std::size_t D,
                                   const std::vector<Complex>& k_truth,
                                   const std::vector<Complex>& c0,
                                   std::size_t T) {
    std::vector<State> seq;
    seq.reserve(T);
    for (std::size_t t = 0; t < T; ++t) {
        std::vector<Complex> a(D, Complex{0.0, 0.0});
        for (std::size_t m = 0; m < k_truth.size(); ++m) {
            Complex coef = c0[m];
            for (std::size_t r = 0; r < t; ++r) coef *= k_truth[m];
            a[m] = coef;   // m-esima base canonica
        }
        seq.emplace_back(std::move(a));
    }
    return seq;
}

EATEST_CASE(koopman_lift_recupera_k_diagonal_real) {
    const std::size_t D = 8;
    std::vector<Complex> k_truth = {Complex{0.9, 0.0}, Complex{0.5, 0.0}};
    std::vector<Complex> c0      = {Complex{1.0, 0.0}, Complex{2.0, 0.0}};
    auto seq = make_seq(D, k_truth, c0, 6);

    std::vector<State> anchors = {State::basis(D, 0), State::basis(D, 1)};
    auto K = koopman_lift(seq, anchors, 0.0);
    EATEST_REQUIRE(K.M == 2);
    EATEST_REQUIRE(std::abs(K.k_diag[0] - k_truth[0]) < 1e-9);
    EATEST_REQUIRE(std::abs(K.k_diag[1] - k_truth[1]) < 1e-9);
}

EATEST_CASE(koopman_lift_recupera_k_diagonal_complejo) {
    const std::size_t D = 8;
    std::vector<Complex> k_truth = {Complex{0.7, 0.3}};
    std::vector<Complex> c0      = {Complex{1.0, 0.0}};
    auto seq = make_seq(D, k_truth, c0, 8);
    std::vector<State> anchors = {State::basis(D, 0)};
    auto K = koopman_lift(seq, anchors, 0.0);
    EATEST_REQUIRE(std::abs(K.k_diag[0] - k_truth[0]) < 1e-9);
}

EATEST_CASE(koopman_predict_next_exacto_en_modelo_ideal) {
    const std::size_t D = 8;
    std::vector<Complex> k_truth = {Complex{0.9, 0.0}, Complex{-0.4, 0.0}};
    std::vector<Complex> c0      = {Complex{1.0, 0.0}, Complex{0.5, 0.0}};
    auto seq = make_seq(D, k_truth, c0, 5);
    std::vector<State> anchors = {State::basis(D, 0), State::basis(D, 1)};
    auto K = koopman_lift(seq, anchors, 0.0);

    State pred = predict_next(K, seq[3]);   // deberia ≈ seq[4]
    const auto& a_p = pred.amplitudes();
    const auto& a_g = seq[4].amplitudes();
    double r2 = 0.0;
    for (std::size_t i = 0; i < D; ++i) r2 += std::norm(a_p[i] - a_g[i]);
    EATEST_REQUIRE(r2 < 1e-12);
}

EATEST_CASE(koopman_cv_tail_residual_pequeno) {
    const std::size_t D = 8;
    std::vector<Complex> k_truth = {Complex{0.95, 0.0}, Complex{0.4, 0.1}};
    std::vector<Complex> c0      = {Complex{1.0, 0.0}, Complex{1.0, 0.0}};
    auto seq = make_seq(D, k_truth, c0, 8);
    std::vector<State> anchors = {State::basis(D, 0), State::basis(D, 1)};
    auto cv = cv_tail(seq, anchors, 2, 0.0);
    EATEST_REQUIRE(cv.holdout_count == 2);
    EATEST_REQUIRE(cv.mean_residual2 < 1e-12);
}

EATEST_CASE(koopman_lift_seq_corta_lanza) {
    bool t = false;
    try {
        std::vector<State> seq = {State::basis(4, 0)};
        std::vector<State> anchors = {State::basis(4, 0)};
        (void)koopman_lift(seq, anchors);
    } catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(koopman_lift_anchors_vacios_lanza) {
    bool t = false;
    try {
        std::vector<State> seq = {State::basis(4, 0), State::basis(4, 1)};
        (void)koopman_lift(seq, {});
    } catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(koopman_predict_next_dim_incompatible_lanza) {
    const std::size_t D = 4;
    std::vector<Complex> k_truth = {Complex{0.5, 0.0}};
    std::vector<Complex> c0      = {Complex{1.0, 0.0}};
    auto seq = make_seq(D, k_truth, c0, 3);
    std::vector<State> anchors = {State::basis(D, 0)};
    auto K = koopman_lift(seq, anchors);
    bool t = false;
    try { (void)predict_next(K, State::basis(8, 0)); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(koopman_cv_tail_holdout_invalido_lanza) {
    auto seq = make_seq(4, {Complex{0.5,0.0}}, {Complex{1.0,0.0}}, 3);
    std::vector<State> anchors = {State::basis(4, 0)};
    bool t1 = false;
    try { (void)cv_tail(seq, anchors, 0); }
    catch (const std::invalid_argument&) { t1 = true; }
    bool t2 = false;
    try { (void)cv_tail(seq, anchors, 5); }
    catch (const std::invalid_argument&) { t2 = true; }
    EATEST_REQUIRE(t1 && t2);
}
