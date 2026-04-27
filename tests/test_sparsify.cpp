// =============================================================================
// tests/test_sparsify.cpp  --  L25
// =============================================================================

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "test_framework.hpp"
#include "easyatom/denoise/sparsify.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"

using easyatom::denoise::concentration;
using easyatom::denoise::denoise_sparse;
using easyatom::denoise::effective_rank;
using easyatom::denoise::quantile_theta;
using easyatom::denoise::sparsify_to_top_k;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;
using easyatom::ops::random_phase_state;

EATEST_CASE(sparsify_quantile_theta_keep_ratio_correcto) {
    std::vector<Complex> c = {{4,0},{3,0},{2,0},{1,0}};
    // keep_ratio=0.5 -> keep=2 -> theta=mags[1]=3.0
    EATEST_REQUIRE(std::abs(quantile_theta(c, 0.5) - 3.0) < 1e-12);
    // keep_ratio=1.0 -> keep=4 -> theta=mags[3]=1.0
    EATEST_REQUIRE(std::abs(quantile_theta(c, 1.0) - 1.0) < 1e-12);
}

EATEST_CASE(sparsify_quantile_theta_redondea_arriba) {
    std::vector<Complex> c = {{4,0},{3,0},{2,0}};
    // 0.34*3=1.02 -> ceil=2 -> theta=3.0
    EATEST_REQUIRE(std::abs(quantile_theta(c, 0.34) - 3.0) < 1e-12);
}

EATEST_CASE(sparsify_quantile_theta_validaciones) {
    bool t1=false; try { (void)quantile_theta({}, 0.5); }
                   catch (const std::invalid_argument&) { t1=true; }
    bool t2=false; try { (void)quantile_theta({{1,0}}, 0.0); }
                   catch (const std::invalid_argument&) { t2=true; }
    bool t3=false; try { (void)quantile_theta({{1,0}}, 1.5); }
                   catch (const std::invalid_argument&) { t3=true; }
    EATEST_REQUIRE(t1 && t2 && t3);
}

EATEST_CASE(sparsify_effective_rank_y_concentration) {
    // 1 sola direccion -> concentration=0
    std::vector<Complex> single = {{1,0},{0,0},{0,0},{0,0}};
    EATEST_REQUIRE(effective_rank(single) == 1);
    EATEST_REQUIRE(concentration(single) < 1e-9);

    // uniforme -> concentration=1
    std::vector<Complex> uni = {{1,0},{1,0},{1,0},{1,0}};
    EATEST_REQUIRE(effective_rank(uni) == 4);
    EATEST_REQUIRE(std::abs(concentration(uni) - 1.0) < 1e-9);
}

EATEST_CASE(sparsify_denoise_sparse_keep_ratio_05_conserva_mitad) {
    State a1 = random_phase_state(64, 1);
    State a2 = random_phase_state(64, 2);
    State a3 = random_phase_state(64, 3);
    State a4 = random_phase_state(64, 4);
    State s  = random_phase_state(64, 99);
    auto r = denoise_sparse(s, {a1, a2, a3, a4}, 0.5);
    EATEST_REQUIRE(r.basis_size == 4);
    EATEST_REQUIRE(r.kept == 2);
    EATEST_REQUIRE(r.post_entropy <= r.pre_entropy + 1e-12);
    EATEST_REQUIRE(r.post_concentration <= r.pre_concentration + 1e-12);
}

EATEST_CASE(sparsify_top_k_exacto) {
    State a1 = random_phase_state(64, 11);
    State a2 = random_phase_state(64, 22);
    State a3 = random_phase_state(64, 33);
    State s  = random_phase_state(64, 100);
    auto r = sparsify_to_top_k(s, {a1, a2, a3}, 1);
    EATEST_REQUIRE(r.kept == 1);
    // post_concentration de un unico modo debe ser 0.
    EATEST_REQUIRE(r.post_concentration < 1e-9);
}

EATEST_CASE(sparsify_top_k_clip_a_basis_size) {
    State a1 = random_phase_state(32, 1);
    State a2 = random_phase_state(32, 2);
    State s  = random_phase_state(32, 50);
    // pido k=10 pero base solo tiene 2 -> kept=2.
    auto r = sparsify_to_top_k(s, {a1, a2}, 10);
    EATEST_REQUIRE(r.kept == 2);
}

EATEST_CASE(sparsify_validaciones_lanzan) {
    State s = random_phase_state(32, 1);
    bool t1=false; try { (void)denoise_sparse(s, {}, 0.5); }
                   catch (const std::invalid_argument&) { t1=true; }
    bool t2=false; try { (void)denoise_sparse(s, {s}, 0.0); }
                   catch (const std::invalid_argument&) { t2=true; }
    bool t3=false; try { (void)sparsify_to_top_k(s, {s}, 0); }
                   catch (const std::invalid_argument&) { t3=true; }
    EATEST_REQUIRE(t1 && t2 && t3);
}
