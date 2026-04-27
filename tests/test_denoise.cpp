// Tests del Ladrillo 22 — denoise / reduccion de entropia.

#include "test_framework.hpp"
#include "easyatom/denoise/entropy.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"

#include <cmath>
#include <stdexcept>

using easyatom::denoise::denoise;
using easyatom::denoise::DenoiseResult;
using easyatom::denoise::gram_schmidt;
using easyatom::denoise::inner;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;
using easyatom::ops::random_phase_state;
using easyatom::ops::bundle;

EATEST_CASE(denoise_gs_produce_base_ortonormal) {
    const std::size_t D = 256;
    State a = random_phase_state(D, 1);
    State b = random_phase_state(D, 2);
    State c = random_phase_state(D, 3);
    auto basis = gram_schmidt({a, b, c});
    EATEST_REQUIRE(basis.size() == 3);
    for (std::size_t i = 0; i < basis.size(); ++i) {
        const double n2 = basis[i].norm_squared();
        EATEST_REQUIRE(std::abs(n2 - 1.0) < 1e-9);
        for (std::size_t j = i + 1; j < basis.size(); ++j) {
            const Complex ip = inner(basis[i], basis[j]);
            EATEST_REQUIRE(std::abs(ip) < 1e-9);
        }
    }
}

EATEST_CASE(denoise_gs_descarta_dependientes) {
    const std::size_t D = 64;
    State a = random_phase_state(D, 1);
    // a y 2*a son dependientes; la base debe quedar con uno solo.
    State a2(D);
    for (std::size_t i = 0; i < D; ++i) a2[i] = a[i] * Complex{2.0, 0.0};
    auto basis = gram_schmidt({a, a2});
    EATEST_REQUIRE(basis.size() == 1);
}

EATEST_CASE(denoise_proyeccion_recupera_input_si_esta_en_subespacio) {
    const std::size_t D = 256;
    State a = random_phase_state(D, 1);
    // s = a esta exactamente en el subespacio generado por {a}.
    DenoiseResult r = denoise(a, {a}, 0.0);
    EATEST_REQUIRE(r.basis_size == 1);
    EATEST_REQUIRE(r.kept == 1);
    EATEST_REQUIRE(r.residual_norm2 < 1e-12);
    EATEST_REQUIRE(fidelity(r.filtered, a) > 0.9999);
}

EATEST_CASE(denoise_residual_no_cero_si_state_fuera_del_subespacio) {
    const std::size_t D = 512;
    State a = random_phase_state(D, 1);
    State b = random_phase_state(D, 2);   // ortogonal en alta D a a
    DenoiseResult r = denoise(b, {a}, 0.0);
    EATEST_REQUIRE(r.residual_norm2 > 1e-3);   // hay informacion fuera del span
}

EATEST_CASE(denoise_umbral_recorta_componentes_pequenas) {
    const std::size_t D = 256;
    State a = random_phase_state(D, 1);
    State b = random_phase_state(D, 2);
    // s mezcla a (peso fuerte) + b (peso debil).
    State s = bundle({a, b}, {Complex{1.0, 0.0}, Complex{0.001, 0.0}});
    DenoiseResult r = denoise(s, {a, b}, 0.5);
    EATEST_REQUIRE(r.basis_size == 2);
    EATEST_REQUIRE(r.kept == 1);             // solo se queda con la fuerte
}

EATEST_CASE(denoise_reduce_o_iguala_entropia) {
    const std::size_t D = 256;
    State a = random_phase_state(D, 1);
    State b = random_phase_state(D, 2);
    State c = random_phase_state(D, 3);
    State s = bundle({a, b, c},
        {Complex{1.0,0.0}, Complex{0.05,0.0}, Complex{0.02,0.0}});
    DenoiseResult r = denoise(s, {a, b, c}, 0.1);
    EATEST_REQUIRE(r.post_entropy <= r.pre_entropy + 1e-12);
}

EATEST_CASE(denoise_anchors_vacios_lanza) {
    State a = random_phase_state(32, 1);
    bool t = false;
    try { (void)denoise(a, {}, 0.0); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(denoise_theta_negativo_lanza) {
    State a = random_phase_state(32, 1);
    bool t = false;
    try { (void)denoise(a, {a}, -0.1); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}
