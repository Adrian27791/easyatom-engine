// =============================================================================
// tests/test_modern_hopfield.cpp  --  L35
// =============================================================================

#include <cmath>
#include <stdexcept>
#include <vector>

#include "easyatom/dynamics/modern_hopfield.hpp"
#include "test_framework.hpp"

using easyatom::dynamics::modern::HopfieldConfig;
using easyatom::dynamics::modern::HopfieldResult;
using easyatom::dynamics::modern::energy;
using easyatom::dynamics::modern::run;
using easyatom::dynamics::modern::step;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;

static State e_i(std::size_t d, std::size_t i, double a = 1.0) {
    State s(d);
    s[i] = Complex{a, 0.0};
    return s;
}

EATEST_CASE(mh_step_devuelve_memoria_mas_cercana) {
    auto m0 = e_i(8, 0);
    auto m1 = e_i(8, 1);
    auto m2 = e_i(8, 2);
    State xi(8);
    xi[0] = Complex{0.9, 0.0};
    xi[1] = Complex{0.1, 0.0};
    auto out = step(xi, {m0, m1, m2}, 16.0);
    EATEST_REQUIRE(fidelity(out, m0) > 0.99);
}

EATEST_CASE(mh_run_converge_a_atractor_unico) {
    auto m0 = e_i(8, 0);
    auto m1 = e_i(8, 3);
    State xi(8);
    xi[0] = Complex{0.7, 0.0};
    xi[3] = Complex{0.3, 0.0};
    HopfieldConfig cfg; cfg.beta = 32.0;
    auto r = run(xi, {m0, m1}, cfg);
    EATEST_REQUIRE(r.converged);
    EATEST_REQUIRE(fidelity(r.xi, m0) > 0.99);
}

EATEST_CASE(mh_energy_decrece_monotonicamente) {
    auto m0 = e_i(8, 0);
    auto m1 = e_i(8, 1);
    State xi(8);
    xi[0] = Complex{0.6, 0.0};
    xi[1] = Complex{0.4, 0.0};
    const double e0 = energy(xi, {m0, m1}, 8.0);
    State xi1 = step(xi, {m0, m1}, 8.0);
    const double e1 = energy(xi1, {m0, m1}, 8.0);
    State xi2 = step(xi1, {m0, m1}, 8.0);
    const double e2 = energy(xi2, {m0, m1}, 8.0);
    EATEST_REQUIRE(e1 <= e0 + 1e-9);
    EATEST_REQUIRE(e2 <= e1 + 1e-9);
}

EATEST_CASE(mh_alta_temperatura_promedia_memorias) {
    // beta -> 0 hace softmax uniforme; el atractor es la media de memorias.
    auto m0 = e_i(8, 0);
    auto m1 = e_i(8, 1);
    State xi(8);
    xi[0] = Complex{0.5, 0.0};
    xi[1] = Complex{0.5, 0.0};
    auto out = step(xi, {m0, m1}, 0.001);  // beta muy bajo
    // Esperamos cuasi-promedio: out[0] ~ out[1] ~ 0.5.
    EATEST_REQUIRE(std::abs(out[0].real() - 0.5) < 1e-3);
    EATEST_REQUIRE(std::abs(out[1].real() - 0.5) < 1e-3);
}

EATEST_CASE(mh_capacidad_distingue_muchas_memorias) {
    // En H_D, Hopfield moderno con beta alto puede distinguir D memorias
    // ortogonales sin colision. Probamos D=16, M=16 todas ortogonales.
    const std::size_t D = 16;
    std::vector<State> mems;
    for (std::size_t i = 0; i < D; ++i) mems.push_back(e_i(D, i));
    // query muy cercana a la memoria #7
    State xi = e_i(D, 7, 0.95);
    xi[0] = Complex{0.05, 0.0};
    HopfieldConfig cfg; cfg.beta = 32.0;
    auto r = run(xi, mems, cfg);
    EATEST_REQUIRE(fidelity(r.xi, mems[7]) > 0.99);
}

EATEST_CASE(mh_memorias_vacias_lanza) {
    State xi = e_i(4, 0);
    bool t = false;
    try { (void)step(xi, {}, 4.0); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(mh_beta_no_positivo_lanza) {
    State xi = e_i(4, 0);
    auto m  = e_i(4, 1);
    bool t = false;
    try { (void)step(xi, {m}, 0.0); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(mh_dim_mismatch_lanza) {
    State xi = e_i(4, 0);
    auto m  = e_i(8, 1);
    bool t = false;
    try { (void)step(xi, {m}, 4.0); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}
