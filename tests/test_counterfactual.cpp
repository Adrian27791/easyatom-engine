// =============================================================================
// tests/test_counterfactual.cpp  --  L30
// =============================================================================

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "test_framework.hpp"
#include "easyatom/decide/counterfactual.hpp"
#include "easyatom/hilbert/state.hpp"

using easyatom::decide::Counterfactual;
using easyatom::decide::find_counterfactual;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;

// Decision dummy: argmax sobre |amplitudes|^2 entre los primeros K canales.
static std::function<std::size_t(const State&)>
make_argmax(std::size_t K) {
    return [K](const State& s) -> std::size_t {
        std::size_t best = 0;
        double      bv   = -1.0;
        const auto& a    = s.amplitudes();
        for (std::size_t i = 0; i < K; ++i) {
            const double v = std::norm(a[i]);
            if (v > bv) { bv = v; best = i; }
        }
        return best;
    };
}

EATEST_CASE(cf_encuentra_delta_que_cambia_ganador) {
    // s tiene canal 0 = 1.0, canal 1 = 0.6 -> ganador 0.
    State s(8);
    s[0] = Complex{1.0, 0.0};
    s[1] = Complex{0.6, 0.0};
    // Direccion: empuja canal 1.
    State d(8);
    d[1] = Complex{1.0, 0.0};
    auto winner = make_argmax(8);
    auto cf = find_counterfactual(s, {d}, winner, 2.0);
    EATEST_REQUIRE(cf.found);
    EATEST_REQUIRE(cf.new_winner == 1);
    EATEST_REQUIRE(cf.direction_index == 0);
    // delta minimo: alpha tal que (0.6+alpha)^2 = 1 -> alpha=0.4.
    // norm2 = alpha^2 ~ 0.16.
    EATEST_REQUIRE(std::abs(cf.delta_norm2 - 0.16) < 5e-3);
}

EATEST_CASE(cf_no_existe_si_ninguna_direccion_cambia) {
    State s(4);
    s[0] = Complex{1.0, 0.0};
    State d(4);
    d[2] = Complex{0.001, 0.0};   // muy pequena, no llega
    auto winner = make_argmax(4);
    auto cf = find_counterfactual(s, {d}, winner, 1.0);
    EATEST_REQUIRE(!cf.found);
    EATEST_REQUIRE(cf.direction_index == Counterfactual::npos);
}

EATEST_CASE(cf_elige_direccion_de_menor_norma) {
    State s(8);
    s[0] = Complex{1.0, 0.0};
    s[1] = Complex{0.5, 0.0};
    // d_pequena empuja canal 1: necesita alpha~0.5, norm2 = 0.25.
    State d_pequena(8); d_pequena[1] = Complex{1.0, 0.0};
    // d_grande empuja canal 2 desde 0: necesita alpha>1, norm2 > 1.0.
    State d_grande(8);  d_grande[2]  = Complex{1.0, 0.0};
    auto winner = make_argmax(8);
    auto cf = find_counterfactual(s, {d_grande, d_pequena}, winner, 4.0);
    EATEST_REQUIRE(cf.found);
    EATEST_REQUIRE(cf.direction_index == 1);   // pequena gana
    EATEST_REQUIRE(cf.delta_norm2 < 0.5);
}

EATEST_CASE(cf_directions_vacios_lanza) {
    State s(4); s[0] = Complex{1.0, 0.0};
    auto winner = make_argmax(4);
    bool t = false;
    try { (void)find_counterfactual(s, {}, winner); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(cf_alpha_max_no_positivo_lanza) {
    State s(4); s[0] = Complex{1.0, 0.0};
    State d(4); d[1] = Complex{1.0, 0.0};
    auto winner = make_argmax(4);
    bool t = false;
    try { (void)find_counterfactual(s, {d}, winner, 0.0); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(cf_dim_incompatible_lanza) {
    State s(4); s[0] = Complex{1.0, 0.0};
    State d(8); d[1] = Complex{1.0, 0.0};
    auto winner = make_argmax(4);
    bool t = false;
    try { (void)find_counterfactual(s, {d}, winner); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(cf_delta_aplicado_realmente_cambia_ganador) {
    State s(8);
    s[0] = Complex{1.0, 0.0};
    s[1] = Complex{0.7, 0.0};
    State d(8); d[1] = Complex{1.0, 0.0};
    auto winner = make_argmax(8);
    auto cf = find_counterfactual(s, {d}, winner, 1.0);
    EATEST_REQUIRE(cf.found);
    State perturbed(8);
    const auto& a_s = s.amplitudes();
    const auto& a_dlt = cf.delta.amplitudes();
    for (std::size_t i = 0; i < 8; ++i)
        perturbed[i] = a_s[i] + a_dlt[i];
    EATEST_REQUIRE(winner(perturbed) != winner(s));
}

EATEST_CASE(cf_dos_iteraciones_bisect_estima_alpha_correcto) {
    State s(4); s[0] = Complex{1.0, 0.0}; s[1] = Complex{0.0, 0.0};
    State d(4); d[1] = Complex{1.0, 0.0};
    auto winner = make_argmax(4);
    // alpha minimo teorico: (0+alpha)^2 = 1 -> alpha = 1.
    auto cf = find_counterfactual(s, {d}, winner, 4.0, 64);
    EATEST_REQUIRE(cf.found);
    EATEST_REQUIRE(std::abs(cf.delta_norm2 - 1.0) < 1e-6);
}
