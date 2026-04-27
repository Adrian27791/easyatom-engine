// =============================================================================
// tests/test_multiprobe.cpp  --  L38
// =============================================================================

#include <stdexcept>
#include <vector>

#include "easyatom/dynamics/multiprobe.hpp"
#include "easyatom/hilbert/state.hpp"
#include "test_framework.hpp"

using easyatom::dynamics::multiprobe::MultiProbeConfig;
using easyatom::dynamics::multiprobe::MultiProbeReport;
using easyatom::dynamics::multiprobe::multi_probe_collapse;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;

static State e_i(std::size_t d, std::size_t i, double a = 1.0) {
    State s(d);
    s[i] = Complex{a, 0.0};
    return s;
}

EATEST_CASE(mp_query_clara_es_estable) {
    auto m0 = e_i(8, 0);
    auto m1 = e_i(8, 4);
    State q(8);
    q[0] = Complex{0.95, 0.0};
    q[4] = Complex{0.05, 0.0};
    MultiProbeConfig cfg;
    cfg.hopfield.beta = 32.0;
    auto r = multi_probe_collapse(q, {m0, m1}, cfg);
    EATEST_REQUIRE(r.stable);
    EATEST_REQUIRE(fidelity(r.consensus, m0) > 0.99);
}

EATEST_CASE(mp_query_ambigua_no_es_estable) {
    auto m0 = e_i(8, 0);
    auto m1 = e_i(8, 4);
    State q(8);
    q[0] = Complex{0.5, 0.0};
    q[4] = Complex{0.5, 0.0};   // exactamente en la frontera
    MultiProbeConfig cfg;
    cfg.n_probes      = 16;
    cfg.perturbation  = 0.3;     // ruido alto -> se separan
    cfg.agreement_tol = 0.95;
    cfg.hopfield.beta = 64.0;    // beta alto -> colapso a uno u otro
    auto r = multi_probe_collapse(q, {m0, m1}, cfg);
    EATEST_REQUIRE(!r.stable);
    EATEST_REQUIRE(r.agreement < 0.95);
}

EATEST_CASE(mp_perturbation_cero_da_acuerdo_perfecto) {
    auto m0 = e_i(8, 0);
    State q = e_i(8, 0, 0.9);
    q[1] = Complex{0.1, 0.0};
    MultiProbeConfig cfg;
    cfg.perturbation = 0.0;
    auto r = multi_probe_collapse(q, {m0}, cfg);
    EATEST_REQUIRE(r.agreement > 0.999);
    EATEST_REQUIRE(r.stable);
}

EATEST_CASE(mp_consensus_normalizado) {
    auto m0 = e_i(8, 0);
    State q = e_i(8, 0, 0.9);
    auto r = multi_probe_collapse(q, {m0});
    double n2 = 0.0;
    for (std::size_t i = 0; i < r.consensus.dim(); ++i)
        n2 += std::norm(r.consensus[i]);
    EATEST_REQUIRE(std::abs(n2 - 1.0) < 1e-9);
}

EATEST_CASE(mp_seed_reproducible) {
    auto m0 = e_i(8, 0);
    auto m1 = e_i(8, 4);
    State q(8);
    q[0] = Complex{0.5, 0.0};
    q[4] = Complex{0.5, 0.0};
    MultiProbeConfig cfg;
    cfg.seed = 12345;
    cfg.perturbation = 0.2;
    auto r1 = multi_probe_collapse(q, {m0, m1}, cfg);
    auto r2 = multi_probe_collapse(q, {m0, m1}, cfg);
    EATEST_REQUIRE(std::abs(r1.agreement - r2.agreement) < 1e-12);
}

EATEST_CASE(mp_memorias_vacias_lanza) {
    State q = e_i(4, 0);
    bool t = false;
    try { (void)multi_probe_collapse(q, {}); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(mp_n_probes_cero_lanza) {
    State q = e_i(4, 0);
    auto m  = e_i(4, 1);
    MultiProbeConfig cfg; cfg.n_probes = 0;
    bool t = false;
    try { (void)multi_probe_collapse(q, {m}, cfg); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(mp_perturbation_negativa_lanza) {
    State q = e_i(4, 0);
    auto m  = e_i(4, 1);
    MultiProbeConfig cfg; cfg.perturbation = -0.1;
    bool t = false;
    try { (void)multi_probe_collapse(q, {m}, cfg); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}
