// =============================================================================
// tests/test_energy.cpp  --  L37
// =============================================================================

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "easyatom/auto/energy.hpp"
#include "easyatom/cst/compile.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/hilbert/state.hpp"
#include "test_framework.hpp"

using easyatom::autoloop::EnergyConfig;
using easyatom::autoloop::EnergyReport;
using easyatom::autoloop::compute_energy;
using easyatom::autoloop::rank_by_energy;
using easyatom::cst::CompiledLaw;
using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;

static State e_i(std::size_t d, std::size_t i, double a = 1.0) {
    State s(d);
    s[i] = Complex{a, 0.0};
    return s;
}

static CompiledLaw mk(std::string s, Relation r, std::string o, State st) {
    CompiledLaw l;
    l.triplet = Triplet{std::move(s), r, std::move(o)};
    l.state   = std::move(st);
    return l;
}

EATEST_CASE(en_codebook_vacio_e_repr_uno) {
    auto cand = mk("a", Relation::Causes, "b", e_i(8, 0));
    auto rep = compute_energy(cand, {});
    EATEST_REQUIRE(std::abs(rep.e_repr - 1.0) < 1e-9);
}

EATEST_CASE(en_estado_ya_soportado_e_repr_bajo) {
    std::vector<CompiledLaw> cb;
    for (std::size_t i = 0; i < 4; ++i)
        cb.push_back(mk("s" + std::to_string(i), Relation::Causes,
                        "o" + std::to_string(i), e_i(8, 0)));
    auto cand = mk("x", Relation::Causes, "y", e_i(8, 0));
    auto rep = compute_energy(cand, cb);
    EATEST_REQUIRE(rep.e_repr < 0.01);
}

EATEST_CASE(en_estado_ortogonal_e_repr_alto) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    auto cand = mk("c", Relation::Causes, "d", e_i(8, 4));
    auto rep = compute_energy(cand, cb);
    EATEST_REQUIRE(rep.e_repr > 0.9);
}

EATEST_CASE(en_contradiccion_da_energia_infinita) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("aspirina", Relation::Causes, "alivio", e_i(8, 0)));
    auto cand = mk("aspirina", Relation::Inhibits, "alivio", e_i(8, 1));
    auto rep = compute_energy(cand, cb);
    EATEST_REQUIRE(std::isinf(rep.e_total));
    EATEST_REQUIRE(std::isinf(rep.e_const));
}

EATEST_CASE(en_lambda_negativo_lanza) {
    auto cand = mk("a", Relation::Causes, "b", e_i(8, 0));
    EnergyConfig cfg; cfg.lambda = -1.0;
    bool t = false;
    try { (void)compute_energy(cand, {}, cfg); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(en_total_es_repr_mas_lambda_const_si_no_contra) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    auto cand = mk("c", Relation::Causes, "d", e_i(8, 4));
    EnergyConfig cfg; cfg.lambda = 2.0;
    auto rep = compute_energy(cand, cb, cfg);
    const double expected = rep.e_repr + cfg.lambda * rep.e_const;
    EATEST_REQUIRE(std::abs(rep.e_total - expected) < 1e-9);
}

EATEST_CASE(en_rank_pone_contradicciones_al_final) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("aspirina", Relation::Causes, "alivio", e_i(8, 0)));
    std::vector<CompiledLaw> cands;
    cands.push_back(mk("aspirina", Relation::Inhibits, "alivio", e_i(8, 1)));
    cands.push_back(mk("c", Relation::Causes, "d", e_i(8, 4)));   // OK
    cands.push_back(mk("e", Relation::Causes, "f", e_i(8, 0)));   // soportado
    auto idx = rank_by_energy(cands, cb);
    EATEST_REQUIRE(idx.size() == 3);
    EATEST_REQUIRE(idx.back() == 0);   // contradiccion al final
    EATEST_REQUIRE(idx.front() == 2);  // mejor: e_repr bajo
}

EATEST_CASE(en_rank_es_estable_con_un_solo_candidato) {
    auto cand = mk("a", Relation::Causes, "b", e_i(8, 0));
    auto idx = rank_by_energy({cand}, {});
    EATEST_REQUIRE(idx.size() == 1);
    EATEST_REQUIRE(idx[0] == 0);
}
