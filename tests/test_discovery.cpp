// =============================================================================
// tests/test_discovery.cpp  --  L36
// =============================================================================

#include <stdexcept>
#include <vector>

#include "easyatom/cst/compile.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/score/discovery.hpp"
#include "test_framework.hpp"

using easyatom::cst::CompiledLaw;
using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;
using easyatom::score::ScoreConfig;
using easyatom::score::ScoreReport;
using easyatom::score::discovery_score;
using easyatom::score::rank_candidates;

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

EATEST_CASE(disc_codebook_vacio_da_novelty_uno) {
    auto cand = mk("a", Relation::Causes, "b", e_i(8, 0));
    auto rep = discovery_score(cand, {});
    EATEST_REQUIRE(rep.novelty > 0.999);
}

EATEST_CASE(disc_estado_existente_tiene_novelty_cero) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    auto cand = mk("c", Relation::Causes, "d", e_i(8, 0));  // misma direccion
    auto rep = discovery_score(cand, cb);
    EATEST_REQUIRE(rep.novelty < 0.01);
}

EATEST_CASE(disc_estado_ortogonal_tiene_novelty_uno) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    auto cand = mk("c", Relation::Causes, "d", e_i(8, 3));
    auto rep = discovery_score(cand, cb);
    EATEST_REQUIRE(rep.novelty > 0.99);
}

EATEST_CASE(disc_cross_domain_cero_si_unica_relacion) {
    std::vector<CompiledLaw> cb;
    for (std::size_t i = 0; i < 5; ++i)
        cb.push_back(mk("s" + std::to_string(i), Relation::Causes,
                        "o" + std::to_string(i), e_i(8, i)));
    auto cand = mk("x", Relation::Causes, "y", e_i(8, 1));
    auto rep = discovery_score(cand, cb);
    EATEST_REQUIRE(rep.cross_domain < 0.001);
}

EATEST_CASE(disc_cross_domain_alto_si_relaciones_diversas) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("s0", Relation::Causes,    "o0", e_i(8, 0)));
    cb.push_back(mk("s1", Relation::Inhibits,  "o1", e_i(8, 1)));
    cb.push_back(mk("s2", Relation::Increases, "o2", e_i(8, 2)));
    cb.push_back(mk("s3", Relation::Treats,    "o3", e_i(8, 3)));
    cb.push_back(mk("s4", Relation::IsA,       "o4", e_i(8, 4)));
    auto cand = mk("x", Relation::HasProperty, "y", e_i(8, 1));
    ScoreConfig cfg; cfg.k_neighbors = 5;
    auto rep = discovery_score(cand, cb, cfg);
    EATEST_REQUIRE(rep.cross_domain > 0.9);
}

EATEST_CASE(disc_score_total_combina_pesos) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    auto cand = mk("c", Relation::Causes, "d", e_i(8, 4));
    ScoreConfig cfg;
    cfg.weights.alpha = 1.0; cfg.weights.beta = 0.0; cfg.weights.gamma = 0.0;
    auto rep = discovery_score(cand, cb, cfg);
    EATEST_REQUIRE(std::abs(rep.total - rep.novelty) < 1e-9);
}

EATEST_CASE(disc_pesos_negativos_lanzan) {
    auto cand = mk("a", Relation::Causes, "b", e_i(8, 0));
    ScoreConfig cfg; cfg.weights.alpha = -1.0;
    bool t = false;
    try { (void)discovery_score(cand, {}, cfg); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(disc_rank_ordena_descendente) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    std::vector<CompiledLaw> cands;
    cands.push_back(mk("p", Relation::Causes, "q", e_i(8, 0)));   // novelty 0
    cands.push_back(mk("r", Relation::Causes, "s", e_i(8, 4)));   // novelty 1
    cands.push_back(mk("t", Relation::Causes, "u", e_i(8, 2)));   // novelty 1
    ScoreConfig cfg;
    cfg.weights.alpha = 1.0; cfg.weights.beta = 0.0; cfg.weights.gamma = 0.0;
    auto idx = rank_candidates(cands, cb, cfg);
    EATEST_REQUIRE(idx.size() == 3);
    EATEST_REQUIRE(idx.back() == 0);  // el de novelty 0 va al final
}
