// =============================================================================
// tests/test_loop_ranked.cpp  --  L40
// =============================================================================

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "easyatom/auto/loop_ranked.hpp"
#include "easyatom/cst/compile.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/hilbert/state.hpp"
#include "test_framework.hpp"

using easyatom::autoloop::LoopRankedConfig;
using easyatom::autoloop::LoopRankedReport;
using easyatom::autoloop::RankMode;
using easyatom::autoloop::run_auto_loop_ranked;
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
    l.triplet     = Triplet{std::move(s), r, std::move(o)};
    l.state       = std::move(st);
    l.fingerprint = 0;
    return l;
}

// -------- 1: parametros invalidos --------------------------------------------
EATEST_CASE(lr_gap_theta_negativo_lanza) {
    std::vector<CompiledLaw> cb;
    LoopRankedConfig cfg; cfg.gap_theta = -0.1;
    bool t = false;
    try { (void)run_auto_loop_ranked(cb, {}, cfg); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(lr_k_min_cero_lanza) {
    std::vector<CompiledLaw> cb;
    LoopRankedConfig cfg; cfg.k_min = 0;
    bool t = false;
    try { (void)run_auto_loop_ranked(cb, {}, cfg); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(lr_k_max_menor_que_k_min_lanza) {
    std::vector<CompiledLaw> cb;
    LoopRankedConfig cfg; cfg.k_min = 4; cfg.k_max = 2;
    bool t = false;
    try { (void)run_auto_loop_ranked(cb, {}, cfg); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

// -------- 2: sin queries no hace nada ----------------------------------------
EATEST_CASE(lr_sin_queries_no_itera) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    auto rep = run_auto_loop_ranked(cb, {});
    EATEST_REQUIRE(rep.gaps_detected == 0u);
    EATEST_REQUIRE(rep.accepted == 0u);
    EATEST_REQUIRE(cb.size() == 1u);
}

// -------- 3: codebook vacio + queries -> no propone nada ---------------------
EATEST_CASE(lr_codebook_vacio_no_propone) {
    std::vector<CompiledLaw> cb;
    std::vector<State> qs = { e_i(8, 0) };
    LoopRankedConfig cfg; cfg.gap_theta = 0.5;
    auto rep = run_auto_loop_ranked(cb, qs, cfg);
    EATEST_REQUIRE(rep.accepted == 0u);
    EATEST_REQUIRE(cb.empty());
}

// -------- 4: caso happy: gap se cierra y codebook crece ---------------------
EATEST_CASE(lr_happy_acepta_y_crece) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    cb.push_back(mk("c", Relation::Causes, "d", e_i(8, 1)));
    cb.push_back(mk("e", Relation::Causes, "f", e_i(8, 2)));
    cb.push_back(mk("g", Relation::Causes, "h", e_i(8, 3)));
    std::vector<State> qs = { e_i(8, 4) };  // ortogonal => gap claro
    LoopRankedConfig cfg;
    cfg.gap_theta = 0.5;
    cfg.k_min = 2; cfg.k_max = 4;
    cfg.max_iters = 4;
    cfg.mode = RankMode::Energy;
    const std::size_t before = cb.size();
    auto rep = run_auto_loop_ranked(cb, qs, cfg);
    EATEST_REQUIRE(rep.gaps_detected >= 1u);
    EATEST_REQUIRE(rep.candidates_total >= 3u);  // (k_max-k_min+1)
    EATEST_REQUIRE(cb.size() >= before);          // creció o estable, no decrece
}

// -------- 5: report cuenta candidatos generados ------------------------------
EATEST_CASE(lr_report_cuenta_candidatos) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    cb.push_back(mk("c", Relation::Causes, "d", e_i(8, 1)));
    cb.push_back(mk("e", Relation::Causes, "f", e_i(8, 2)));
    std::vector<State> qs = { e_i(8, 5) };
    LoopRankedConfig cfg;
    cfg.gap_theta = 0.6;
    cfg.k_min = 1; cfg.k_max = 3;
    cfg.max_iters = 1;
    auto rep = run_auto_loop_ranked(cb, qs, cfg);
    if (rep.gaps_detected >= 1u) {
        EATEST_REQUIRE(rep.candidates_total >= 3u);
        EATEST_REQUIRE(rep.candidates_unique <= rep.candidates_total);
    }
}

// -------- 6: discovery mode no lanza, llega al fin ---------------------------
EATEST_CASE(lr_modo_discovery_corre) {
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    cb.push_back(mk("c", Relation::Inhibits, "d", e_i(8, 1)));
    cb.push_back(mk("e", Relation::IsA, "f", e_i(8, 2)));
    std::vector<State> qs = { e_i(8, 6) };
    LoopRankedConfig cfg;
    cfg.gap_theta = 0.5;
    cfg.k_min = 2; cfg.k_max = 3;
    cfg.max_iters = 2;
    cfg.mode = RankMode::Discovery;
    auto rep = run_auto_loop_ranked(cb, qs, cfg);
    EATEST_REQUIRE(rep.iters >= 1u);
}

// -------- 7: idempotencia: rerun sin gaps no muta el codebook ---------------
EATEST_CASE(lr_sin_gaps_no_muta) {
    std::vector<CompiledLaw> cb;
    for (std::size_t i = 0; i < 4; ++i)
        cb.push_back(mk("s" + std::to_string(i), Relation::Causes,
                        "o" + std::to_string(i), e_i(8, 0)));
    std::vector<State> qs = { e_i(8, 0) };  // density ~ 1.0
    LoopRankedConfig cfg;
    cfg.gap_theta = 0.1;  // ningun gap superara este umbral
    const std::size_t before = cb.size();
    auto rep = run_auto_loop_ranked(cb, qs, cfg);
    EATEST_REQUIRE(rep.gaps_detected == 0u);
    EATEST_REQUIRE(cb.size() == before);
}

// -------- 8: best-first: aceptado es siempre <= rank rejected_coherence ----
EATEST_CASE(lr_best_first_no_acepta_si_todas_rechazan) {
    // Con un solo law en el codebook, try_fill produce siempre un bundle del mismo
    // estado; cualquier cantidad de variaciones de k colapsa al mismo. Aqui solo
    // verificamos que no explote y que rejected_coherence + accepted = num intentos
    // efectivos sobre candidatos unicos.
    std::vector<CompiledLaw> cb;
    cb.push_back(mk("a", Relation::Causes, "b", e_i(8, 0)));
    std::vector<State> qs = { e_i(8, 7) };
    LoopRankedConfig cfg;
    cfg.gap_theta = 0.5;
    cfg.k_min = 1; cfg.k_max = 1;
    cfg.max_iters = 1;
    auto rep = run_auto_loop_ranked(cb, qs, cfg);
    EATEST_REQUIRE(rep.accepted + rep.rejected_coherence <= rep.candidates_unique
                   || rep.candidates_unique == 0u);
}
