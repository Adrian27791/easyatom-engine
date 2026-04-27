// =============================================================================
// tests/test_autoloop.cpp  --  L32 circulo autonomo
// =============================================================================

#include <cmath>
#include <stdexcept>

#include "easyatom/auto/loop.hpp"
#include "test_framework.hpp"

using easyatom::autoloop::LoopConfig;
using easyatom::autoloop::LoopReport;
using easyatom::autoloop::run_auto_loop;
using easyatom::cst::CompiledLaw;
using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;

static CompiledLaw mk_law(std::string s, Relation r, std::string o,
                          std::size_t d, std::size_t i) {
    CompiledLaw L;
    L.triplet = Triplet{std::move(s), r, std::move(o)};
    State st(d);
    st[i] = Complex{1.0, 0.0};
    L.state = std::move(st);
    L.fingerprint = 0xAAAA + i;
    return L;
}

EATEST_CASE(autoloop_no_gaps_no_crece) {
    std::vector<CompiledLaw> codebook = {
        mk_law("a", Relation::IsA, "b", 8, 0),
        mk_law("c", Relation::IsA, "d", 8, 0),
    };
    std::vector<State> queries = { codebook[0].state };  // ya cubierta
    LoopConfig cfg; cfg.gap_theta = 0.5;
    auto rep = run_auto_loop(codebook, queries, cfg);
    EATEST_REQUIRE(rep.gaps_detected == 0);
    EATEST_REQUIRE(rep.accepted == 0);
    EATEST_REQUIRE(codebook.size() == 2);
}

EATEST_CASE(autoloop_detecta_y_propone) {
    std::vector<CompiledLaw> codebook = {
        mk_law("a", Relation::IsA, "b", 8, 0),
        mk_law("c", Relation::IsA, "d", 8, 1),
    };
    State q(8); q[2] = Complex{1.0, 0.0};      // ortogonal a todos
    LoopConfig cfg; cfg.gap_theta = 0.5; cfg.max_iters = 1; cfg.k_top = 2;
    auto rep = run_auto_loop(codebook, {q}, cfg);
    EATEST_REQUIRE(rep.gaps_detected >= 1);
    EATEST_REQUIRE(rep.proposals >= 1);
}

EATEST_CASE(autoloop_acepta_y_crece_codebook) {
    std::vector<CompiledLaw> codebook = {
        mk_law("a", Relation::IsA, "b", 8, 0),
        mk_law("c", Relation::IsA, "d", 8, 1),
    };
    State q(8); q[2] = Complex{1.0, 0.0};
    const std::size_t before = codebook.size();
    LoopConfig cfg; cfg.gap_theta = 0.5; cfg.max_iters = 2;
    auto rep = run_auto_loop(codebook, {q}, cfg);
    // El cuerpo debe haber crecido en exactamente rep.accepted leyes.
    EATEST_REQUIRE(codebook.size() == before + rep.accepted);
}

EATEST_CASE(autoloop_max_iters_se_respeta) {
    std::vector<CompiledLaw> codebook = {
        mk_law("a", Relation::IsA, "b", 8, 0),
    };
    State q(8); q[1] = Complex{1.0, 0.0};
    LoopConfig cfg; cfg.gap_theta = 0.5; cfg.max_iters = 3;
    auto rep = run_auto_loop(codebook, {q}, cfg);
    EATEST_REQUIRE(rep.iters <= 3);
}

EATEST_CASE(autoloop_corta_si_no_acepta_nada) {
    // Theta absurdo: todo es gap pero ninguna propuesta sobrevivira si
    // beta_1 sube. Aun asi el loop debe terminar (no infinito).
    std::vector<CompiledLaw> codebook = {
        mk_law("a", Relation::IsA, "b", 8, 0),
        mk_law("c", Relation::IsA, "d", 8, 1),
    };
    State q(8); q[3] = Complex{1.0, 0.0};
    LoopConfig cfg; cfg.gap_theta = 0.99; cfg.max_iters = 50;
    auto rep = run_auto_loop(codebook, {q}, cfg);
    EATEST_REQUIRE(rep.iters <= 50);  // termino
}

EATEST_CASE(autoloop_lanza_si_theta_negativo) {
    std::vector<CompiledLaw> codebook = {
        mk_law("a", Relation::IsA, "b", 8, 0),
    };
    LoopConfig cfg; cfg.gap_theta = -0.1;
    bool t = false;
    try { (void)run_auto_loop(codebook, {}, cfg); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(autoloop_codebook_vacio_no_propone_nada) {
    std::vector<CompiledLaw> codebook;
    State q(8); q[0] = Complex{1.0, 0.0};
    LoopConfig cfg; cfg.gap_theta = 0.5; cfg.max_iters = 3;
    auto rep = run_auto_loop(codebook, {q}, cfg);
    // try_fill devuelve nullopt sobre codebook vacio; gaps detectados pero
    // sin propuestas posibles, el loop corta.
    EATEST_REQUIRE(rep.proposals == 0);
    EATEST_REQUIRE(codebook.empty());
}

EATEST_CASE(autoloop_report_es_consistente) {
    std::vector<CompiledLaw> codebook = {
        mk_law("a", Relation::IsA, "b", 8, 0),
        mk_law("c", Relation::IsA, "d", 8, 1),
    };
    State q(8); q[2] = Complex{1.0, 0.0};
    LoopConfig cfg; cfg.gap_theta = 0.5; cfg.max_iters = 3;
    auto rep = run_auto_loop(codebook, {q}, cfg);
    EATEST_REQUIRE(rep.accepted + rep.rejected_coherence == rep.proposals);
}
