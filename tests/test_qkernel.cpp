// Tests del Ladrillo 7 — fachada Q-Kernel.

#include "test_framework.hpp"
#include "easyatom/qkernel/qkernel.hpp"

#include <string>
#include <vector>

using easyatom::qkernel::QKernel;
using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;

EATEST_CASE(qkernel_dim_cero_lanza) {
    bool threw = false;
    try { QKernel q(0, 42); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(qkernel_ingest_es_idempotente) {
    QKernel q(256, 7);
    const State& a = q.ingest("alfa");
    const State& b = q.ingest("alfa");
    EATEST_REQUIRE(&a == &b);  // misma referencia
    EATEST_REQUIRE(q.codebook_size() == 1);
    (void)q.ingest("beta");
    EATEST_REQUIRE(q.codebook_size() == 2);
}

EATEST_CASE(qkernel_misma_seed_misma_codebook) {
    QKernel q1(128, 12345);
    QKernel q2(128, 12345);
    State s1 = q1.ingest("nombre_x");
    State s2 = q2.ingest("nombre_x");
    EATEST_REQUIRE_NEAR(fidelity(s1, s2), 1.0, 1e-12);
}

EATEST_CASE(qkernel_seed_distinta_estados_distintos) {
    QKernel q1(256, 1);
    QKernel q2(256, 2);
    State s1 = q1.ingest("hola");
    State s2 = q2.ingest("hola");
    // Casi ortogonales en alta dimensión.
    EATEST_REQUIRE(fidelity(s1, s2) < 0.05);
}

EATEST_CASE(qkernel_compose_y_query_recuperan_filler) {
    QKernel q(512, 99);
    const State& role   = q.ingest("color");
    const State& filler = q.ingest("rojo");
    State composed = q.compose(role, filler);
    State recovered = q.query(composed, role);
    // unbind exacto en bind de fase pura → fidelidad 1 con el filler original.
    EATEST_REQUIRE_NEAR(fidelity(recovered, filler), 1.0, 1e-9);
}

EATEST_CASE(qkernel_bundle_pairs_query_recupera_filler_con_argmax) {
    QKernel q(1024, 2024);
    const State& color   = q.ingest("color");
    const State& tamano  = q.ingest("tamano");
    const State& rojo    = q.ingest("rojo");
    const State& grande  = q.ingest("grande");
    const State& azul    = q.ingest("azul");   // distractor
    const State& peque   = q.ingest("pequeno"); // distractor
    (void)azul; (void)peque;

    State composite = q.bundle_pairs({{color, rojo}, {tamano, grande}});
    State guess = q.query(composite, color);
    std::string best = q.argmax_collapse(
        guess, {"rojo", "grande", "azul", "pequeno"});
    EATEST_REQUIRE(best == "rojo");

    State guess2 = q.query(composite, tamano);
    std::string best2 = q.argmax_collapse(
        guess2, {"rojo", "grande", "azul", "pequeno"});
    EATEST_REQUIRE(best2 == "grande");
}

EATEST_CASE(qkernel_collapse_es_distribucion_valida) {
    QKernel q(512, 11);
    (void)q.ingest("a"); (void)q.ingest("b"); (void)q.ingest("c");
    State s = q.ingest("b");  // estado puro = b
    auto d = q.collapse(s, {"a", "b", "c"});
    EATEST_REQUIRE(d.dim() == 3);
    // p[b] dominante.
    EATEST_REQUIRE(d[1] > d[0]);
    EATEST_REQUIRE(d[1] > d[2]);
    // Suma = 1 (con tolerancia interna).
    double s_p = d[0] + d[1] + d[2];
    EATEST_REQUIRE_NEAR(s_p, 1.0, 1e-9);
}

EATEST_CASE(qkernel_collapse_nombre_inexistente_lanza) {
    QKernel q(64, 3);
    (void)q.ingest("x");
    State s = q.ingest("x");
    bool threw = false;
    try { (void)q.collapse(s, {"x", "no_existe"}); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(qkernel_bundle_pairs_vacio_lanza) {
    QKernel q(32, 1);
    bool threw = false;
    try { (void)q.bundle_pairs({}); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}
