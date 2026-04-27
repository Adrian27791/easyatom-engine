// Tests del Ladrillo 9 — TracedKernel / introspección.

#include "test_framework.hpp"
#include "easyatom/introspect/trace.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using easyatom::introspect::TracedKernel;
using easyatom::introspect::Op;
using easyatom::hilbert::State;

EATEST_CASE(trace_eventos_se_acumulan_en_orden) {
    TracedKernel tk(64, 1);
    (void)tk.ingest("a");
    (void)tk.ingest("b");
    auto c = tk.compose("a", "b");
    (void)c;
    const auto& ev = tk.events();
    EATEST_REQUIRE(ev.size() == 3);
    EATEST_REQUIRE(ev[0].op == Op::Ingest);
    EATEST_REQUIRE(ev[1].op == Op::Ingest);
    EATEST_REQUIRE(ev[2].op == Op::Compose);
    EATEST_REQUIRE(ev[2].inputs.size() == 2);
    EATEST_REQUIRE(ev[2].inputs[0] == "a");
    EATEST_REQUIRE(ev[2].inputs[1] == "b");
}

EATEST_CASE(trace_state_norm_es_uno_para_estados_normalizados) {
    TracedKernel tk(128, 99);
    (void)tk.ingest("x");
    const auto& ev = tk.events();
    EATEST_REQUIRE_NEAR(ev[0].state_norm, 1.0, 1e-9);
}

EATEST_CASE(trace_fidelity_prev_es_uno_si_repites_el_mismo_ingest) {
    TracedKernel tk(256, 5);
    (void)tk.ingest("alpha");
    (void)tk.ingest("alpha");  // mismo estado
    const auto& ev = tk.events();
    EATEST_REQUIRE(ev.size() == 2);
    EATEST_REQUIRE_NEAR(ev[1].fidelity_prev, 1.0, 1e-9);
}

EATEST_CASE(trace_fidelity_prev_es_baja_entre_conceptos_distintos) {
    TracedKernel tk(512, 7);
    (void)tk.ingest("perro");
    (void)tk.ingest("gato");  // distinto seed → casi ortogonal
    const auto& ev = tk.events();
    EATEST_REQUIRE(ev[1].fidelity_prev < 0.05);
}

EATEST_CASE(trace_collapse_genera_distribucion_y_fisher_step) {
    TracedKernel tk(1024, 11);
    (void)tk.ingest("rojo");
    (void)tk.ingest("azul");
    (void)tk.ingest("verde");
    State s_rojo = tk.kernel().ingest("rojo");
    State s_azul = tk.kernel().ingest("azul");
    auto d1 = tk.collapse(s_rojo, {"rojo","azul","verde"});
    auto d2 = tk.collapse(s_azul, {"rojo","azul","verde"});
    (void)d1; (void)d2;
    const auto& ev = tk.events();
    // últimos dos eventos son collapses
    EATEST_REQUIRE(ev[ev.size()-2].op == Op::Collapse);
    EATEST_REQUIRE(ev[ev.size()-1].op == Op::Collapse);
    EATEST_REQUIRE(ev[ev.size()-2].has_dist);
    EATEST_REQUIRE(ev[ev.size()-1].has_dist);
    // El primer collapse no tiene fisher_step previo → NaN.
    EATEST_REQUIRE(std::isnan(ev[ev.size()-2].fisher_step));
    // El segundo sí: dos distribuciones distintas → fisher_step > 0.
    EATEST_REQUIRE(!std::isnan(ev[ev.size()-1].fisher_step));
    EATEST_REQUIRE(ev[ev.size()-1].fisher_step > 0.0);
}

EATEST_CASE(trace_pipeline_role_filler_completo_es_auditable) {
    TracedKernel tk(1024, 2024,
        /*probe_codebook*/ {});
    // Pre-ingestar todos los candidatos del codebook de respuesta.
    (void)tk.ingest("rojo");
    (void)tk.ingest("grande");
    (void)tk.ingest("azul");
    (void)tk.ingest("pequeno");
    (void)tk.ingest("color");
    (void)tk.ingest("tamano");
    auto composite = tk.bundle_pairs(
        {{"color","rojo"}, {"tamano","grande"}});
    auto guess = tk.query(composite, "color");
    auto winner = tk.argmax_collapse(guess,
        {"rojo","grande","azul","pequeno"});
    EATEST_REQUIRE(winner == "rojo");
    auto sum = tk.summarize();
    EATEST_REQUIRE(sum.n_events >= 9);  // 6 ingest + bundle + query + argmax
    EATEST_REQUIRE(sum.n_states >= 2);
}

EATEST_CASE(trace_probe_codebook_proyecta_cualquier_estado) {
    TracedKernel tk(512, 3);
    (void)tk.ingest("a"); (void)tk.ingest("b"); (void)tk.ingest("c");
    tk.set_probe_codebook({"a","b","c"});
    State s = tk.kernel().ingest("b");
    auto d = tk.probe(s);
    EATEST_REQUIRE(d.dim() == 3);
    // p[b] dominante.
    EATEST_REQUIRE(d[1] > d[0]);
    EATEST_REQUIRE(d[1] > d[2]);
}

EATEST_CASE(trace_probe_sin_codebook_lanza) {
    TracedKernel tk(64, 1);
    (void)tk.ingest("x");
    State s = tk.kernel().ingest("x");
    bool threw = false;
    try { (void)tk.probe(s); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(trace_to_json_contiene_op_y_distribucion) {
    TracedKernel tk(128, 1);
    (void)tk.ingest("a");
    State s = tk.kernel().ingest("a");
    (void)tk.collapse(s, {"a"});
    std::string j = tk.to_json();
    // chequeos triviales: contiene los nombres.
    EATEST_REQUIRE(j.find("\"op\":\"ingest\"")    != std::string::npos);
    EATEST_REQUIRE(j.find("\"op\":\"collapse\"")  != std::string::npos);
    EATEST_REQUIRE(j.find("\"distribution\"")     != std::string::npos);
    EATEST_REQUIRE(j.find("\"name\":\"a\"")       != std::string::npos);
}

EATEST_CASE(trace_clear_resetea_eventos_y_dist_previa) {
    TracedKernel tk(64, 1);
    (void)tk.ingest("x");
    State s = tk.kernel().ingest("x");
    (void)tk.collapse(s, {"x"});
    EATEST_REQUIRE(tk.events().size() == 2);
    tk.clear_trace();
    EATEST_REQUIRE(tk.events().empty());
    // Tras clear, el siguiente collapse no debe tener fisher_step previo.
    (void)tk.ingest("x");
    State s2 = tk.kernel().ingest("x");
    (void)tk.collapse(s2, {"x"});
    const auto& ev = tk.events();
    EATEST_REQUIRE(std::isnan(ev.back().fisher_step));
}

EATEST_CASE(trace_summarize_acumula_fisher_path_y_min_fidelity) {
    TracedKernel tk(256, 42);
    (void)tk.ingest("a");
    (void)tk.ingest("b");  // ortogonal → fidelity_prev baja
    auto sum = tk.summarize();
    EATEST_REQUIRE(sum.n_states == 2);
    EATEST_REQUIRE(sum.min_fidelity_prev < 0.1);
}

EATEST_CASE(trace_bundle_pairs_vacio_lanza) {
    TracedKernel tk(32, 1);
    bool threw = false;
    try { (void)tk.bundle_pairs({}); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}
