#include "test_framework.hpp"
#include "easyatom/explain/decoder.hpp"
#include "easyatom/decide/decisor.hpp"
#include "easyatom/introspect/trace.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using easyatom::decide::Decision;
using easyatom::decide::DecisionKind;
using easyatom::decide::decide;
using easyatom::explain::decode_decision;
using easyatom::explain::decode_topk;
using easyatom::explain::decode_event;
using easyatom::explain::decode_trace;
using easyatom::explain::decode_full;
using easyatom::infogeo::Distribution;
using easyatom::introspect::TraceEvent;
using easyatom::introspect::TracedKernel;
using easyatom::introspect::Op;

EATEST_CASE(decoder_accept_menciona_ganador_y_confianza) {
    Distribution p({0.85, 0.10, 0.05});
    auto d = decide(p, {"rojo","azul","verde"});
    auto s = decode_decision(d);
    EATEST_REQUIRE(s.find("'rojo'") != std::string::npos);
    EATEST_REQUIRE(s.find("85,0%") != std::string::npos);
    EATEST_REQUIRE(s.find("muy alta") != std::string::npos);
}

EATEST_CASE(decoder_ambiguous_menciona_a_los_dos_top) {
    Distribution p({0.46, 0.44, 0.10});
    auto d = decide(p, {"a","b","c"});
    auto s = decode_decision(d);
    EATEST_REQUIRE(s.find("'a'") != std::string::npos);
    EATEST_REQUIRE(s.find("'b'") != std::string::npos);
    EATEST_REQUIRE(s.find("duda") != std::string::npos);
}

EATEST_CASE(decoder_abstain_dice_se_abstiene) {
    Distribution p({0.20, 0.18, 0.16, 0.16, 0.15, 0.15});
    auto d = decide(p, {"a","b","c","d","e","f"});
    auto s = decode_decision(d);
    EATEST_REQUIRE(s.find("abstiene") != std::string::npos ||
                   s.find("dispersa") != std::string::npos);
}

EATEST_CASE(decoder_topk_lista_los_k_mejores_en_orden) {
    Distribution p({0.50, 0.30, 0.15, 0.05});
    std::vector<std::string> n = {"a","b","c","d"};
    auto s = decode_topk(p, n, 3);
    auto pa = s.find("'a'"), pb = s.find("'b'"), pc = s.find("'c'");
    EATEST_REQUIRE(pa != std::string::npos);
    EATEST_REQUIRE(pb != std::string::npos);
    EATEST_REQUIRE(pc != std::string::npos);
    EATEST_REQUIRE(pa < pb);
    EATEST_REQUIRE(pb < pc);
    EATEST_REQUIRE(s.find("'d'") == std::string::npos);
}

EATEST_CASE(decoder_topk_dim_mismatch_lanza) {
    Distribution p({0.5, 0.5});
    bool threw = false;
    try { (void)decode_topk(p, {"a"}, 1); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(decoder_evento_ingest_es_legible) {
    TraceEvent e;
    e.op = Op::Ingest;
    e.inputs = {"perro"};
    e.output_label = "perro";
    auto s = decode_event(e);
    EATEST_REQUIRE(s.find("'perro'") != std::string::npos);
    EATEST_REQUIRE(s.find("Registr") != std::string::npos);
}

EATEST_CASE(decoder_evento_compose_menciona_rol_y_valor) {
    TraceEvent e;
    e.op = Op::Compose;
    e.inputs = {"color","rojo"};
    e.output_label = "color:rojo";
    auto s = decode_event(e);
    EATEST_REQUIRE(s.find("'color'") != std::string::npos);
    EATEST_REQUIRE(s.find("'rojo'") != std::string::npos);
}

EATEST_CASE(decoder_trace_completo_pipeline_role_filler) {
    TracedKernel tk(512, 7);
    (void)tk.ingest("rojo");
    (void)tk.ingest("azul");
    (void)tk.ingest("color");
    auto comp = tk.bundle_pairs({{"color","rojo"}});
    auto g = tk.query(comp, "color");
    auto winner = tk.argmax_collapse(g, {"rojo","azul"});
    EATEST_REQUIRE(winner == "rojo");

    auto narrative = decode_trace(tk.events());
    EATEST_REQUIRE(narrative.find("Registr") != std::string::npos);
    EATEST_REQUIRE(narrative.find("Combin") != std::string::npos);
    EATEST_REQUIRE(narrative.find("Consult") != std::string::npos);
    EATEST_REQUIRE(narrative.find("Eligi") != std::string::npos);
    EATEST_REQUIRE(narrative.find("'rojo'") != std::string::npos);
}

EATEST_CASE(decoder_full_combina_decision_y_topk) {
    Distribution p({0.7, 0.2, 0.1});
    auto d = decide(p, {"perro","gato","loro"});
    auto s = decode_full(d, p, {"perro","gato","loro"}, 2);
    EATEST_REQUIRE(s.find("'perro'") != std::string::npos);
    EATEST_REQUIRE(s.find("Top-2") != std::string::npos);
    EATEST_REQUIRE(s.find("'gato'") != std::string::npos);
}
