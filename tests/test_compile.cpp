// Tests del Ladrillo 20 — orquestador compile_law(text).

#include "test_framework.hpp"
#include "easyatom/cst/compile.hpp"
#include "easyatom/qkernel/qkernel.hpp"

#include <stdexcept>

using easyatom::cst::CompileError;
using easyatom::cst::CompiledLaw;
using easyatom::cst::compile_corpus;
using easyatom::cst::compile_law;
using easyatom::cst::Relation;
using easyatom::cst::fingerprint;
using easyatom::qkernel::QKernel;

EATEST_CASE(compile_law_extrae_y_compila_una_frase_clinica) {
    QKernel k(4096, 0xC0DECAFEULL);
    CompiledLaw L = compile_law(k, "la insulina disminuye la glucosa");
    EATEST_REQUIRE(L.triplet.subject == "insulina");
    EATEST_REQUIRE(L.triplet.relation == Relation::Decreases);
    EATEST_REQUIRE(L.triplet.object == "glucosa");
    EATEST_REQUIRE(L.state.dim() == 4096);
    EATEST_REQUIRE(L.fingerprint != 0);
    EATEST_REQUIRE(L.provenance_hash != 0);
}

EATEST_CASE(compile_law_es_idempotente_misma_frase_misma_huella) {
    QKernel k(2048, 7ULL);
    CompiledLaw a = compile_law(k, "paracetamol treats fever");
    CompiledLaw b = compile_law(k, "paracetamol treats fever");
    EATEST_REQUIRE(a.fingerprint == b.fingerprint);
    EATEST_REQUIRE(a.provenance_hash == b.provenance_hash);
}

EATEST_CASE(compile_law_distinta_frase_distinta_huella) {
    QKernel k(2048, 7ULL);
    CompiledLaw a = compile_law(k, "insulina disminuye glucosa");
    CompiledLaw b = compile_law(k, "ejercicio aumenta pulso");
    EATEST_REQUIRE(a.fingerprint     != b.fingerprint);
    EATEST_REQUIRE(a.provenance_hash != b.provenance_hash);
}

EATEST_CASE(compile_law_ingest_es_idempotente_no_duplica_codebook) {
    QKernel k(1024, 1ULL);
    compile_law(k, "insulina disminuye glucosa");
    const auto sz1 = k.codebook_size();
    compile_law(k, "insulina disminuye glucosa");   // no añade simbolos nuevos
    EATEST_REQUIRE(k.codebook_size() == sz1);
    EATEST_REQUIRE(k.contains("insulina"));
    EATEST_REQUIRE(k.contains("glucosa"));
}

EATEST_CASE(compile_law_frase_sin_verbo_lanza_compileerror) {
    QKernel k(512, 1ULL);
    bool t = false;
    try { compile_law(k, "solo unas palabras sueltas sin verbo"); }
    catch (const CompileError&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(compile_corpus_compila_varias_y_omite_invalidas) {
    QKernel k(1024, 99ULL);
    auto v = compile_corpus(k,
        "insulina disminuye glucosa. ejercicio aumenta pulso; "
        "frase invalida sin verbo. paracetamol trata fiebre.");
    EATEST_REQUIRE(v.size() == 3);
    EATEST_REQUIRE(v[0].triplet.relation == Relation::Decreases);
    EATEST_REQUIRE(v[1].triplet.relation == Relation::Increases);
    EATEST_REQUIRE(v[2].triplet.relation == Relation::Treats);
}

EATEST_CASE(compile_law_provenance_distinto_pero_fingerprint_igual_si_compila_a_lo_mismo) {
    // Dos textos con misma forma normalizada compilan al mismo state.
    QKernel k(512, 5ULL);
    CompiledLaw a = compile_law(k, "insulina disminuye glucosa");
    CompiledLaw b = compile_law(k, "La INSULINA disminuye la Glucosa");
    EATEST_REQUIRE(a.fingerprint == b.fingerprint);
    EATEST_REQUIRE(a.provenance_hash != b.provenance_hash);  // texto distinto
}

EATEST_CASE(compile_law_fingerprint_state_es_estable) {
    QKernel k(256, 3ULL);
    CompiledLaw L = compile_law(k, "insulina disminuye glucosa");
    EATEST_REQUIRE(fingerprint(L.state) == L.fingerprint);
}
