// Tests del Ladrillo 21 — generador de teoremas.

#include "test_framework.hpp"
#include "easyatom/cst/compile.hpp"
#include "easyatom/qkernel/qkernel.hpp"
#include "easyatom/reason/theorems.hpp"

#include <algorithm>
#include <stdexcept>

using easyatom::cst::CompiledLaw;
using easyatom::cst::compile_corpus;
using easyatom::cst::Relation;
using easyatom::qkernel::QKernel;
using easyatom::reason::compile_theorems;
using easyatom::reason::derive_theorems;
using easyatom::reason::infer;
using easyatom::reason::Theorem;

EATEST_CASE(theorems_infer_tabla_basica) {
    EATEST_REQUIRE(infer(Relation::IsA,       Relation::IsA)        == Relation::IsA);
    EATEST_REQUIRE(infer(Relation::Causes,    Relation::Causes)     == Relation::Causes);
    EATEST_REQUIRE(infer(Relation::Decreases, Relation::Decreases)  == Relation::Increases);
    EATEST_REQUIRE(infer(Relation::Increases, Relation::Decreases)  == Relation::Decreases);
    EATEST_REQUIRE(infer(Relation::Causes,    Relation::Inhibits)   == Relation::Inhibits);
    EATEST_REQUIRE(infer(Relation::IsA,       Relation::HasProperty)== Relation::HasProperty);
    EATEST_REQUIRE(infer(Relation::Equivalent,Relation::Causes)     == Relation::Causes);
    EATEST_REQUIRE(infer(Relation::Treats,    Relation::Treats)     == Relation::Unknown);
    EATEST_REQUIRE(infer(Relation::Unknown,   Relation::Causes)     == Relation::Unknown);
}

EATEST_CASE(theorems_transitividad_de_isa) {
    QKernel k(512, 1ULL);
    auto base = compile_corpus(k, "perro es mamifero. mamifero es animal.");
    EATEST_REQUIRE(base.size() == 2);
    auto th = derive_theorems(base, 2);
    bool found = false;
    for (const auto& t : th) {
        if (t.triplet.subject == "perro" &&
            t.triplet.relation == Relation::IsA &&
            t.triplet.object  == "animal") { found = true; break; }
    }
    EATEST_REQUIRE(found);
}

EATEST_CASE(theorems_doble_negacion_decreases_decreases) {
    QKernel k(512, 2ULL);
    auto base = compile_corpus(k, "estres disminuye serotonina. serotonina disminuye ansiedad.");
    auto th = derive_theorems(base, 2);
    bool found = false;
    for (const auto& t : th) {
        if (t.triplet.subject == "estres" &&
            t.triplet.relation == Relation::Increases &&
            t.triplet.object  == "ansiedad") { found = true; break; }
    }
    EATEST_REQUIRE(found);
}

EATEST_CASE(theorems_substitucion_por_clase_isa_propaga) {
    QKernel k(512, 3ULL);
    // perro es mamifero ; mamifero tiene pelo  =>  perro tiene pelo
    auto base = compile_corpus(k, "perro es mamifero. mamifero tiene pelo.");
    auto th = derive_theorems(base, 2);
    bool found = false;
    for (const auto& t : th) {
        if (t.triplet.subject == "perro" &&
            t.triplet.relation == Relation::HasProperty &&
            t.triplet.object  == "pelo") { found = true; break; }
    }
    EATEST_REQUIRE(found);
}

EATEST_CASE(theorems_no_inventa_si_no_hay_regla) {
    QKernel k(512, 4ULL);
    auto base = compile_corpus(k, "paracetamol trata fiebre. fiebre trata gripe.");
    // (Treats, Treats) -> Unknown en la tabla; no debe aparecer paracetamol-Treats-gripe.
    auto th = derive_theorems(base, 2);
    for (const auto& t : th) {
        bool bad = (t.triplet.subject == "paracetamol" &&
                    t.triplet.relation == Relation::Treats &&
                    t.triplet.object == "gripe");
        EATEST_REQUIRE(!bad);
    }
}

EATEST_CASE(theorems_no_genera_autoloops) {
    QKernel k(512, 5ULL);
    auto base = compile_corpus(k, "a es b. b es a.");   // ciclo
    auto th = derive_theorems(base, 3);
    for (const auto& t : th) {
        EATEST_REQUIRE(t.triplet.subject != t.triplet.object);
    }
}

EATEST_CASE(theorems_compile_theorems_genera_compiledlaws_validas) {
    QKernel k(1024, 6ULL);
    auto base = compile_corpus(k, "perro es mamifero. mamifero es animal.");
    auto th   = derive_theorems(base, 2);
    EATEST_REQUIRE(!th.empty());
    auto laws = compile_theorems(k, th);
    EATEST_REQUIRE(laws.size() == th.size());
    for (const auto& L : laws) {
        EATEST_REQUIRE(L.state.dim() == 1024);
        EATEST_REQUIRE(L.fingerprint != 0);
        EATEST_REQUIRE(L.provenance_hash == 0);   // teorema derivado
    }
}

EATEST_CASE(theorems_max_depth_cero_no_genera) {
    QKernel k(256, 7ULL);
    auto base = compile_corpus(k, "perro es mamifero. mamifero es animal.");
    auto th   = derive_theorems(base, 0);
    EATEST_REQUIRE(th.empty());
}
