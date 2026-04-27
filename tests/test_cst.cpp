// Tests del Ladrillo 16 — CST etapa 1 (verbos + extractor de tripletas).

#include "test_framework.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/cst/triplet.hpp"

#include <stdexcept>
#include <string>

using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::cst::classify_verb;
using easyatom::cst::extract;
using easyatom::cst::extract_all;

EATEST_CASE(cst_verbs_catalogo_no_vacio_y_minimo_30_lemas) {
    EATEST_REQUIRE(easyatom::cst::catalog_size() >= 30);
}

EATEST_CASE(cst_verbs_clasifica_es_y_en_principales) {
    EATEST_REQUIRE(classify_verb("disminuye") == Relation::Decreases);
    EATEST_REQUIRE(classify_verb("decreases") == Relation::Decreases);
    EATEST_REQUIRE(classify_verb("causa")     == Relation::Causes);
    EATEST_REQUIRE(classify_verb("triggers")  == Relation::Causes);
    EATEST_REQUIRE(classify_verb("inhibe")    == Relation::Inhibits);
    EATEST_REQUIRE(classify_verb("trata")     == Relation::Treats);
}

EATEST_CASE(cst_verbs_desconocido_devuelve_unknown) {
    EATEST_REQUIRE(classify_verb("xyzqwerty") == Relation::Unknown);
}

EATEST_CASE(cst_extract_frase_clinica_simple_es) {
    Triplet t = extract("la insulina disminuye la glucosa");
    EATEST_REQUIRE(t.valid());
    EATEST_REQUIRE(t.subject  == "insulina");
    EATEST_REQUIRE(t.relation == Relation::Decreases);
    EATEST_REQUIRE(t.object   == "glucosa");
}

EATEST_CASE(cst_extract_frase_simple_en) {
    Triplet t = extract("paracetamol treats fever");
    EATEST_REQUIRE(t.valid());
    EATEST_REQUIRE(t.subject  == "paracetamol");
    EATEST_REQUIRE(t.relation == Relation::Treats);
    EATEST_REQUIRE(t.object   == "fever");
}

EATEST_CASE(cst_extract_sin_verbo_es_invalido) {
    Triplet t = extract("solo unas palabras sueltas");
    EATEST_REQUIRE(!t.valid());
}

EATEST_CASE(cst_extract_normaliza_acentos_y_mayusculas) {
    Triplet t = extract("La INSULINA disminuye la Glucosa.");
    EATEST_REQUIRE(t.valid());
    EATEST_REQUIRE(t.subject == "insulina");
    EATEST_REQUIRE(t.object  == "glucosa");
}

EATEST_CASE(cst_extract_all_separa_por_punto_y_punto_y_coma) {
    auto v = extract_all(
        "insulina disminuye glucosa. ejercicio aumenta pulso; "
        "frase invalida sin verbo.");
    EATEST_REQUIRE(v.size() == 2);
    EATEST_REQUIRE(v[0].relation == Relation::Decreases);
    EATEST_REQUIRE(v[1].relation == Relation::Increases);
    EATEST_REQUIRE(v[1].subject  == "ejercicio");
    EATEST_REQUIRE(v[1].object   == "pulso");
}
