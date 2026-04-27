// =============================================================================
// tests/test_proof.cpp  --  L26
// =============================================================================

#include <stdexcept>
#include <vector>

#include "test_framework.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/reason/proof.hpp"

using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::reason::check_proof;
using easyatom::reason::ProofStep;
using easyatom::reason::verify_step;

static Triplet T(const char* s, Relation r, const char* o) {
    Triplet t;
    t.subject  = s;
    t.relation = r;
    t.object   = o;
    return t;
}

EATEST_CASE(proof_isa_transitivo_valido) {
    std::vector<Triplet> prem = {
        T("perro", Relation::IsA, "mamifero"),
        T("mamifero", Relation::IsA, "animal"),
    };
    std::vector<ProofStep> steps = {
        {0, 1, T("perro", Relation::IsA, "animal")},
    };
    auto cert = check_proof(prem, steps);
    EATEST_REQUIRE(cert.valid);
    EATEST_REQUIRE(cert.first_invalid_index == 1);
    EATEST_REQUIRE(cert.pool.size() == 3);
}

EATEST_CASE(proof_causes_inhibits_da_inhibits) {
    std::vector<Triplet> prem = {
        T("A", Relation::Causes,   "B"),
        T("B", Relation::Inhibits, "C"),
    };
    std::vector<ProofStep> steps = {
        {0, 1, T("A", Relation::Inhibits, "C")},
    };
    EATEST_REQUIRE(check_proof(prem, steps).valid);
}

EATEST_CASE(proof_isa_propaga_causes) {
    std::vector<Triplet> prem = {
        T("aspirina", Relation::IsA,    "AINE"),
        T("AINE",     Relation::Causes, "irritacion_gastrica"),
    };
    std::vector<ProofStep> steps = {
        {0, 1, T("aspirina", Relation::Causes, "irritacion_gastrica")},
    };
    EATEST_REQUIRE(check_proof(prem, steps).valid);
}

EATEST_CASE(proof_relacion_derivada_incorrecta_falla) {
    std::vector<Triplet> prem = {
        T("A", Relation::Causes, "B"),
        T("B", Relation::Causes, "C"),
    };
    std::vector<ProofStep> steps = {
        {0, 1, T("A", Relation::Inhibits, "C")},   // deberia ser Causes
    };
    auto cert = check_proof(prem, steps);
    EATEST_REQUIRE(!cert.valid);
    EATEST_REQUIRE(cert.first_invalid_index == 0);
}

EATEST_CASE(proof_subject_object_no_encadenan) {
    std::vector<Triplet> prem = {
        T("A", Relation::IsA, "B"),
        T("X", Relation::IsA, "Y"),    // B != X -> no encadena
    };
    std::vector<ProofStep> steps = {
        {0, 1, T("A", Relation::IsA, "Y")},
    };
    EATEST_REQUIRE(!check_proof(prem, steps).valid);
}

EATEST_CASE(proof_indice_fuera_de_rango) {
    std::vector<Triplet> prem = { T("A", Relation::IsA, "B") };
    std::vector<ProofStep> steps = {
        {0, 5, T("A", Relation::IsA, "Z")},
    };
    EATEST_REQUIRE(!check_proof(prem, steps).valid);
}

EATEST_CASE(proof_cadena_dos_pasos_usa_derivado_intermedio) {
    std::vector<Triplet> prem = {
        T("perro",     Relation::IsA, "mamifero"),
        T("mamifero",  Relation::IsA, "animal"),
        T("animal",    Relation::IsA, "ser_vivo"),
    };
    std::vector<ProofStep> steps = {
        {0, 1, T("perro", Relation::IsA, "animal")},      // pool[3]
        {3, 2, T("perro", Relation::IsA, "ser_vivo")},    // usa derivado
    };
    auto cert = check_proof(prem, steps);
    EATEST_REQUIRE(cert.valid);
    EATEST_REQUIRE(cert.pool.size() == 5);
}

EATEST_CASE(proof_segundo_paso_invalido_se_localiza) {
    std::vector<Triplet> prem = {
        T("A", Relation::IsA, "B"),
        T("B", Relation::IsA, "C"),
        T("C", Relation::IsA, "D"),
    };
    std::vector<ProofStep> steps = {
        {0, 1, T("A", Relation::IsA, "C")},          // ok
        {3, 2, T("A", Relation::Causes, "D")},       // mal: deberia IsA
    };
    auto cert = check_proof(prem, steps);
    EATEST_REQUIRE(!cert.valid);
    EATEST_REQUIRE(cert.first_invalid_index == 1);
    EATEST_REQUIRE(cert.pool.size() == 4);   // contiene el primer derivado
}
