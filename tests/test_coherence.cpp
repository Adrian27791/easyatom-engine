// =============================================================================
// tests/test_coherence.cpp  --  L23
// =============================================================================

#include <stdexcept>
#include <string>
#include <vector>

#include "test_framework.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"
#include "easyatom/reason/coherence.hpp"

using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::hilbert::State;
using easyatom::ops::random_phase_state;
using namespace easyatom::reason;

EATEST_CASE(coherence_corpus_limpio_es_coherente) {
    std::vector<Triplet> ts = {
        {"aspirina", Relation::Treats,    "dolor"},
        {"fiebre",   Relation::IsA,       "sintoma"},
        {"sueno",    Relation::Increases, "energia"},
    };
    EATEST_REQUIRE(is_coherent(ts));
    EATEST_REQUIRE(contradictions(ts).empty());
}

EATEST_CASE(coherence_detecta_causes_vs_inhibits) {
    std::vector<Triplet> ts = {
        {"x", Relation::Causes,   "y"},
        {"x", Relation::Inhibits, "y"},
    };
    auto cs = contradictions(ts);
    EATEST_REQUIRE(cs.size() == 1);
    EATEST_REQUIRE(cs[0].kind == ContradictionKind::CausesVsInhibits);
    EATEST_REQUIRE(!is_coherent(ts));
}

EATEST_CASE(coherence_detecta_increases_vs_decreases) {
    std::vector<Triplet> ts = {
        {"a", Relation::Increases, "b"},
        {"a", Relation::Decreases, "b"},
    };
    auto cs = contradictions(ts);
    EATEST_REQUIRE(cs.size() == 1);
    EATEST_REQUIRE(cs[0].kind == ContradictionKind::IncreasesVsDecreases);
}

EATEST_CASE(coherence_detecta_isa_ciclo) {
    std::vector<Triplet> ts = {
        {"perro", Relation::IsA, "animal"},
        {"animal", Relation::IsA, "perro"},
    };
    auto cs = contradictions(ts);
    EATEST_REQUIRE(cs.size() == 1);
    EATEST_REQUIRE(cs[0].kind == ContradictionKind::IsACycle);
}

EATEST_CASE(coherence_isa_reflexivo_no_es_ciclo) {
    // IsA(X,X) no se reporta como ciclo (X != Y exigido).
    std::vector<Triplet> ts = {
        {"x", Relation::IsA, "x"},
        {"x", Relation::IsA, "x"},
    };
    auto cs = contradictions(ts);
    EATEST_REQUIRE(cs.empty());
}

EATEST_CASE(coherence_project_states_basico) {
    std::vector<State> v = {
        random_phase_state(64, 1),
        random_phase_state(64, 2),
        random_phase_state(64, 3),
    };
    auto pts = project_states(v, 4);
    EATEST_REQUIRE(pts.size() == 3);
    EATEST_REQUIRE(pts[0].size() == 4);
}

EATEST_CASE(coherence_evaluate_addition_acepta_si_no_rompe_nada) {
    std::vector<Triplet> base = {
        {"x", Relation::Causes, "y"},
    };
    std::vector<State> states = { random_phase_state(64, 11) };
    Triplet nuevo{"a", Relation::Treats, "b"};
    State   s_nuevo = random_phase_state(64, 22);
    auto rep = evaluate_addition(base, states, nuevo, s_nuevo, 4, 10.0);
    EATEST_REQUIRE(!rep.has_contradiction);
    EATEST_REQUIRE(rep.accepted);
}

EATEST_CASE(coherence_evaluate_addition_rechaza_contradiccion) {
    std::vector<Triplet> base = {
        {"x", Relation::Causes, "y"},
    };
    std::vector<State> states = { random_phase_state(64, 11) };
    Triplet nuevo{"x", Relation::Inhibits, "y"};
    State   s_nuevo = random_phase_state(64, 22);
    auto rep = evaluate_addition(base, states, nuevo, s_nuevo, 4, 10.0);
    EATEST_REQUIRE(rep.has_contradiction);
    EATEST_REQUIRE(!rep.accepted);
}
