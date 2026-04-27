// =============================================================================
// tests/test_hypothesis.cpp  --  L27
// =============================================================================

#include <stdexcept>
#include <vector>

#include "test_framework.hpp"
#include "easyatom/cst/compile.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/hypothesis/engine.hpp"
#include "easyatom/ops/fundamental.hpp"

using easyatom::cst::CompiledLaw;
using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::hilbert::State;
using easyatom::hypothesis::HypothesisChoice;
using easyatom::hypothesis::iterate_hypothesis;
using easyatom::hypothesis::split_holdout;
using easyatom::hypothesis::validate;
using easyatom::ops::random_phase_state;

static CompiledLaw make_law(const State& s, const char* subj, const char* obj) {
    CompiledLaw L;
    L.triplet.subject  = subj;
    L.triplet.relation = Relation::IsA;
    L.triplet.object   = obj;
    L.state            = s;
    return L;
}

EATEST_CASE(hypo_split_holdout_determinista) {
    std::vector<CompiledLaw> corpus;
    for (std::size_t i = 0; i < 6; ++i) {
        corpus.push_back(make_law(random_phase_state(32, i + 1), "a", "b"));
    }
    auto h = split_holdout(corpus, 4);
    EATEST_REQUIRE(h.train.size() == 4);
    EATEST_REQUIRE(h.test.size() == 2);
}

EATEST_CASE(hypo_split_holdout_validaciones) {
    std::vector<CompiledLaw> corpus = {
        make_law(random_phase_state(8, 1), "a", "b"),
        make_law(random_phase_state(8, 2), "a", "b"),
    };
    bool t1 = false; try { (void)split_holdout({}, 1); }
                     catch (const std::invalid_argument&) { t1 = true; }
    bool t2 = false; try { (void)split_holdout(corpus, 0); }
                     catch (const std::invalid_argument&) { t2 = true; }
    bool t3 = false; try { (void)split_holdout(corpus, 2); }
                     catch (const std::invalid_argument&) { t3 = true; }
    EATEST_REQUIRE(t1 && t2 && t3);
}

EATEST_CASE(hypo_validate_perfecta_con_clones) {
    State sA = random_phase_state(32, 1);
    State sB = random_phase_state(32, 2);
    std::vector<CompiledLaw> train = {
        make_law(sA, "a", "x"), make_law(sB, "b", "y"),
    };
    std::vector<CompiledLaw> test = {
        make_law(sA, "a", "x"), make_law(sB, "b", "y"),
    };
    auto r = validate(train, test, 0.4);
    EATEST_REQUIRE(r.total == 2);
    EATEST_REQUIRE(r.hits  == 2);
    EATEST_REQUIRE(r.accuracy > 0.99);
}

EATEST_CASE(hypo_validate_threshold_invalido_lanza) {
    bool t1 = false; try { (void)validate({}, {}, -0.1); }
                     catch (const std::invalid_argument&) { t1 = true; }
    bool t2 = false; try { (void)validate({}, {}, 1.5); }
                     catch (const std::invalid_argument&) { t2 = true; }
    EATEST_REQUIRE(t1 && t2);
}

EATEST_CASE(hypo_validate_test_vacio_devuelve_zero) {
    std::vector<CompiledLaw> train = {
        make_law(random_phase_state(8, 1), "a", "b"),
    };
    auto r = validate(train, {}, 0.5);
    EATEST_REQUIRE(r.total == 0);
    EATEST_REQUIRE(r.hits  == 0);
    EATEST_REQUIRE(r.accuracy == 0.0);
}

EATEST_CASE(hypo_iterate_elige_candidato_que_mas_mejora) {
    State sA = random_phase_state(32, 11);
    State sB = random_phase_state(32, 22);
    State sC = random_phase_state(32, 33);
    State sD = random_phase_state(32, 44);

    // Train inicial: solo sA (rara vez explica B,C,D).
    std::vector<CompiledLaw> train = { make_law(sA, "a", "x") };
    std::vector<CompiledLaw> test  = {
        make_law(sB, "b", "y"),
        make_law(sC, "c", "z"),
    };
    // Candidatos: clon de sB, clon de sC, ruido sD.
    std::vector<CompiledLaw> cands = {
        make_law(sB, "b", "y"),
        make_law(sC, "c", "z"),
        make_law(sD, "d", "w"),
    };
    auto choice = iterate_hypothesis(train, test, cands, 0.4);
    EATEST_REQUIRE(choice.candidate_index != HypothesisChoice::npos);
    EATEST_REQUIRE(choice.candidate_index <= 1);   // clon de B o C
    EATEST_REQUIRE(choice.delta_accuracy > 0.0);
    EATEST_REQUIRE(choice.best_report.accuracy >
                   choice.baseline_report.accuracy);
}

EATEST_CASE(hypo_iterate_sin_mejora_devuelve_npos) {
    State sA = random_phase_state(32, 1);
    State sB = random_phase_state(32, 2);
    std::vector<CompiledLaw> train = { make_law(sA, "a", "x") };
    std::vector<CompiledLaw> test  = { make_law(sA, "a", "x") };  // ya hit
    std::vector<CompiledLaw> cands = { make_law(sB, "b", "y") };
    auto choice = iterate_hypothesis(train, test, cands, 0.4);
    EATEST_REQUIRE(choice.baseline_report.accuracy > 0.99);
    EATEST_REQUIRE(choice.candidate_index == HypothesisChoice::npos);
    EATEST_REQUIRE(choice.delta_accuracy == 0.0);
}

EATEST_CASE(hypo_iterate_candidatos_vacios_no_mejora) {
    State sA = random_phase_state(32, 1);
    std::vector<CompiledLaw> train = { make_law(sA, "a", "x") };
    std::vector<CompiledLaw> test  = { make_law(sA, "a", "x") };
    auto choice = iterate_hypothesis(train, test, {}, 0.4);
    EATEST_REQUIRE(choice.candidate_index == HypothesisChoice::npos);
    EATEST_REQUIRE(choice.delta_accuracy == 0.0);
}
