// =============================================================================
// tests/test_gap.cpp  --  L24
// =============================================================================

#include <stdexcept>
#include <vector>

#include "test_framework.hpp"
#include "easyatom/cst/compile.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/epistemic/gap.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"

using easyatom::cst::CompiledLaw;
using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;
using easyatom::ops::random_phase_state;
using namespace easyatom::epistemic;

static CompiledLaw make_law(const std::string& s, Relation r,
                            const std::string& o, std::uint64_t seed) {
    CompiledLaw l;
    l.triplet = Triplet{s, r, o};
    l.state   = random_phase_state(64, seed);
    l.fingerprint     = seed;
    l.provenance_hash = 0;
    return l;
}

EATEST_CASE(gap_density_codebook_vacio_es_cero) {
    State q = random_phase_state(64, 1);
    EATEST_REQUIRE(density(q, {}) == 0.0);
}

EATEST_CASE(gap_density_consigo_mismo_es_uno) {
    State q = random_phase_state(64, 7);
    CompiledLaw l;
    l.triplet = Triplet{"a", Relation::Causes, "b"};
    l.state   = q;
    EATEST_REQUIRE(std::abs(density(q, {l}) - 1.0) < 1e-9);
}

EATEST_CASE(gap_find_gaps_detecta_hueco) {
    std::vector<CompiledLaw> book = {
        make_law("a", Relation::Causes, "b", 1),
        make_law("c", Relation::Treats, "d", 2),
    };
    // query lejana al book.
    std::vector<State> queries = { random_phase_state(64, 999) };
    auto gaps = find_gaps(queries, book, 0.9);
    EATEST_REQUIRE(gaps.size() == 1);
    EATEST_REQUIRE(gaps[0].query_index == 0);
    EATEST_REQUIRE(gaps[0].density < 0.9);
}

EATEST_CASE(gap_find_gaps_no_falsos_positivos_si_query_esta_dentro) {
    auto law = make_law("x", Relation::Causes, "y", 11);
    std::vector<CompiledLaw> book = { law };
    std::vector<State> queries = { law.state };  // exactamente dentro
    auto gaps = find_gaps(queries, book, 0.5);
    EATEST_REQUIRE(gaps.empty());
}

EATEST_CASE(gap_find_gaps_theta_negativo_lanza) {
    bool t = false;
    try { (void)find_gaps({}, {}, -0.1); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(gap_try_fill_codebook_vacio_devuelve_nullopt) {
    State q = random_phase_state(64, 3);
    auto r = try_fill({}, q, 3);
    EATEST_REQUIRE(!r.has_value());
}

EATEST_CASE(gap_try_fill_propuesta_es_mas_cercana_que_la_media) {
    std::vector<CompiledLaw> book = {
        make_law("a", Relation::Causes,    "b", 1),
        make_law("c", Relation::Treats,    "d", 2),
        make_law("e", Relation::Increases, "f", 3),
        make_law("g", Relation::IsA,       "h", 4),
    };
    State q = random_phase_state(64, 50);
    const double base = density(q, book);
    auto r = try_fill(book, q, 2);
    EATEST_REQUIRE(r.has_value());
    // El bundle de los top-2 vecinos debe acercarse mas a q que la media global.
    EATEST_REQUIRE(fidelity(r->state, q) >= base);
}

EATEST_CASE(gap_try_fill_marca_provenance_como_sintetica) {
    std::vector<CompiledLaw> book = {
        make_law("a", Relation::Causes, "b", 1),
    };
    State q = random_phase_state(64, 99);
    auto r = try_fill(book, q, 1);
    EATEST_REQUIRE(r.has_value());
    EATEST_REQUIRE(r->provenance_hash == 0);  // sintetico
    EATEST_REQUIRE(r->triplet.subject.rfind("gap__", 0) == 0);
    EATEST_REQUIRE(r->triplet.object.rfind("gap__", 0) == 0);
    EATEST_REQUIRE(r->triplet.relation == Relation::Causes);  // hereda del top-1
}
