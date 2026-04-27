// =============================================================================
// tests/test_episodic.cpp  --  L29
// =============================================================================

#include <stdexcept>
#include <vector>

#include "test_framework.hpp"
#include "easyatom/cst/compile.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/memory/episodic.hpp"
#include "easyatom/ops/fundamental.hpp"

using easyatom::cst::CompiledLaw;
using easyatom::cst::Relation;
using easyatom::hilbert::State;
using easyatom::memory::Episode;
using easyatom::memory::EpisodicStore;
using easyatom::ops::random_phase_state;

static Episode mk_ep(std::uint64_t ts, const State& s) {
    Episode e;
    e.ts                = ts;
    e.law.state         = s;
    e.law.triplet.subject  = "x";
    e.law.triplet.relation = Relation::IsA;
    e.law.triplet.object   = "y";
    return e;
}

EATEST_CASE(episodic_append_y_size) {
    EpisodicStore st;
    st.append(mk_ep(10, random_phase_state(16, 1)));
    st.append(mk_ep(20, random_phase_state(16, 2)));
    EATEST_REQUIRE(st.size() == 2);
}

EATEST_CASE(episodic_recall_window_inclusivo) {
    EpisodicStore st;
    st.append(mk_ep(5,  random_phase_state(16, 1)));
    st.append(mk_ep(10, random_phase_state(16, 2)));
    st.append(mk_ep(20, random_phase_state(16, 3)));
    auto w = st.recall_window(10, 20);
    EATEST_REQUIRE(w.size() == 2);
    EATEST_REQUIRE(w[0].ts == 10);
    EATEST_REQUIRE(w[1].ts == 20);
}

EATEST_CASE(episodic_recall_window_t0_mayor_que_t1_lanza) {
    EpisodicStore st;
    bool t = false;
    try { (void)st.recall_window(20, 5); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(episodic_recall_by_content_devuelve_clon_primero) {
    State sA = random_phase_state(32, 11);
    State sB = random_phase_state(32, 22);
    State sC = random_phase_state(32, 33);
    EpisodicStore st;
    st.append(mk_ep(1, sA));
    st.append(mk_ep(2, sB));
    st.append(mk_ep(3, sC));
    auto idxs = st.recall_by_content(sB, 1);
    EATEST_REQUIRE(idxs.size() == 1);
    EATEST_REQUIRE(idxs[0] == 1);
}

EATEST_CASE(episodic_recall_by_content_orden_descendente_por_fidelity) {
    State sA = random_phase_state(32, 11);
    State sB = random_phase_state(32, 22);
    EpisodicStore st;
    st.append(mk_ep(1, sA));
    st.append(mk_ep(2, sB));
    auto idxs = st.recall_by_content(sA, 2);
    EATEST_REQUIRE(idxs.size() == 2);
    EATEST_REQUIRE(idxs[0] == 0);   // sA primero
}

EATEST_CASE(episodic_recall_by_content_k_mayor_que_size_clip) {
    EpisodicStore st;
    st.append(mk_ep(1, random_phase_state(8, 1)));
    auto idxs = st.recall_by_content(random_phase_state(8, 99), 10);
    EATEST_REQUIRE(idxs.size() == 1);
}

EATEST_CASE(episodic_recall_in_window_filtra_por_tiempo) {
    State sA = random_phase_state(32, 11);
    State sB = random_phase_state(32, 22);
    EpisodicStore st;
    st.append(mk_ep(1, sA));
    st.append(mk_ep(50, sA));   // mismo state, ts fuera de ventana
    st.append(mk_ep(2, sB));
    auto idxs = st.recall_by_content_in_window(sA, 0, 10, 5);
    // Solo episodios con ts in [0,10] -> indices 0 y 2.
    EATEST_REQUIRE(idxs.size() == 2);
    EATEST_REQUIRE(idxs[0] == 0);   // sA primero por contenido
}

EATEST_CASE(episodic_recall_by_content_k_cero_lanza) {
    EpisodicStore st;
    st.append(mk_ep(1, random_phase_state(8, 1)));
    bool t = false;
    try { (void)st.recall_by_content(random_phase_state(8, 2), 0); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}
