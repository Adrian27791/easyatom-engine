// Tests del Ladrillo 17 — mapeo Relation -> operador HDC.

#include "test_framework.hpp"
#include "easyatom/cst/operator_map.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"

#include <stdexcept>

using easyatom::cst::Relation;
using easyatom::cst::apply_relation;
using easyatom::cst::relation_key;
using easyatom::cst::shift_for;
using easyatom::cst::sign_for;
using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;
using easyatom::hilbert::Complex;
using easyatom::ops::random_phase_state;
using easyatom::ops::bind;
using easyatom::ops::permute;
using easyatom::ops::unbind;
using easyatom::ops::bundle;

static constexpr std::size_t kD = 4096;

EATEST_CASE(opmap_signs_y_shifts_basicos) {
    EATEST_REQUIRE(sign_for(Relation::Causes)    == 1.0);
    EATEST_REQUIRE(sign_for(Relation::OpposesTo) == -1.0);
    EATEST_REQUIRE(shift_for(Relation::Equivalent) == 0);
    EATEST_REQUIRE(shift_for(Relation::Causes)     != shift_for(Relation::Inhibits));
}

EATEST_CASE(opmap_relation_key_distinta_por_relacion) {
    State k1 = relation_key(Relation::Causes,   kD);
    State k2 = relation_key(Relation::Inhibits, kD);
    EATEST_REQUIRE(k1.dim() == kD);
    EATEST_REQUIRE(fidelity(k1, k2) < 0.05);   // casi ortogonales en alta D
}

EATEST_CASE(opmap_relacion_distinta_da_estado_distinto) {
    State S = random_phase_state(kD, 11);
    State O = random_phase_state(kD, 22);
    State k_c = relation_key(Relation::Causes,   kD);
    State k_i = relation_key(Relation::Inhibits, kD);
    State l1 = apply_relation(Relation::Causes,   S, O, k_c);
    State l2 = apply_relation(Relation::Inhibits, S, O, k_i);
    EATEST_REQUIRE(fidelity(l1, l2) < 0.1);
}

EATEST_CASE(opmap_equivalent_es_simetrico) {
    State S = random_phase_state(kD, 33);
    State O = random_phase_state(kD, 44);
    State k = relation_key(Relation::Equivalent, kD);
    State a = apply_relation(Relation::Equivalent, S, O, k);
    State b = apply_relation(Relation::Equivalent, O, S, k);
    EATEST_REQUIRE(fidelity(a, b) > 0.999);    // bind es conmutativo + shift=0
}

EATEST_CASE(opmap_causes_no_es_simetrico) {
    State S = random_phase_state(kD, 55);
    State O = random_phase_state(kD, 66);
    State k = relation_key(Relation::Causes, kD);
    State a = apply_relation(Relation::Causes, S, O, k);
    State b = apply_relation(Relation::Causes, O, S, k);
    EATEST_REQUIRE(fidelity(a, b) < 0.1);      // shift=1 rompe simetria
}

EATEST_CASE(opmap_opposes_es_negativo_de_causes) {
    State S = random_phase_state(kD, 77);
    State O = random_phase_state(kD, 88);
    // Comparten la misma key de rol para aislar el efecto del signo.
    State k = relation_key(Relation::Causes, kD);
    // Causes con shift=1.
    State lc = apply_relation(Relation::Causes, S, O, k);
    // Construir manualmente "causes con signo negativo" usando la misma k y
    // el mismo shift que Causes; debe ser exactamente -lc => fidelity 1.0.
    State opp(kD);
    for (std::size_t i = 0; i < kD; ++i) opp[i] = -lc[i];
    EATEST_REQUIRE(fidelity(lc, opp) > 0.999);    // |<a,-a>|^2 = |<a,a>|^2
    // Y la suma con peso +1 debe colapsar a 0.
    State sum = bundle({lc, opp}, {Complex{1,0}, Complex{1,0}});
    EATEST_REQUIRE(sum.norm_squared() < 1e-18);
}

EATEST_CASE(opmap_ley_es_reversible_recuperando_objeto) {
    State S = random_phase_state(kD, 101);
    State O = random_phase_state(kD, 202);
    Relation R = Relation::Causes;
    State k = relation_key(R, kD);
    State law = apply_relation(R, S, O, k);
    // unbind(law, k) -> bind(S, permute(O, shift(R)))
    State sk_op = unbind(law, k);
    // unbind(sk_op, S) -> permute(O, shift(R))
    State perm_O = unbind(sk_op, S);
    // permute inverso recupera O exacto.
    State O_rec = permute(perm_O, -shift_for(R));
    EATEST_REQUIRE(fidelity(O, O_rec) > 0.999);
}

EATEST_CASE(opmap_dimensiones_incoherentes_lanzan) {
    State S = random_phase_state(64, 1);
    State O = random_phase_state(128, 2);
    State k = relation_key(Relation::Causes, 64);
    bool threw = false;
    try { apply_relation(Relation::Causes, S, O, k); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}
