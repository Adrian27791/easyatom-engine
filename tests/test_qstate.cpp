// Tests del Ladrillo C — CState (qubits sintéticos con amplitudes complejas).

#include "test_framework.hpp"
#include "easyatom/qstate/cstate.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>

using easyatom::qstate::CState;
using easyatom::qstate::Complex;
using easyatom::qstate::kPi_C;

EATEST_CASE(cstate_init_es_estado_cero) {
    CState s(3);
    EATEST_REQUIRE(s.dim() == 8);
    EATEST_REQUIRE_NEAR(s.norm_squared(), 1.0, 1e-12);
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[0], 1.0, 1e-12);
    for (std::size_t i = 1; i < 8; ++i) EATEST_REQUIRE_NEAR(p[i], 0.0, 1e-12);
}

EATEST_CASE(cstate_X_voltea_qubit) {
    CState s(2);
    s.apply_X(0);
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[1], 1.0, 1e-12);  // |01> en little-endian -> idx 1
    s.apply_X(1);
    p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[3], 1.0, 1e-12);  // |11>
}

EATEST_CASE(cstate_H_crea_superposicion) {
    CState s(1);
    s.apply_H(0);
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[0], 0.5, 1e-12);
    EATEST_REQUIRE_NEAR(p[1], 0.5, 1e-12);
    EATEST_REQUIRE_NEAR(s.norm_squared(), 1.0, 1e-12);
}

EATEST_CASE(cstate_HH_es_identidad) {
    CState s(1);
    s.apply_X(0);  // |1>
    s.apply_H(0);
    s.apply_H(0);
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[1], 1.0, 1e-10);
}

EATEST_CASE(cstate_Z_da_fase_a_uno) {
    CState s(1);
    s.apply_H(0);
    s.apply_Z(0);
    s.apply_H(0);
    // H Z H = X, así que |0> -> |1>
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[1], 1.0, 1e-10);
}

EATEST_CASE(cstate_Y_es_iXZ) {
    CState a(1); a.apply_Y(0);
    CState b(1); b.apply_Z(0); b.apply_X(0);
    // Hasta fase global: |<a|b>|^2 == 1.
    EATEST_REQUIRE_NEAR(a.fidelity(b), 1.0, 1e-10);
}

EATEST_CASE(cstate_Rx_pi_es_X_salvo_fase) {
    CState a(1); a.apply_Rx(0, kPi_C);
    CState b(1); b.apply_X(0);
    EATEST_REQUIRE_NEAR(a.fidelity(b), 1.0, 1e-10);
}

EATEST_CASE(cstate_Ry_dos_pi_es_identidad_modulo_fase) {
    CState s(1);
    s.apply_Ry(0, 2.0 * kPi_C);
    CState ref(1);
    EATEST_REQUIRE_NEAR(s.fidelity(ref), 1.0, 1e-10);
}

EATEST_CASE(cstate_Rz_no_cambia_probabilidades) {
    CState s(1); s.apply_H(0);
    auto p_before = s.probabilities();
    s.apply_Rz(0, 1.234);
    auto p_after = s.probabilities();
    EATEST_REQUIRE_NEAR(p_before[0], p_after[0], 1e-12);
    EATEST_REQUIRE_NEAR(p_before[1], p_after[1], 1e-12);
}

EATEST_CASE(cstate_CNOT_propaga_uno) {
    CState s(2); s.apply_X(0);  // |01>
    s.apply_CNOT(0, 1);          // -> |11>
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[3], 1.0, 1e-12);
}

EATEST_CASE(cstate_bell_es_entrelazado) {
    auto s = easyatom::qstate::make_bell_phi_plus();
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[0], 0.5, 1e-12);  // |00>
    EATEST_REQUIRE_NEAR(p[3], 0.5, 1e-12);  // |11>
    EATEST_REQUIRE_NEAR(p[1], 0.0, 1e-12);
    EATEST_REQUIRE_NEAR(p[2], 0.0, 1e-12);
    EATEST_REQUIRE_NEAR(s.norm_squared(), 1.0, 1e-12);
}

EATEST_CASE(cstate_bell_correlacion_post_medida) {
    auto s = easyatom::qstate::make_bell_phi_plus();
    int q0 = s.measure_qubit(0, 42);
    // Tras medir q0, q1 debe ser determinista e igual a q0.
    EATEST_REQUIRE_NEAR(s.prob_of_one(1), q0 == 1 ? 1.0 : 0.0, 1e-10);
}

EATEST_CASE(cstate_GHZ_correlacion_n_qubits) {
    auto s = easyatom::qstate::make_ghz(4);
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[0],  0.5, 1e-12);
    EATEST_REQUIRE_NEAR(p[15], 0.5, 1e-12);
    int q0 = s.measure_qubit(0, 7);
    // Todos los demás colapsan al mismo valor.
    for (std::size_t q = 1; q < 4; ++q) {
        EATEST_REQUIRE_NEAR(s.prob_of_one(q), q0 == 1 ? 1.0 : 0.0, 1e-10);
    }
}

EATEST_CASE(cstate_measure_all_es_consistente_con_born) {
    // Estado: 3/5 |0> + 4/5 |1>.
    CState s(1);
    auto& a = s.amplitudes_mut();
    a[0] = Complex{0.6, 0.0};
    a[1] = Complex{0.8, 0.0};
    EATEST_REQUIRE_NEAR(s.norm_squared(), 1.0, 1e-12);
    int ones = 0;
    const int N = 4000;
    for (int i = 0; i < N; ++i) {
        CState s2(1);
        auto& a2 = s2.amplitudes_mut();
        a2[0] = Complex{0.6, 0.0};
        a2[1] = Complex{0.8, 0.0};
        ones += static_cast<int>(s2.measure_all(static_cast<std::uint64_t>(i + 1)));
    }
    const double freq = static_cast<double>(ones) / N;
    // p = 0.64. Tolerancia ancha por muestreo finito.
    EATEST_REQUIRE(std::fabs(freq - 0.64) < 0.03);
}

EATEST_CASE(cstate_swap_intercambia_qubits) {
    CState s(2); s.apply_X(0);   // |01>
    s.apply_SWAP(0, 1);
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[2], 1.0, 1e-12);  // |10>
}

EATEST_CASE(cstate_CZ_solo_marca_11) {
    CState s(2);
    s.apply_X(0); s.apply_X(1);   // |11>
    s.apply_CZ(0, 1);
    EATEST_REQUIRE_NEAR(std::norm(s.amplitudes()[3]), 1.0, 1e-12);
    // Fase -1.
    EATEST_REQUIRE(s.amplitudes()[3].real() < 0.0);
}

EATEST_CASE(cstate_interferencia_destructiva_via_H) {
    // H|0> + H sobre el estado tras interferencia debe volver a |0>.
    CState s(1);
    s.apply_H(0);  // (|0>+|1>)/sqrt2
    s.apply_H(0);  // |0>
    auto p = s.probabilities();
    EATEST_REQUIRE_NEAR(p[0], 1.0, 1e-12);
    EATEST_REQUIRE_NEAR(p[1], 0.0, 1e-12);
}

EATEST_CASE(cstate_normas_tras_secuencia_compleja) {
    CState s(4);
    s.apply_H(0); s.apply_H(1); s.apply_H(2); s.apply_H(3);
    s.apply_CNOT(0, 1); s.apply_CNOT(2, 3);
    s.apply_Rx(0, 0.7); s.apply_Ry(2, 1.3); s.apply_Rz(3, -0.4);
    s.apply_CZ(1, 2);
    EATEST_REQUIRE_NEAR(s.norm_squared(), 1.0, 1e-10);
}

EATEST_CASE(cstate_CNOT_control_igual_target_lanza) {
    CState s(2);
    bool threw = false;
    try { s.apply_CNOT(0, 0); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(cstate_qubit_fuera_de_rango_lanza) {
    CState s(2);
    bool threw = false;
    try { s.apply_X(5); }
    catch (const std::out_of_range&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(cstate_n_qubits_invalido_lanza) {
    bool threw = false;
    try { CState s(0); (void)s; }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}
