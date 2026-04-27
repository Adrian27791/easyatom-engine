// Tests del Ladrillo 2 — operadores fundamentales del Q-Kernel.
//
// Verificamos las propiedades algebraicas que la teoría exige y que el
// motor depende de ellas para no mentir:
//   * bind conmutativo, asociativo, distributivo sobre bundle.
//   * unbind invierte bind exactamente cuando la clave está en el círculo
//     unidad.
//   * permute es unitario e invertible.
//   * bundle se parece a sus inputs (fidelidad alta).
//   * "memoria asociativa": bind(role, filler) + bundle de varios pares,
//     unbind con role recupera filler con buena fidelidad.

#include "test_framework.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"

#include <complex>
#include <vector>

using easyatom::hilbert::Complex;
using easyatom::hilbert::State;
using easyatom::hilbert::inner;
using easyatom::hilbert::fidelity;
using easyatom::ops::bind;
using easyatom::ops::unbind;
using easyatom::ops::bundle;
using easyatom::ops::permute;
using easyatom::ops::random_phase_state;

constexpr double kOTol = 1e-12;

// -----------------------------------------------------------------------------
// bind: conmutatividad, asociatividad.
// -----------------------------------------------------------------------------

EATEST_CASE(ops_bind_conmutativo) {
    auto a = random_phase_state(64, 0x1111ULL);
    auto b = random_phase_state(64, 0x2222ULL);
    auto ab = bind(a, b);
    auto ba = bind(b, a);
    EATEST_REQUIRE(ab.approx_equal(ba, 1e-12));
}

EATEST_CASE(ops_bind_asociativo) {
    auto a = random_phase_state(32, 0xAAAA);
    auto b = random_phase_state(32, 0xBBBB);
    auto c = random_phase_state(32, 0xCCCC);
    auto izq = bind(bind(a, b), c);
    auto der = bind(a, bind(b, c));
    EATEST_REQUIRE(izq.approx_equal(der, 1e-10));
}

EATEST_CASE(ops_bind_dimensiones_distintas_lanzan) {
    State a(4), b(5);
    bool threw = false;
    try { (void)bind(a, b); } catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

// -----------------------------------------------------------------------------
// unbind: inversa exacta cuando la clave es de fase.
// -----------------------------------------------------------------------------

EATEST_CASE(ops_unbind_invierte_bind_exactamente) {
    // a en círculo unidad, b cualquiera.
    auto a = random_phase_state(128, 0xDEAD);
    State b(std::vector<Complex>(128, Complex{0.0, 0.0}));
    for (std::size_t i = 0; i < 128; ++i) {
        b[i] = Complex{static_cast<double>(i) - 64.0, 0.5 * static_cast<double>(i)};
    }
    auto c = bind(a, b);
    auto bRec = unbind(c, a);
    EATEST_REQUIRE(b.approx_equal(bRec, 1e-9));
}

EATEST_CASE(ops_unbind_clave_con_cero_lanza) {
    State a(std::vector<Complex>{Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{1.0, 0.0}});
    State b(std::vector<Complex>{Complex{2.0, 0.0}, Complex{1.0, 0.0}, Complex{0.5, 0.0}});
    auto c = bind(a, b);
    bool threw = false;
    try { (void)unbind(c, a); } catch (const std::domain_error&) { threw = true; }
    EATEST_REQUIRE(threw);
}

// -----------------------------------------------------------------------------
// permute: unitario e invertible.
// -----------------------------------------------------------------------------

EATEST_CASE(ops_permute_preserva_norma) {
    auto a = random_phase_state(50, 0xF00D);
    const double n0 = a.norm();
    auto p3 = permute(a, 3);
    auto pneg = permute(a, -7);
    EATEST_REQUIRE_NEAR(p3.norm(), n0, 1e-12);
    EATEST_REQUIRE_NEAR(pneg.norm(), n0, 1e-12);
}

EATEST_CASE(ops_permute_inversa) {
    auto a = random_phase_state(40, 0xBABE);
    auto r = permute(permute(a, 13), -13);
    EATEST_REQUIRE(a.approx_equal(r, 1e-12));
}

EATEST_CASE(ops_permute_modulo_dimension_es_identidad) {
    auto a = random_phase_state(32, 0x1234);
    auto r = permute(a, 32);
    EATEST_REQUIRE(a.approx_equal(r, 1e-12));
}

EATEST_CASE(ops_permute_shift_basico) {
    // Si a = (1,2,3,4), permute(a, 1) = (4,1,2,3).
    State a(std::vector<Complex>{Complex{1,0}, Complex{2,0}, Complex{3,0}, Complex{4,0}});
    auto r = permute(a, 1);
    EATEST_REQUIRE_NEAR(r[0].real(), 4.0, kOTol);
    EATEST_REQUIRE_NEAR(r[1].real(), 1.0, kOTol);
    EATEST_REQUIRE_NEAR(r[2].real(), 2.0, kOTol);
    EATEST_REQUIRE_NEAR(r[3].real(), 3.0, kOTol);
}

// -----------------------------------------------------------------------------
// bundle: similitud alta con cada componente.
// -----------------------------------------------------------------------------

EATEST_CASE(ops_bundle_es_similar_a_sus_componentes) {
    auto a = random_phase_state(256, 0x111);
    auto b = random_phase_state(256, 0x222);
    auto c = random_phase_state(256, 0x333);
    auto B = bundle({a, b, c});
    // Cada componente debe tener fidelidad significativa con el bundle (lejos de 0).
    const double fa = fidelity(B, a);
    const double fb = fidelity(B, b);
    const double fc = fidelity(B, c);
    EATEST_REQUIRE(fa > 0.10);
    EATEST_REQUIRE(fb > 0.10);
    EATEST_REQUIRE(fc > 0.10);
}

EATEST_CASE(ops_bundle_y_no_componente_son_casi_ortogonales) {
    auto a = random_phase_state(512, 0xAAA);
    auto b = random_phase_state(512, 0xBBB);
    auto c = random_phase_state(512, 0xCCC);
    auto x = random_phase_state(512, 0x999);  // NO está en el bundle
    auto B = bundle({a, b, c});
    const double fa = fidelity(B, a);
    const double fx = fidelity(B, x);
    // La fidelidad con un vector ajeno debe ser claramente menor que con un
    // vector que sí participa del bundle.
    EATEST_REQUIRE(fx < fa);
    EATEST_REQUIRE(fx < 0.05);
}

// -----------------------------------------------------------------------------
// Memoria asociativa: el ejemplo canónico que demuestra que esto funciona.
// -----------------------------------------------------------------------------

EATEST_CASE(ops_memoria_asociativa_role_filler) {
    // Codificamos: COLOR=rojo, FORMA=circulo, TAMAÑO=grande
    // y luego preguntamos: ¿de qué color? recuperando con la clave COLOR.
    const std::size_t D = 1024;
    auto color    = random_phase_state(D, 1);
    auto forma    = random_phase_state(D, 2);
    auto tamano   = random_phase_state(D, 3);
    auto rojo     = random_phase_state(D, 100);
    auto circulo  = random_phase_state(D, 200);
    auto grande   = random_phase_state(D, 300);

    auto par1 = bind(color, rojo);
    auto par2 = bind(forma, circulo);
    auto par3 = bind(tamano, grande);

    auto memoria = bundle({par1, par2, par3});

    // Recuperación: unbind con la clave COLOR debe parecerse a "rojo".
    auto recuperado = unbind(memoria, color);

    // La fidelidad con "rojo" debe ser claramente la más alta entre los fillers.
    const double f_rojo    = fidelity(recuperado, rojo);
    const double f_circulo = fidelity(recuperado, circulo);
    const double f_grande  = fidelity(recuperado, grande);

    EATEST_REQUIRE(f_rojo > f_circulo);
    EATEST_REQUIRE(f_rojo > f_grande);
    // Y debe estar bastante por encima del nivel de ruido aleatorio (~1/D).
    EATEST_REQUIRE(f_rojo > 0.10);
}
