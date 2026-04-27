// Tests de Ladrillo 0 — multivectores de Clifford Cl(p,q).
//
// Verificamos las identidades fundamentales del álgebra geométrica con
// tolerancia numérica estricta. Si cualquiera falla, el ladrillo 0 no se da
// por bueno.

#include "test_framework.hpp"

#include "easyatom/clifford/multivector.hpp"

using easyatom::clifford::G2;
using easyatom::clifford::G3;
using easyatom::clifford::STA;
using easyatom::clifford::blade_merge_sign;

constexpr double kTol = 1e-12;

// -----------------------------------------------------------------------------
// Helpers numéricos del cálculo de signos.
// -----------------------------------------------------------------------------

EATEST_CASE(blade_merge_sign_basico) {
    // (e0)(e1)  -> orden ya canónico: signo +1
    EATEST_REQUIRE(blade_merge_sign(0b001, 0b010) == +1);
    // (e1)(e0) -> hace falta una transposición: signo -1
    EATEST_REQUIRE(blade_merge_sign(0b010, 0b001) == -1);
    // (e0 e1)(e2) -> 0 transposiciones
    EATEST_REQUIRE(blade_merge_sign(0b011, 0b100) == +1);
    // (e2)(e0 e1) -> e2 cruza e0 y e1 -> 2 transposiciones -> +1
    EATEST_REQUIRE(blade_merge_sign(0b100, 0b011) == +1);
}

// -----------------------------------------------------------------------------
// Cl(2,0) — plano euclídeo.
// -----------------------------------------------------------------------------

EATEST_CASE(g2_generadores_al_cuadrado_son_uno) {
    auto e0 = G2::e(0);
    auto e1 = G2::e(1);
    auto e0e0 = e0 * e0;
    auto e1e1 = e1 * e1;
    EATEST_REQUIRE(e0e0.approx_equal(G2(1.0), kTol));
    EATEST_REQUIRE(e1e1.approx_equal(G2(1.0), kTol));
}

EATEST_CASE(g2_anticonmutacion) {
    auto e0 = G2::e(0);
    auto e1 = G2::e(1);
    auto suma = (e0 * e1) + (e1 * e0);
    auto cero = G2(0.0);
    EATEST_REQUIRE(suma.approx_equal(cero, kTol));
}

EATEST_CASE(g2_pseudoescalar_al_cuadrado) {
    // En Cl(2,0): I = e0*e1, I^2 = -1.
    auto I = G2::pseudoscalar();
    auto I2 = I * I;
    EATEST_REQUIRE(I2.approx_equal(G2(-1.0), kTol));
}

// -----------------------------------------------------------------------------
// Cl(3,0) — espacio 3D euclídeo.
// -----------------------------------------------------------------------------

EATEST_CASE(g3_generadores_al_cuadrado_son_uno) {
    for (std::size_t i = 0; i < 3; ++i) {
        auto ei = G3::e(i);
        auto sq = ei * ei;
        EATEST_REQUIRE(sq.approx_equal(G3(1.0), kTol));
    }
}

EATEST_CASE(g3_bivector_al_cuadrado_es_menos_uno) {
    auto e0 = G3::e(0);
    auto e1 = G3::e(1);
    auto e2 = G3::e(2);
    for (auto B : {e0 * e1, e1 * e2, e0 * e2}) {
        auto BB = B * B;
        EATEST_REQUIRE(BB.approx_equal(G3(-1.0), kTol));
    }
}

EATEST_CASE(g3_pseudoescalar_conmuta_y_cuadrado_menos_uno) {
    // En Cl(3,0): I = e0*e1*e2, I^2 = -1, y conmuta con todos los vectores.
    auto I = G3::pseudoscalar();
    auto I2 = I * I;
    EATEST_REQUIRE(I2.approx_equal(G3(-1.0), kTol));
    for (std::size_t i = 0; i < 3; ++i) {
        auto ei = G3::e(i);
        auto a = I * ei;
        auto b = ei * I;
        EATEST_REQUIRE(a.approx_equal(b, kTol));
    }
}

EATEST_CASE(g3_producto_geometrico_es_inner_mas_outer) {
    // Para vectores u,v en Cl(3,0):
    //   u*v = u.v + u^v
    // donde u.v es escalar y u^v es bivector.
    G3 u = G3::e(0) * 1.0 + G3::e(1) * 2.0 + G3::e(2) * 3.0;
    G3 v = G3::e(0) * 4.0 + G3::e(1) * 5.0 + G3::e(2) * 6.0;
    auto prod = u * v;
    // parte escalar: u.v = 1*4 + 2*5 + 3*6 = 32
    EATEST_REQUIRE_NEAR(prod.scalar(), 32.0, kTol);
    // parte bivectorial: e0e1: 1*5 - 2*4 = -3; e0e2: 1*6 - 3*4 = -6; e1e2: 2*6 - 3*5 = -3
    EATEST_REQUIRE_NEAR(prod.coeff(0b011), -3.0, kTol);  // e0 e1
    EATEST_REQUIRE_NEAR(prod.coeff(0b101), -6.0, kTol);  // e0 e2
    EATEST_REQUIRE_NEAR(prod.coeff(0b110), -3.0, kTol);  // e1 e2
    // parte trivectorial: 0
    EATEST_REQUIRE_NEAR(prod.coeff(0b111), 0.0, kTol);
}

EATEST_CASE(g3_reverso_de_bivector_cambia_signo) {
    auto B = G3::e(0) * G3::e(1);
    auto rB = B.reverse();
    auto suma = B + rB;
    EATEST_REQUIRE(suma.approx_equal(G3(0.0), kTol));
}

EATEST_CASE(g3_involucion_grado) {
    // grade_involution: cambia signo de blades de grado impar (vectores y trivectores).
    auto v = G3::e(0) + G3::e(1) * 2.0;       // grado 1
    auto B = G3::e(0) * G3::e(1);             // grado 2
    auto T = G3::pseudoscalar();              // grado 3
    auto vh = v.grade_involution();
    auto Bh = B.grade_involution();
    auto Th = T.grade_involution();
    EATEST_REQUIRE(vh.approx_equal(-1.0 * v, kTol));
    EATEST_REQUIRE(Bh.approx_equal(B, kTol));
    EATEST_REQUIRE(Th.approx_equal(-1.0 * T, kTol));
}

EATEST_CASE(g3_norma_escalar_de_vector) {
    // Para v vector euclídeo en Cl(3,0): <v ~v>_0 = |v|^2.
    G3 v = G3::e(0) * 3.0 + G3::e(1) * 4.0 + G3::e(2) * 12.0;
    const double n2 = v.scalar_norm_squared();
    EATEST_REQUIRE_NEAR(n2, 9.0 + 16.0 + 144.0, 1e-10);
    EATEST_REQUIRE_NEAR(std::sqrt(n2), 13.0, 1e-10);
}

EATEST_CASE(g3_rotor_rota_vector_90_grados_en_plano_xy) {
    // Rotor R = exp(-theta/2 * e0e1) = cos(theta/2) - sin(theta/2) e0e1.
    // Aplicado a v:  v' = R v ~R.
    // Para theta = pi/2 y v = e0, esperamos v' = e1.
    const double theta = std::acos(-1.0) / 2.0;  // pi/2
    auto e01 = G3::e(0) * G3::e(1);
    G3 R = G3(std::cos(theta / 2.0)) + (-std::sin(theta / 2.0)) * e01;
    G3 Rrev = R.reverse();
    G3 v = G3::e(0);
    G3 vp = R * v * Rrev;
    G3 expected = G3::e(1);
    EATEST_REQUIRE(vp.approx_equal(expected, 1e-10));
}

EATEST_CASE(g3_doble_rotacion_180_grados_devuelve_menos_v) {
    // R(pi) v ~R(pi) = -v en el plano de rotación (caso conocido).
    const double pi = std::acos(-1.0);
    auto e01 = G3::e(0) * G3::e(1);
    G3 R = G3(std::cos(pi / 2.0)) + (-std::sin(pi / 2.0)) * e01;
    G3 v = G3::e(0);
    G3 vp = R * v * R.reverse();
    EATEST_REQUIRE(vp.approx_equal(-1.0 * v, 1e-10));
}

EATEST_CASE(g3_no_conmutatividad_explicita) {
    // e0*e1 != e1*e0 (anticonmutan), ergo el producto geométrico no es conmutativo.
    auto a = G3::e(0) * G3::e(1);
    auto b = G3::e(1) * G3::e(0);
    EATEST_REQUIRE(!a.approx_equal(b, kTol));
    auto suma = a + b;
    EATEST_REQUIRE(suma.approx_equal(G3(0.0), kTol));
}

EATEST_CASE(g3_asociatividad_producto_geometrico) {
    // (a b) c = a (b c) para multivectores arbitrarios.
    auto a = G3::e(0) * 1.5 + G3::e(1) * (-2.0) + G3(0.7);
    auto b = G3::e(2) * 0.3 + (G3::e(0) * G3::e(1)) * 1.1;
    auto c = G3(2.0) + G3::e(1) * 0.5;
    auto izq = (a * b) * c;
    auto der = a * (b * c);
    EATEST_REQUIRE(izq.approx_equal(der, 1e-10));
}

// -----------------------------------------------------------------------------
// Cl(1,3) — Spacetime Algebra (Hestenes). Métrica (+,-,-,-).
// -----------------------------------------------------------------------------

EATEST_CASE(sta_signatura_correcta) {
    // e0^2 = +1 (tiempo), e1^2 = e2^2 = e3^2 = -1 (espacio).
    auto e0 = STA::e(0);
    auto e1 = STA::e(1);
    auto e2 = STA::e(2);
    auto e3 = STA::e(3);
    EATEST_REQUIRE((e0 * e0).approx_equal(STA(+1.0), kTol));
    EATEST_REQUIRE((e1 * e1).approx_equal(STA(-1.0), kTol));
    EATEST_REQUIRE((e2 * e2).approx_equal(STA(-1.0), kTol));
    EATEST_REQUIRE((e3 * e3).approx_equal(STA(-1.0), kTol));
}

EATEST_CASE(sta_intervalo_lorentziano) {
    // Para v = t e0 + x e1 + y e2 + z e3, <v ~v>_0 = t^2 - x^2 - y^2 - z^2.
    const double t = 5.0, x = 3.0, y = 0.0, z = 0.0;
    STA v = STA::e(0) * t + STA::e(1) * x + STA::e(2) * y + STA::e(3) * z;
    const double s2 = v.scalar_norm_squared();
    EATEST_REQUIRE_NEAR(s2, t * t - x * x - y * y - z * z, 1e-10);
}

EATEST_CASE(sta_pseudoescalar_al_cuadrado_es_menos_uno) {
    // En Cl(1,3): I = e0 e1 e2 e3, I^2 = -1.
    auto I = STA::pseudoscalar();
    auto I2 = I * I;
    EATEST_REQUIRE(I2.approx_equal(STA(-1.0), kTol));
}

// -----------------------------------------------------------------------------
// Identidades algebraicas globales.
// -----------------------------------------------------------------------------

EATEST_CASE(g3_reverso_es_involucion) {
    // ~~A = A para cualquier multivector.
    auto A = G3(1.5) + G3::e(0) * 0.7 + (G3::e(0) * G3::e(2)) * (-1.3) +
             G3::pseudoscalar() * 2.0;
    auto AA = A.reverse().reverse();
    EATEST_REQUIRE(A.approx_equal(AA, kTol));
}

EATEST_CASE(g3_distributividad) {
    // a (b + c) = a b + a c.
    auto a = G3::e(0) * 1.1 + G3::e(1) * (-0.4);
    auto b = G3::e(2) * 0.7;
    auto c = G3(2.5) + G3::e(0) * G3::e(1);
    auto izq = a * (b + c);
    auto der = (a * b) + (a * c);
    EATEST_REQUIRE(izq.approx_equal(der, 1e-12));
}
