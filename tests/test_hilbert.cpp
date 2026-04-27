// Tests del Ladrillo 1 — Espacio de Hilbert H_D.
//
// Verificamos las propiedades algebraicas y geométricas básicas:
// hermicidad, Cauchy-Schwarz, ortonormalidad, idempotencia del proyector,
// fidelidad invariante por escala, etc.

#include "test_framework.hpp"
#include "easyatom/hilbert/state.hpp"

#include <cmath>
#include <complex>
#include <vector>

using easyatom::hilbert::Complex;
using easyatom::hilbert::State;
using easyatom::hilbert::inner;
using easyatom::hilbert::project;
using easyatom::hilbert::fidelity;
using easyatom::hilbert::superpose;

constexpr double kHTol = 1e-12;

// -----------------------------------------------------------------------------
// Construcción y acceso.
// -----------------------------------------------------------------------------

EATEST_CASE(hilbert_basis_es_canonico) {
    auto e0 = State::basis(3, 0);
    auto e1 = State::basis(3, 1);
    EATEST_REQUIRE(e0.dim() == 3);
    EATEST_REQUIRE_NEAR(e0[0].real(), 1.0, kHTol);
    EATEST_REQUIRE_NEAR(e0[1].real(), 0.0, kHTol);
    EATEST_REQUIRE_NEAR(e1[1].real(), 1.0, kHTol);
}

EATEST_CASE(hilbert_dim_cero_lanza) {
    bool threw = false;
    try { State s(0); } catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

// -----------------------------------------------------------------------------
// Producto interno.
// -----------------------------------------------------------------------------

EATEST_CASE(hilbert_inner_de_base_es_kronecker) {
    auto e0 = State::basis(4, 0);
    auto e1 = State::basis(4, 1);
    auto e3 = State::basis(4, 3);
    EATEST_REQUIRE_NEAR(inner(e0, e0).real(), 1.0, kHTol);
    EATEST_REQUIRE_NEAR(inner(e0, e1).real(), 0.0, kHTol);
    EATEST_REQUIRE_NEAR(inner(e1, e3).real(), 0.0, kHTol);
}

EATEST_CASE(hilbert_inner_es_hermitico) {
    // ⟨φ|ψ⟩ = conj(⟨ψ|φ⟩) para cualquier par.
    State a(std::vector<Complex>{Complex{1.0, 2.0}, Complex{-0.5, 0.3}, Complex{0.0, 1.7}});
    State b(std::vector<Complex>{Complex{0.4, -1.1}, Complex{2.0, 0.0}, Complex{-1.0, 0.5}});
    Complex ab = inner(a, b);
    Complex ba = inner(b, a);
    EATEST_REQUIRE_NEAR(ab.real(), std::conj(ba).real(), 1e-12);
    EATEST_REQUIRE_NEAR(ab.imag(), std::conj(ba).imag(), 1e-12);
}

EATEST_CASE(hilbert_inner_es_lineal_en_el_segundo_argumento) {
    // ⟨φ| (α|ψ1⟩ + β|ψ2⟩) = α⟨φ|ψ1⟩ + β⟨φ|ψ2⟩
    State phi(std::vector<Complex>{Complex{1.0, 0.0}, Complex{0.0, 1.0}});
    State p1 (std::vector<Complex>{Complex{0.5, 0.0}, Complex{1.0, 0.0}});
    State p2 (std::vector<Complex>{Complex{0.0, 1.0}, Complex{2.0, 0.0}});
    const Complex alpha{1.5, -0.7};
    const Complex beta {0.3,  2.0};
    State combo = alpha * p1 + beta * p2;
    Complex izq = inner(phi, combo);
    Complex der = alpha * inner(phi, p1) + beta * inner(phi, p2);
    EATEST_REQUIRE_NEAR(izq.real(), der.real(), 1e-12);
    EATEST_REQUIRE_NEAR(izq.imag(), der.imag(), 1e-12);
}

EATEST_CASE(hilbert_norma_es_real_no_negativa) {
    State a(std::vector<Complex>{Complex{3.0, 4.0}, Complex{0.0, 12.0}});
    // |3+4i|^2 + |12i|^2 = 25 + 144 = 169
    EATEST_REQUIRE_NEAR(a.norm_squared(), 169.0, 1e-12);
    EATEST_REQUIRE_NEAR(a.norm(), 13.0, 1e-12);
}

EATEST_CASE(hilbert_cauchy_schwarz) {
    // |⟨φ|ψ⟩|² ≤ ⟨φ|φ⟩ ⟨ψ|ψ⟩ con igualdad si y solo si son colineales.
    State a(std::vector<Complex>{Complex{1.0, 2.0}, Complex{-1.0, 0.5}, Complex{0.7, -0.3}});
    State b(std::vector<Complex>{Complex{0.0, 1.0}, Complex{2.0, -1.0}, Complex{1.0,  0.0}});
    const double lhs = std::norm(inner(a, b));
    const double rhs = a.norm_squared() * b.norm_squared();
    EATEST_REQUIRE(lhs <= rhs + 1e-12);

    // Caso colineal estricto: b = c * a debe dar igualdad.
    State c = Complex{2.5, -1.7} * a;
    const double lhs2 = std::norm(inner(a, c));
    const double rhs2 = a.norm_squared() * c.norm_squared();
    EATEST_REQUIRE_NEAR(lhs2, rhs2, 1e-10);
}

// -----------------------------------------------------------------------------
// Normalización.
// -----------------------------------------------------------------------------

EATEST_CASE(hilbert_normalized_tiene_norma_uno) {
    State a(std::vector<Complex>{Complex{3.0, 0.0}, Complex{0.0, 4.0}});
    auto an = a.normalized();
    EATEST_REQUIRE_NEAR(an.norm(), 1.0, 1e-12);
}

EATEST_CASE(hilbert_normalized_de_estado_nulo_lanza) {
    State zero(3);  // todo cero
    bool threw = false;
    try { (void)zero.normalized(); } catch (const std::domain_error&) { threw = true; }
    EATEST_REQUIRE(threw);
}

// -----------------------------------------------------------------------------
// Proyección.
// -----------------------------------------------------------------------------

EATEST_CASE(hilbert_proyector_es_idempotente) {
    // P_ψ (P_ψ φ) = P_ψ φ.
    State psi (std::vector<Complex>{Complex{1.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}});
    State phi (std::vector<Complex>{Complex{2.0, 1.0}, Complex{-1.0, 0.5}, Complex{3.0, 0.0}});
    State p1 = project(phi, psi);
    State p2 = project(p1, psi);
    EATEST_REQUIRE(p1.approx_equal(p2, 1e-12));
}

EATEST_CASE(hilbert_proyector_sobre_paralelo_es_identidad) {
    State psi(std::vector<Complex>{Complex{1.0, 0.0}, Complex{0.0, 1.0}});
    State phi = Complex{2.5, -0.7} * psi;  // exactamente paralelo a ψ
    State p = project(phi, psi);
    EATEST_REQUIRE(p.approx_equal(phi, 1e-12));
}

EATEST_CASE(hilbert_proyector_sobre_ortogonal_es_cero) {
    auto e0 = State::basis(3, 0);
    auto e1 = State::basis(3, 1);
    auto p = project(e1, e0);
    State zero(3);
    EATEST_REQUIRE(p.approx_equal(zero, 1e-12));
}

// -----------------------------------------------------------------------------
// Fidelidad.
// -----------------------------------------------------------------------------

EATEST_CASE(hilbert_fidelity_invariante_por_escala_global) {
    State a(std::vector<Complex>{Complex{1.0, 0.5}, Complex{0.7, -0.2}, Complex{0.0, 1.3}});
    State b(std::vector<Complex>{Complex{0.0, 1.0}, Complex{1.0,  0.0}, Complex{-0.5, 0.4}});
    State a2 = Complex{3.0, 1.0} * a;
    State b2 = Complex{0.5, -2.0} * b;
    const double f1 = fidelity(a, b);
    const double f2 = fidelity(a2, b2);
    EATEST_REQUIRE_NEAR(f1, f2, 1e-12);
}

EATEST_CASE(hilbert_fidelity_consigo_mismo_es_uno) {
    State a(std::vector<Complex>{Complex{1.5, -0.5}, Complex{2.0, 1.0}});
    EATEST_REQUIRE_NEAR(fidelity(a, a), 1.0, 1e-12);
}

EATEST_CASE(hilbert_fidelity_de_ortogonales_es_cero) {
    auto e0 = State::basis(2, 0);
    auto e1 = State::basis(2, 1);
    EATEST_REQUIRE_NEAR(fidelity(e0, e1), 0.0, 1e-12);
}

EATEST_CASE(hilbert_fidelity_en_rango_unidad) {
    // 0 ≤ F(φ,ψ) ≤ 1 siempre.
    State a(std::vector<Complex>{Complex{1.0, 2.0}, Complex{-1.0, 0.0}, Complex{0.5, -0.7}});
    State b(std::vector<Complex>{Complex{0.3, 0.4}, Complex{ 1.1, 0.0}, Complex{0.0,  1.0}});
    const double f = fidelity(a, b);
    EATEST_REQUIRE(f >= 0.0);
    EATEST_REQUIRE(f <= 1.0 + 1e-12);
}

// -----------------------------------------------------------------------------
// Superposición.
// -----------------------------------------------------------------------------

EATEST_CASE(hilbert_superpose_basico) {
    auto e0 = State::basis(3, 0);
    auto e1 = State::basis(3, 1);
    auto e2 = State::basis(3, 2);
    State r = superpose(
        {Complex{1.0, 0.0}, Complex{0.0, 1.0}, Complex{2.0, 0.0}},
        {e0, e1, e2});
    EATEST_REQUIRE_NEAR(r[0].real(), 1.0, kHTol);
    EATEST_REQUIRE_NEAR(r[1].imag(), 1.0, kHTol);
    EATEST_REQUIRE_NEAR(r[2].real(), 2.0, kHTol);
}

EATEST_CASE(hilbert_dimensiones_inconsistentes_lanzan) {
    State a(2);
    State b(3);
    bool threw = false;
    try { (void)inner(a, b); } catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}
