// Tests del Ladrillo 3 — geometría de la información sobre el símplex.
//
// Verificamos las propiedades exigidas por la teoría:
//   * Distribution valida: rechaza negativos y sumas != 1.
//   * BC, Hellinger, Fisher-Rao son simétricas; cero sii p=q; en rango.
//   * Fisher-Rao satisface desigualdad triangular (caso concreto).
//   * KL ≥ 0; KL(p,p)=0; NO simétrica.
//   * α-divergencia: D_{+1}=KL(p,q), D_{-1}=KL(q,p), D_0 = 4 H².
//   * Dualidad de Amari: D_α(p,q) = D_{-α}(q,p).
//   * Métrica de Fisher: g_ii = 1/p_i; en uniforme, g_ii = n.
//   * Geodésica: γ(0)=p, γ(1)=q, γ(0.5) entre p y q.

#include "test_framework.hpp"
#include "easyatom/infogeo/fisher.hpp"

#include <cmath>
#include <vector>

using easyatom::infogeo::Distribution;
using easyatom::infogeo::bhattacharyya_coefficient;
using easyatom::infogeo::hellinger_distance;
using easyatom::infogeo::fisher_rao_distance;
using easyatom::infogeo::kl_divergence;
using easyatom::infogeo::alpha_divergence;
using easyatom::infogeo::fisher_metric_diagonal;
using easyatom::infogeo::fisher_norm_squared;
using easyatom::infogeo::fisher_rao_geodesic;

constexpr double kITol = 1e-12;

// -----------------------------------------------------------------------------
// Distribution — validación.
// -----------------------------------------------------------------------------

EATEST_CASE(infogeo_distribution_valida_suma_uno) {
    Distribution p({0.2, 0.3, 0.5});
    EATEST_REQUIRE(p.dim() == 3);
    EATEST_REQUIRE_NEAR(p[0], 0.2, kITol);
}

EATEST_CASE(infogeo_distribution_rechaza_negativos) {
    bool threw = false;
    try { Distribution p({0.5, -0.1, 0.6}); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(infogeo_distribution_rechaza_suma_no_uno) {
    bool threw = false;
    try { Distribution p({0.2, 0.2, 0.2}); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(infogeo_distribution_uniforme) {
    auto u = Distribution::uniform(5);
    for (std::size_t i = 0; i < 5; ++i) {
        EATEST_REQUIRE_NEAR(u[i], 0.2, kITol);
    }
}

EATEST_CASE(infogeo_distribution_from_scores_normaliza) {
    auto p = Distribution::from_scores({1.0, 2.0, 1.0});
    EATEST_REQUIRE_NEAR(p[0], 0.25, kITol);
    EATEST_REQUIRE_NEAR(p[1], 0.50, kITol);
    EATEST_REQUIRE_NEAR(p[2], 0.25, kITol);
}

// -----------------------------------------------------------------------------
// Bhattacharyya, Hellinger, Fisher-Rao.
// -----------------------------------------------------------------------------

EATEST_CASE(infogeo_bc_de_p_consigo_es_uno) {
    Distribution p({0.1, 0.2, 0.3, 0.4});
    EATEST_REQUIRE_NEAR(bhattacharyya_coefficient(p, p), 1.0, 1e-12);
}

EATEST_CASE(infogeo_bc_simetrico) {
    Distribution p({0.1, 0.2, 0.7});
    Distribution q({0.5, 0.4, 0.1});
    const double a = bhattacharyya_coefficient(p, q);
    const double b = bhattacharyya_coefficient(q, p);
    EATEST_REQUIRE_NEAR(a, b, 1e-12);
}

EATEST_CASE(infogeo_hellinger_cero_sii_iguales) {
    Distribution p({0.4, 0.6});
    EATEST_REQUIRE_NEAR(hellinger_distance(p, p), 0.0, 1e-12);
    Distribution q({0.6, 0.4});
    EATEST_REQUIRE(hellinger_distance(p, q) > 0.0);
}

EATEST_CASE(infogeo_hellinger_en_rango_unidad) {
    Distribution p({1.0, 0.0, 0.0});
    Distribution q({0.0, 1.0, 0.0});
    // Distribuciones disjuntas: BC = 0 → H = 1.
    EATEST_REQUIRE_NEAR(hellinger_distance(p, q), 1.0, 1e-12);
}

EATEST_CASE(infogeo_fisher_rao_consigo_mismo_cero) {
    Distribution p({0.25, 0.25, 0.5});
    EATEST_REQUIRE_NEAR(fisher_rao_distance(p, p), 0.0, 1e-9);
}

EATEST_CASE(infogeo_fisher_rao_simetrica) {
    Distribution p({0.7, 0.2, 0.1});
    Distribution q({0.1, 0.4, 0.5});
    EATEST_REQUIRE_NEAR(fisher_rao_distance(p, q),
                       fisher_rao_distance(q, p), 1e-12);
}

EATEST_CASE(infogeo_fisher_rao_disjuntas_es_pi) {
    Distribution p({1.0, 0.0});
    Distribution q({0.0, 1.0});
    constexpr double kPi = 3.14159265358979323846;
    EATEST_REQUIRE_NEAR(fisher_rao_distance(p, q), kPi, 1e-9);
}

EATEST_CASE(infogeo_fisher_rao_triangular) {
    Distribution p({0.6, 0.3, 0.1});
    Distribution q({0.2, 0.5, 0.3});
    Distribution r({0.1, 0.2, 0.7});
    const double dpq = fisher_rao_distance(p, q);
    const double dqr = fisher_rao_distance(q, r);
    const double dpr = fisher_rao_distance(p, r);
    EATEST_REQUIRE(dpr <= dpq + dqr + 1e-12);
}

// -----------------------------------------------------------------------------
// Kullback-Leibler.
// -----------------------------------------------------------------------------

EATEST_CASE(infogeo_kl_consigo_mismo_es_cero) {
    Distribution p({0.3, 0.4, 0.3});
    EATEST_REQUIRE_NEAR(kl_divergence(p, p), 0.0, 1e-12);
}

EATEST_CASE(infogeo_kl_no_negativa) {
    Distribution p({0.7, 0.2, 0.1});
    Distribution q({0.1, 0.5, 0.4});
    EATEST_REQUIRE(kl_divergence(p, q) > 0.0);
}

EATEST_CASE(infogeo_kl_no_simetrica) {
    Distribution p({0.7, 0.2, 0.1});
    Distribution q({0.1, 0.5, 0.4});
    const double a = kl_divergence(p, q);
    const double b = kl_divergence(q, p);
    EATEST_REQUIRE(std::abs(a - b) > 1e-3);
}

EATEST_CASE(infogeo_kl_q_con_cero_lanza) {
    Distribution p({0.5, 0.5});
    Distribution q({1.0, 0.0});
    bool threw = false;
    try { (void)kl_divergence(p, q); }
    catch (const std::domain_error&) { threw = true; }
    EATEST_REQUIRE(threw);
}

// -----------------------------------------------------------------------------
// α-divergencia de Amari.
// -----------------------------------------------------------------------------

EATEST_CASE(infogeo_alpha_uno_recupera_kl) {
    Distribution p({0.6, 0.3, 0.1});
    Distribution q({0.2, 0.5, 0.3});
    EATEST_REQUIRE_NEAR(alpha_divergence(p, q, 1.0),
                       kl_divergence(p, q), 1e-12);
}

EATEST_CASE(infogeo_alpha_menos_uno_recupera_kl_inversa) {
    Distribution p({0.6, 0.3, 0.1});
    Distribution q({0.2, 0.5, 0.3});
    EATEST_REQUIRE_NEAR(alpha_divergence(p, q, -1.0),
                       kl_divergence(q, p), 1e-12);
}

EATEST_CASE(infogeo_alpha_cero_es_cuatro_hellinger_cuadrado) {
    // D_0(p,q) = 4 (1 - BC) = 4 H(p,q)^2
    Distribution p({0.1, 0.4, 0.5});
    Distribution q({0.5, 0.3, 0.2});
    const double H = hellinger_distance(p, q);
    EATEST_REQUIRE_NEAR(alpha_divergence(p, q, 0.0), 4.0 * H * H, 1e-12);
}

EATEST_CASE(infogeo_alpha_dualidad) {
    // D_α(p,q) = D_{-α}(q,p)
    Distribution p({0.6, 0.3, 0.1});
    Distribution q({0.2, 0.5, 0.3});
    for (double alpha : {-0.7, -0.3, 0.0, 0.3, 0.7}) {
        const double a = alpha_divergence(p, q,  alpha);
        const double b = alpha_divergence(q, p, -alpha);
        EATEST_REQUIRE_NEAR(a, b, 1e-12);
    }
}

EATEST_CASE(infogeo_alpha_consigo_mismo_es_cero) {
    Distribution p({0.4, 0.35, 0.25});
    for (double alpha : {-0.9, -0.5, 0.0, 0.5, 0.9}) {
        EATEST_REQUIRE_NEAR(alpha_divergence(p, p, alpha), 0.0, 1e-12);
    }
}

// -----------------------------------------------------------------------------
// Métrica de Fisher.
// -----------------------------------------------------------------------------

EATEST_CASE(infogeo_fisher_metric_diagonal_correcta) {
    Distribution p({0.25, 0.5, 0.25});
    auto g = fisher_metric_diagonal(p);
    EATEST_REQUIRE_NEAR(g[0], 4.0, kITol);
    EATEST_REQUIRE_NEAR(g[1], 2.0, kITol);
    EATEST_REQUIRE_NEAR(g[2], 4.0, kITol);
}

EATEST_CASE(infogeo_fisher_metric_uniforme_es_n) {
    auto u = Distribution::uniform(5);
    auto g = fisher_metric_diagonal(u);
    for (double gi : g) EATEST_REQUIRE_NEAR(gi, 5.0, kITol);
}

EATEST_CASE(infogeo_fisher_norm_de_vector_tangente) {
    // En p uniforme con n=4, ||v||²_p = 4 * sum v_i².
    auto u = Distribution::uniform(4);
    std::vector<double> v{0.1, -0.2, 0.05, 0.05};  // sum = 0 (tangente)
    const double esperado = 4.0 * (0.01 + 0.04 + 0.0025 + 0.0025);
    EATEST_REQUIRE_NEAR(fisher_norm_squared(u, v), esperado, 1e-12);
}

// -----------------------------------------------------------------------------
// Geodésica de Fisher-Rao.
// -----------------------------------------------------------------------------

EATEST_CASE(infogeo_geodesic_t_cero_es_p) {
    Distribution p({0.7, 0.2, 0.1});
    Distribution q({0.1, 0.5, 0.4});
    auto g = fisher_rao_geodesic(p, q, 0.0);
    EATEST_REQUIRE(p.approx_equal(g, 1e-9));
}

EATEST_CASE(infogeo_geodesic_t_uno_es_q) {
    Distribution p({0.7, 0.2, 0.1});
    Distribution q({0.1, 0.5, 0.4});
    auto g = fisher_rao_geodesic(p, q, 1.0);
    EATEST_REQUIRE(q.approx_equal(g, 1e-9));
}

EATEST_CASE(infogeo_geodesic_punto_medio_es_equidistante) {
    // d(p, γ(1/2)) ≈ d(γ(1/2), q) = (1/2) d(p, q)
    Distribution p({0.7, 0.2, 0.1});
    Distribution q({0.1, 0.5, 0.4});
    auto m = fisher_rao_geodesic(p, q, 0.5);
    const double dpq = fisher_rao_distance(p, q);
    const double dpm = fisher_rao_distance(p, m);
    const double dmq = fisher_rao_distance(m, q);
    EATEST_REQUIRE_NEAR(dpm, dpq * 0.5, 1e-9);
    EATEST_REQUIRE_NEAR(dmq, dpq * 0.5, 1e-9);
}

EATEST_CASE(infogeo_geodesic_es_distribucion_valida) {
    Distribution p({0.6, 0.3, 0.1});
    Distribution q({0.2, 0.5, 0.3});
    for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        auto g = fisher_rao_geodesic(p, q, t);
        double s = 0.0;
        for (std::size_t i = 0; i < g.dim(); ++i) {
            EATEST_REQUIRE(g[i] >= -1e-12);
            s += g[i];
        }
        EATEST_REQUIRE_NEAR(s, 1.0, 1e-9);
    }
}
