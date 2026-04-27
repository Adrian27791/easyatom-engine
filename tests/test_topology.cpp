// Tests del Ladrillo 4 — homología persistente.

#include "test_framework.hpp"
#include "easyatom/topology/persistence.hpp"

#include <cmath>
#include <vector>

using easyatom::topology::DistanceMatrix;
using easyatom::topology::Diagram;
using easyatom::topology::PersistencePair;
using easyatom::topology::persistence_h0;
using easyatom::topology::betti_at_epsilon;
using easyatom::topology::bottleneck_distance;
using easyatom::topology::kInf;

constexpr double kTTol = 1e-9;

// -----------------------------------------------------------------------------
// DistanceMatrix.
// -----------------------------------------------------------------------------

EATEST_CASE(topo_distmatrix_from_points_euclidean) {
    std::vector<std::vector<double>> pts = {{0,0}, {3,0}, {0,4}};
    auto D = DistanceMatrix::from_points_euclidean(pts);
    EATEST_REQUIRE_NEAR(D(0,1), 3.0, kTTol);
    EATEST_REQUIRE_NEAR(D(0,2), 4.0, kTTol);
    EATEST_REQUIRE_NEAR(D(1,2), 5.0, kTTol);
    EATEST_REQUIRE_NEAR(D(0,0), 0.0, kTTol);
}

// -----------------------------------------------------------------------------
// H_0: persistencia de componentes.
// -----------------------------------------------------------------------------

EATEST_CASE(topo_h0_un_solo_punto) {
    DistanceMatrix D(1);
    auto diag = persistence_h0(D);
    EATEST_REQUIRE(diag.size() == 1);
    EATEST_REQUIRE(diag[0].death == kInf);
}

EATEST_CASE(topo_h0_tres_puntos_en_linea) {
    // Puntos en x = 0, 1, 3. Aristas: (0,1)=1, (1,2)=2, (0,2)=3.
    // Las dos primeras aristas matan dos clases → pares (0,1) y (0,2).
    // Queda 1 clase esencial.
    std::vector<std::vector<double>> pts = {{0}, {1}, {3}};
    auto D = DistanceMatrix::from_points_euclidean(pts);
    auto diag = persistence_h0(D);
    EATEST_REQUIRE(diag.size() == 3);

    // Contamos cuántos pares con muerte 1, 2 e ∞.
    int n_inf = 0, n_1 = 0, n_2 = 0;
    for (const auto& p : diag) {
        if (p.death == kInf) ++n_inf;
        else if (std::abs(p.death - 1.0) < kTTol) ++n_1;
        else if (std::abs(p.death - 2.0) < kTTol) ++n_2;
    }
    EATEST_REQUIRE(n_inf == 1);
    EATEST_REQUIRE(n_1 == 1);
    EATEST_REQUIRE(n_2 == 1);
}

EATEST_CASE(topo_h0_dos_clusters_separados) {
    // Cluster A en (0,0),(1,0). Cluster B en (10,0),(11,0).
    // Aristas internas matan 1 clase cada una en ε=1. La fusión inter-cluster
    // a ε=9 mata otra clase. Queda 1 esencial.
    std::vector<std::vector<double>> pts = {{0,0},{1,0},{10,0},{11,0}};
    auto D = DistanceMatrix::from_points_euclidean(pts);
    auto diag = persistence_h0(D);
    EATEST_REQUIRE(diag.size() == 4);

    int n_inf = 0, n_1 = 0, n_9 = 0;
    for (const auto& p : diag) {
        if (p.death == kInf) ++n_inf;
        else if (std::abs(p.death - 1.0) < kTTol) ++n_1;
        else if (std::abs(p.death - 9.0) < kTTol) ++n_9;
    }
    EATEST_REQUIRE(n_inf == 1);
    EATEST_REQUIRE(n_1 == 2);  // dos pares con muerte 1
    EATEST_REQUIRE(n_9 == 1);
}

// -----------------------------------------------------------------------------
// Números de Betti.
// -----------------------------------------------------------------------------

EATEST_CASE(topo_betti_dos_puntos_aislados) {
    std::vector<std::vector<double>> pts = {{0,0},{10,0}};
    auto D = DistanceMatrix::from_points_euclidean(pts);
    auto b = betti_at_epsilon(D, 1.0);
    EATEST_REQUIRE(b.b0 == 2);
    EATEST_REQUIRE(b.b1 == 0);
    auto b2 = betti_at_epsilon(D, 11.0);
    EATEST_REQUIRE(b2.b0 == 1);
    EATEST_REQUIRE(b2.b1 == 0);
}

EATEST_CASE(topo_betti_cuadrado_es_un_ciclo) {
    // Cuadrado: (0,0),(1,0),(1,1),(0,1).
    // Aristas: 4 lados de longitud 1. Diagonales de √2.
    // En ε=1.0 (excluyendo diagonales): V=4, E=4, T=0 → b_0=1, b_1=1.
    std::vector<std::vector<double>> pts = {{0,0},{1,0},{1,1},{0,1}};
    auto D = DistanceMatrix::from_points_euclidean(pts);
    auto b = betti_at_epsilon(D, 1.0);
    EATEST_REQUIRE(b.b0 == 1);
    EATEST_REQUIRE(b.b1 == 1);
    // En ε=√2: incluye diagonales, complejo se rellena con triángulos.
    // V=4, E=6, T=4 → b_1 = 6 - 4 + 1 - 4 = -1 → 0. Correcto: ya no hay agujero.
    auto b2 = betti_at_epsilon(D, 1.5);
    EATEST_REQUIRE(b2.b0 == 1);
    EATEST_REQUIRE(b2.b1 == 0);
}

EATEST_CASE(topo_betti_triangulo_relleno_no_tiene_ciclo) {
    // Triángulo equilátero de lado 1.
    std::vector<std::vector<double>> pts = {
        {0.0, 0.0}, {1.0, 0.0}, {0.5, std::sqrt(3.0)/2.0}};
    auto D = DistanceMatrix::from_points_euclidean(pts);
    auto b = betti_at_epsilon(D, 1.01);
    EATEST_REQUIRE(b.b0 == 1);
    // V=3, E=3, T=1 → b_1 = 3 - 3 + 1 - 1 = 0
    EATEST_REQUIRE(b.b1 == 0);
}

EATEST_CASE(topo_betti_anillo_de_5_puntos) {
    // Pentágono regular de circunradio 1.
    std::vector<std::vector<double>> pts;
    constexpr double kPi = 3.14159265358979323846;
    for (int i = 0; i < 5; ++i) {
        double a = 2.0 * kPi * i / 5.0;
        pts.push_back({std::cos(a), std::sin(a)});
    }
    auto D = DistanceMatrix::from_points_euclidean(pts);
    // Lado del pentágono ≈ 1.1756. Diagonal ≈ 1.9021.
    // En ε ligeramente mayor que el lado: V=5, E=5 (los 5 lados), T=0.
    // → b_0 = 1, b_1 = 1.
    auto b = betti_at_epsilon(D, 1.20);
    EATEST_REQUIRE(b.b0 == 1);
    EATEST_REQUIRE(b.b1 == 1);
}

// -----------------------------------------------------------------------------
// Distancia bottleneck.
// -----------------------------------------------------------------------------

EATEST_CASE(topo_bottleneck_diagramas_iguales_es_cero) {
    Diagram A = {{0.0, 1.0}, {0.0, 2.0}};
    EATEST_REQUIRE_NEAR(bottleneck_distance(A, A), 0.0, kTTol);
}

EATEST_CASE(topo_bottleneck_traslacion_pequena) {
    // Dos diagramas con el mismo punto desplazado por (0.1, 0.1).
    Diagram A = {{0.0, 1.0}};
    Diagram B = {{0.1, 1.1}};
    // L∞((0,1),(0.1,1.1)) = 0.1. Es mejor emparejar directo que con diagonal.
    EATEST_REQUIRE_NEAR(bottleneck_distance(A, B), 0.1, kTTol);
}

EATEST_CASE(topo_bottleneck_punto_extra_se_empareja_con_diagonal) {
    // Diagrama A vacío, B con un punto cerca de la diagonal.
    Diagram A = {};
    Diagram B = {{0.0, 0.2}};
    // Distancia a la diagonal de (0, 0.2) = 0.1.
    EATEST_REQUIRE_NEAR(bottleneck_distance(A, B), 0.1, kTTol);
}

EATEST_CASE(topo_bottleneck_simetrica) {
    Diagram A = {{0.0, 1.0}, {0.5, 2.0}};
    Diagram B = {{0.1, 1.2}, {0.6, 1.8}};
    EATEST_REQUIRE_NEAR(bottleneck_distance(A, B),
                       bottleneck_distance(B, A), kTTol);
}
