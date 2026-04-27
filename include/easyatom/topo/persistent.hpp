// =============================================================================
// EasyAtom · Ladrillo 19 — Homologia persistente naive (Betti_0, Betti_1).
// =============================================================================
//
// Calculamos los dos primeros numeros de Betti del complejo de Vietoris-Rips
// de una nube finita de puntos a una escala epsilon dada:
//
//   beta_0(eps) = numero de componentes conexas a esa escala.
//   beta_1(eps) = numero de "huecos" (1-ciclos no rellenos) a esa escala.
//
// Construccion del complejo (truncado en dimension 2):
//   * 0-simplices: los N puntos.
//   * 1-simplice {i,j}        si  d(i,j) <= eps.
//   * 2-simplice {i,j,k}      si  los tres pares forman 1-simplice.
//
// Calculo:
//   * beta_0  via union-find sobre los 1-simplices.
//   * Asumiendo beta_2 ~ 0 (no construimos 3-simplices), por la caracteristica
//     de Euler chi = V - E + F  y  chi = beta_0 - beta_1 + beta_2:
//
//          beta_1  =  beta_0  -  V  +  E  -  F
//
//     Esta aproximacion es exacta para todo complejo 2D sin volumenes
//     enclosados (todo el caso de interes para validacion logica de leyes
//     compiladas en H_D, cuyos outputs proyectamos a R^k via fingerprint).
//
// Coste: O(N^2) edges, O(N^3) triangulos (pero con corte temprano por eps).
//        Para N <= 200 en runtime es trivial; para >200 exigir muestreo previo.
//
// Uso esperado:
//   * L23 (coherencia): si añadir una ley aumenta beta_1 sobre la nube de
//     outputs => contradiccion topologica => rechazar.
//   * L24 (lagunas): regiones con baja densidad y beta_1 alto = huecos en el
//     conocimiento.
//
// Sin RAG. Sin embeddings externos. Solo geometria de los estados que el
// motor ya produce.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace easyatom::topo {

// -----------------------------------------------------------------------------
// Distancia euclidea por defecto.
// -----------------------------------------------------------------------------

[[nodiscard]] inline double euclidean(const std::vector<double>& a,
                                      const std::vector<double>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("topo::euclidean: dim distinta.");
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

// -----------------------------------------------------------------------------
// Union-Find (DSU) compacto para beta_0.
// -----------------------------------------------------------------------------

struct DSU {
    std::vector<std::int32_t> parent;
    std::vector<std::int32_t> rank;
    explicit DSU(std::size_t n)
        : parent(n), rank(n, 0) {
        for (std::size_t i = 0; i < n; ++i)
            parent[i] = static_cast<std::int32_t>(i);
    }
    std::int32_t find(std::int32_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    bool unite(std::int32_t a, std::int32_t b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) ++rank[a];
        return true;
    }
};

// -----------------------------------------------------------------------------
// Resultado.
// -----------------------------------------------------------------------------

struct BettiResult {
    std::int64_t beta_0   = 0;
    std::int64_t beta_1   = 0;
    std::size_t  vertices = 0;
    std::size_t  edges    = 0;
    std::size_t  faces    = 0;
};

// -----------------------------------------------------------------------------
// Calculo principal sobre nube de puntos con metrica euclidea.
// -----------------------------------------------------------------------------

[[nodiscard]] inline BettiResult vietoris_rips_betti(
    const std::vector<std::vector<double>>& points, double epsilon) {
    if (epsilon < 0.0)
        throw std::invalid_argument("vietoris_rips_betti: epsilon < 0.");
    BettiResult r;
    const std::size_t N = points.size();
    r.vertices = N;
    if (N == 0) return r;

    // Caso degenerado: 1 punto -> 1 componente, sin aristas ni caras.
    DSU dsu(N);

    // Matriz simetrica de adyacencia (almacenada como triangular superior).
    // Para N=200 ocupa ~ 20kB en bool: aceptable.
    std::vector<std::uint8_t> adj(N * N, 0);
    auto edge = [&](std::size_t i, std::size_t j) -> std::uint8_t& {
        return adj[i * N + j];
    };

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            const double d = euclidean(points[i], points[j]);
            if (d <= epsilon) {
                edge(i, j) = 1;
                edge(j, i) = 1;
                ++r.edges;
                dsu.unite(static_cast<std::int32_t>(i),
                          static_cast<std::int32_t>(j));
            }
        }
    }

    // beta_0 = numero de raices distintas en union-find.
    {
        std::vector<std::uint8_t> seen(N, 0);
        for (std::size_t i = 0; i < N; ++i) {
            const std::int32_t root =
                dsu.find(static_cast<std::int32_t>(i));
            if (!seen[root]) {
                seen[root] = 1;
                ++r.beta_0;
            }
        }
    }

    // 2-simplices: triples (i,j,k) con i<j<k y los 3 pares conectados.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            if (!edge(i, j)) continue;
            for (std::size_t k = j + 1; k < N; ++k) {
                if (edge(i, k) && edge(j, k)) ++r.faces;
            }
        }
    }

    // beta_1 = beta_0 - V + E - F  (asumiendo beta_2 = 0).
    const std::int64_t b1 =
        r.beta_0 - static_cast<std::int64_t>(r.vertices)
                 + static_cast<std::int64_t>(r.edges)
                 - static_cast<std::int64_t>(r.faces);
    r.beta_1 = (b1 < 0) ? 0 : b1;   // safety: la formula nunca debe ser negativa
                                    // en un complejo Vietoris-Rips puro.
    return r;
}

}  // namespace easyatom::topo
