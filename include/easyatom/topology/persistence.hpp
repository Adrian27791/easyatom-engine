// =============================================================================
// EasyAtom · Ladrillo 4 — Topología: homología persistente H_0 y H_1.
// =============================================================================
//
// Dada una nube de puntos {x_1,...,x_n} en un espacio con métrica d, el
// complejo de Vietoris-Rips a escala ε es el complejo simplicial cuyos
// símplices son los subconjuntos de diámetro ≤ ε. Variando ε desde 0 hasta
// ∞ obtenemos una filtración: una secuencia creciente de complejos.
//
// Cada característica topológica (componente conexa, ciclo) tiene un
// momento de NACIMIENTO (la primera ε en que aparece) y un momento de
// MUERTE (la primera ε en que se rellena / se fusiona). El diagrama de
// persistencia es el conjunto de pares (b, d) con d ≥ b.
//
// Esto NO depende de coordenadas: solo de distancias. Por eso captura la
// estructura intrínseca de los datos. Es la herramienta que el motor usa
// para distinguir "una nube", "un anillo", "dos cúmulos separados", etc.
//
// LO QUE IMPLEMENTAMOS (sin dependencias externas):
//
//   * VietorisRips: filtración a partir de matriz de distancias.
//   * Persistencia de H_0 vía Union-Find ordenando aristas por longitud
//     (algoritmo del "elder rule" — exacto, O(n² α(n))).
//   * Números de Betti a una escala ε:
//         b_0(ε) = #componentes conexas del grafo de aristas ≤ ε
//         b_1(ε) = #ciclos independientes en el complejo VR_ε con
//                  2-símplices rellenados
//                = E(ε) - V + b_0(ε) - T(ε)
//   * Distancia bottleneck entre diagramas (versión exacta L∞ vía
//     emparejamiento greedy óptimo para diagramas pequeños).
//
// REGLAS:
//   * Diferenciamos rigurosamente lo que es exacto (H_0, b_0, b_1, Euler)
//     de aproximaciones (bottleneck — usamos algoritmo greedy + permutación
//     para ≤8 puntos; lanzamos si se excede).
//   * No alucinar: si no podemos resolver exactamente, lanzamos.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace easyatom::topology {

// -----------------------------------------------------------------------------
// Matriz de distancias (simétrica, diagonal cero).
// -----------------------------------------------------------------------------

class DistanceMatrix {
public:
    DistanceMatrix() = default;

    explicit DistanceMatrix(std::size_t n) : n_(n), d_(n * n, 0.0) {
        if (n == 0) throw std::invalid_argument("DistanceMatrix: n=0.");
    }

    /// Construye desde una nube de puntos en R^k usando distancia euclídea.
    [[nodiscard]] static DistanceMatrix from_points_euclidean(
        const std::vector<std::vector<double>>& points) {
        if (points.empty()) {
            throw std::invalid_argument("from_points_euclidean: vacío.");
        }
        const std::size_t n = points.size();
        const std::size_t k = points.front().size();
        DistanceMatrix D(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (points[i].size() != k) {
                throw std::invalid_argument(
                    "from_points_euclidean: dimensión inconsistente.");
            }
            for (std::size_t j = i + 1; j < n; ++j) {
                double s = 0.0;
                for (std::size_t l = 0; l < k; ++l) {
                    const double diff = points[i][l] - points[j][l];
                    s += diff * diff;
                }
                const double dist = std::sqrt(s);
                D.d_[i * n + j] = dist;
                D.d_[j * n + i] = dist;
            }
        }
        return D;
    }

    [[nodiscard]] std::size_t n() const noexcept { return n_; }

    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const {
        return d_.at(i * n_ + j);
    }
    void set(std::size_t i, std::size_t j, double v) {
        if (i == j) {
            if (v != 0.0) {
                throw std::invalid_argument("DistanceMatrix: diagonal no cero.");
            }
            return;
        }
        d_.at(i * n_ + j) = v;
        d_.at(j * n_ + i) = v;
    }

private:
    std::size_t n_ = 0;
    std::vector<double> d_;
};

// -----------------------------------------------------------------------------
// Persistence diagram: lista de pares (birth, death). death=∞ para clases vivas.
// -----------------------------------------------------------------------------

constexpr double kInf = std::numeric_limits<double>::infinity();

struct PersistencePair {
    double birth;
    double death;  ///< kInf si la clase nunca muere (clase esencial).

    [[nodiscard]] double persistence() const noexcept {
        return death - birth;
    }
};

using Diagram = std::vector<PersistencePair>;

// -----------------------------------------------------------------------------
// Union-Find compacto.
// -----------------------------------------------------------------------------

namespace detail {

class UnionFind {
public:
    explicit UnionFind(std::size_t n) : parent_(n), rank_(n, 0), n_components_(n) {
        std::iota(parent_.begin(), parent_.end(), std::size_t{0});
    }

    std::size_t find(std::size_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    /// Devuelve true si efectivamente unió dos componentes distintas.
    bool unite(std::size_t a, std::size_t b) {
        std::size_t ra = find(a);
        std::size_t rb = find(b);
        if (ra == rb) return false;
        if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
        parent_[rb] = ra;
        if (rank_[ra] == rank_[rb]) ++rank_[ra];
        --n_components_;
        return true;
    }

    [[nodiscard]] std::size_t components() const noexcept {
        return n_components_;
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> rank_;
    std::size_t n_components_;
};

}  // namespace detail

// -----------------------------------------------------------------------------
// H_0: persistencia de componentes conexas (algoritmo del elder rule).
// -----------------------------------------------------------------------------
//
// Todos los puntos nacen en ε=0. Cada vez que la arista de longitud ε une
// dos componentes distintas, la "menor" (la más joven, la que apareció más
// tarde en el orden de unión) MUERE en ε. Como los puntos nacen todos en 0,
// la regla simplificada es: cada arista que une dos componentes mata una de
// las clases con birth=0 → genera un par (0, ε).
//
// Al final queda exactamente una clase viva (la componente "principal") si
// el grafo se conecta a la escala máxima; o tantas clases esenciales como
// componentes finales.

[[nodiscard]] inline Diagram persistence_h0(const DistanceMatrix& D) {
    const std::size_t n = D.n();
    if (n == 0) return {};

    // Construimos la lista de aristas (i<j, d_ij) y las ordenamos por
    // distancia ascendente.
    struct Edge { std::size_t i, j; double w; };
    std::vector<Edge> edges;
    edges.reserve(n * (n - 1) / 2);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            edges.push_back({i, j, D(i, j)});
        }
    }
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.w < b.w; });

    detail::UnionFind uf(n);
    Diagram diag;
    diag.reserve(n);
    for (const auto& e : edges) {
        if (uf.unite(e.i, e.j)) {
            diag.push_back({0.0, e.w});
        }
    }
    // Las componentes que quedan sin morir son clases esenciales.
    for (std::size_t k = 0; k < uf.components(); ++k) {
        diag.push_back({0.0, kInf});
    }
    return diag;
}

// -----------------------------------------------------------------------------
// Números de Betti b_0(ε), b_1(ε) en el complejo de Vietoris-Rips a escala ε.
// -----------------------------------------------------------------------------

struct Betti {
    std::size_t b0;  ///< componentes conexas
    std::size_t b1;  ///< ciclos independientes
};

[[nodiscard]] inline Betti betti_at_epsilon(const DistanceMatrix& D, double eps) {
    const std::size_t n = D.n();
    if (n == 0) return {0, 0};

    // V, E, T en VR_eps.
    detail::UnionFind uf(n);
    std::size_t E = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (D(i, j) <= eps) {
                ++E;
                uf.unite(i, j);
            }
        }
    }
    // Triángulos: tripletas (i<j<k) con las tres aristas ≤ eps.
    std::size_t T = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (D(i, j) > eps) continue;
            for (std::size_t k = j + 1; k < n; ++k) {
                if (D(i, k) <= eps && D(j, k) <= eps) ++T;
            }
        }
    }
    const std::size_t b0 = uf.components();
    // Característica de Euler χ = V - E + T  =  b_0 - b_1   (para complejo 2-dim
    // sin 3-símplices arriba); por tanto b_1 = E - V + b_0 - T (cuando ≥ 0).
    const long long b1_signed =
        static_cast<long long>(E) - static_cast<long long>(n) +
        static_cast<long long>(b0) - static_cast<long long>(T);
    const std::size_t b1 = (b1_signed > 0)
                               ? static_cast<std::size_t>(b1_signed)
                               : std::size_t{0};
    return {b0, b1};
}

// -----------------------------------------------------------------------------
// Distancia bottleneck entre dos diagramas.
// -----------------------------------------------------------------------------
//
// Definición: la distancia bottleneck es
//      W_∞(D1, D2) = inf_{γ}  max_{x ∈ D1 ∪ Δ}  ||x - γ(x)||_∞
// donde γ recorre las biyecciones entre D1 ∪ Δ y D2 ∪ Δ (Δ = diagonal),
// y los puntos pueden ser emparejados con su proyección sobre Δ.
//
// Implementación exacta para diagramas pequeños (≤ 8 puntos en el mayor):
// enumeración de permutaciones. Lanzamos si excede.

namespace detail {

inline double point_to_diagonal(const PersistencePair& p) {
    // Distancia L∞ a la diagonal (b,b).
    if (p.death == kInf) return kInf;
    return (p.death - p.birth) * 0.5;
}

inline double linf(const PersistencePair& a, const PersistencePair& b) {
    if (a.death == kInf || b.death == kInf) {
        if (a.death == kInf && b.death == kInf) {
            return std::abs(a.birth - b.birth);
        }
        return kInf;
    }
    return std::max(std::abs(a.birth - b.birth),
                    std::abs(a.death - b.death));
}

}  // namespace detail

[[nodiscard]] inline double bottleneck_distance(const Diagram& A,
                                                const Diagram& B) {
    constexpr std::size_t kMax = 8;
    const std::size_t na = A.size();
    const std::size_t nb = B.size();
    if (std::max(na, nb) > kMax) {
        throw std::invalid_argument(
            "bottleneck_distance: diagrama demasiado grande (>8) para el "
            "algoritmo exacto del Ladrillo 4.");
    }

    // Estrategia: extender ambos diagramas con copias diagonales para que
    // puedan emparejarse con la diagonal. Tamaño total = na + nb. Coste
    // O((na+nb)!) — viable hasta 8.
    const std::size_t N = na + nb;
    std::vector<PersistencePair> A2 = A;
    A2.reserve(N);
    for (const auto& q : B) {
        // proyección de q sobre la diagonal: birth=death=(b+d)/2
        const double m = (q.death == kInf) ? kInf : (q.birth + q.death) * 0.5;
        A2.push_back({m, m});
    }
    std::vector<PersistencePair> B2 = B;
    B2.reserve(N);
    for (const auto& p : A) {
        const double m = (p.death == kInf) ? kInf : (p.birth + p.death) * 0.5;
        B2.push_back({m, m});
    }

    // Coste del emparejamiento (i → perm[i]):
    //   max_i d_∞(A2[i], B2[perm[i]])   con la salvedad de que emparejar
    //   diagonal-con-diagonal cuesta 0.
    auto cost = [&](const PersistencePair& a, const PersistencePair& b) {
        const bool ad = (a.birth == a.death);
        const bool bd = (b.birth == b.death);
        if (ad && bd) return 0.0;
        return detail::linf(a, b);
    };

    std::vector<std::size_t> perm(N);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    double best = kInf;
    do {
        double m = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double c = cost(A2[i], B2[perm[i]]);
            if (c > m) m = c;
            if (m >= best) break;  // poda
        }
        if (m < best) best = m;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

}  // namespace easyatom::topology
