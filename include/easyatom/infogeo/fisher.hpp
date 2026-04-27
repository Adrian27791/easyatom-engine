// =============================================================================
// EasyAtom · Ladrillo 3 — Geometría de la información sobre el símplex.
// =============================================================================
//
// Una distribución de probabilidad discreta p ∈ Δ^{n-1} es un punto en el
// símplex (p_i ≥ 0, sum p_i = 1). El conjunto de todas las distribuciones
// es una VARIEDAD RIEMANNIANA con una métrica natural — la métrica de
// Fisher — y dos conexiones afines duales — las α-conexiones de Amari.
//
// Esta es la geometría correcta para hablar de "cerca / lejos" entre
// estados de creencia, modelos, o categorías. Es lo que convierte al motor
// en algo que *mide* información, no que la cuenta.
//
// Definiciones (todas verificadas en tests):
//
//   1. Coeficiente de Bhattacharyya:
//          BC(p, q) = sum_i sqrt(p_i q_i)              ∈ [0, 1]
//
//   2. Distancia de Hellinger:
//          H(p, q) = sqrt( (1/2) sum_i (sqrt(p_i) - sqrt(q_i))^2 )
//                  = sqrt(1 - BC(p, q))                ∈ [0, 1]
//      Métrica verdadera; simétrica; H=0 ⇔ p=q.
//
//   3. Distancia de Fisher-Rao (geodésica en la métrica de Fisher):
//          d_FR(p, q) = 2 arccos( BC(p, q) )           ∈ [0, π]
//      Es la longitud del arco esférico entre sqrt(p) y sqrt(q) en S^{n-1}.
//      Es una métrica genuina (simétrica, definida positiva, desigualdad
//      triangular).
//
//   4. Divergencia de Kullback-Leibler:
//          KL(p || q) = sum_i p_i log(p_i / q_i)       ≥ 0
//      NO es simétrica; es la divergencia inducida por la 1-conexión.
//
//   5. α-divergencia de Amari (1-parámetro, generaliza KL/Hellinger):
//          D_α(p, q) = 4 / (1 - α²) · ( 1 - sum_i p_i^{(1-α)/2} q_i^{(1+α)/2} )
//      Casos límite (definidos por continuidad):
//          α = +1  →  KL(p || q)
//          α = -1  →  KL(q || p)
//          α =  0  →  4 · H(p, q)²  =  4 (1 - BC)
//      Dualidad: D_α(p, q) = D_{-α}(q, p).
//
//   6. Métrica de Fisher (forma diagonal en coordenadas del símplex):
//          g_ii(p) = 1 / p_i        (sin acoplamientos cruzados)
//      Implica que la "longitud" de un vector tangente v en p es
//          ||v||_p² = sum_i v_i² / p_i.
//
//   7. Geodésica de Fisher-Rao:
//      Interpolación esférica de los radicales sqrt(p), sqrt(q):
//          γ(t)_i = ( sin((1-t)·θ)·sqrt(p_i) + sin(t·θ)·sqrt(q_i) ) / sin(θ)
//          con θ = arccos( BC(p,q) ).
//      γ(0) = p, γ(1) = q. Caso degenerado p≈q: interpolación lineal.
//
// REGLAS DEL LADRILLO 3:
//   * Las distribuciones se validan al construirse: p_i > 0 y suma ≈ 1.
//   * KL y α-divergencia EXIGEN p_i, q_i > 0 (no las extendemos a soporte
//     parcial de forma silenciosa; eso sería alucinar).
//   * Hellinger / Fisher-Rao toleran ceros (sqrt(0) = 0 está bien definido).
//
// Este ladrillo no depende del Ladrillo 2: opera en el símplex, no en H_D.
// Pero se conecta naturalmente con el Ladrillo 1 vía la inmersión
// p ↦ sqrt(p) ∈ S^{n-1} ⊂ R^n.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace easyatom::infogeo {

// -----------------------------------------------------------------------------
// Distribution — punto en el símplex Δ^{n-1}.
// -----------------------------------------------------------------------------

class Distribution {
public:
    Distribution() = default;

    explicit Distribution(std::vector<double> p, double sum_tol = 1e-9)
        : p_(std::move(p)) {
        if (p_.empty()) {
            throw std::invalid_argument("Distribution: vector vacío.");
        }
        double s = 0.0;
        for (double x : p_) {
            if (x < 0.0) {
                throw std::invalid_argument(
                    "Distribution: probabilidad negativa.");
            }
            s += x;
        }
        if (std::abs(s - 1.0) > sum_tol) {
            throw std::invalid_argument(
                "Distribution: la suma no es 1 (es " + std::to_string(s) + ").");
        }
    }

    /// Construye uniforme sobre n estados.
    [[nodiscard]] static Distribution uniform(std::size_t n) {
        if (n == 0) throw std::invalid_argument("uniform: n=0.");
        std::vector<double> v(n, 1.0 / static_cast<double>(n));
        return Distribution(std::move(v));
    }

    /// Construye desde scores no normalizados (todos ≥ 0, no todos cero).
    [[nodiscard]] static Distribution from_scores(const std::vector<double>& s) {
        if (s.empty()) throw std::invalid_argument("from_scores: vacío.");
        double total = 0.0;
        for (double x : s) {
            if (x < 0.0) {
                throw std::invalid_argument("from_scores: score negativo.");
            }
            total += x;
        }
        if (total <= 0.0) {
            throw std::invalid_argument("from_scores: suma no positiva.");
        }
        std::vector<double> v(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) v[i] = s[i] / total;
        return Distribution(std::move(v));
    }

    [[nodiscard]] std::size_t dim() const noexcept { return p_.size(); }
    [[nodiscard]] const std::vector<double>& probs() const noexcept { return p_; }
    [[nodiscard]] double operator[](std::size_t i) const { return p_.at(i); }

    /// ¿Tiene soporte estricto (todas las p_i > 0)?
    [[nodiscard]] bool is_strictly_positive(double eps = 0.0) const noexcept {
        for (double x : p_) if (x <= eps) return false;
        return true;
    }

    [[nodiscard]] bool approx_equal(const Distribution& o, double tol = 1e-12) const {
        if (dim() != o.dim()) return false;
        for (std::size_t i = 0; i < dim(); ++i) {
            if (std::abs(p_[i] - o.p_[i]) > tol) return false;
        }
        return true;
    }

private:
    std::vector<double> p_;
};

// -----------------------------------------------------------------------------
// Helpers internos.
// -----------------------------------------------------------------------------

namespace detail {

inline void require_same_dim(const Distribution& p, const Distribution& q,
                             const char* op) {
    if (p.dim() != q.dim()) {
        throw std::invalid_argument(
            std::string("infogeo::") + op + ": dimensiones distintas.");
    }
}

inline void require_strictly_positive(const Distribution& p, const char* op) {
    if (!p.is_strictly_positive()) {
        throw std::domain_error(
            std::string("infogeo::") + op +
            ": distribución con probabilidad cero (no admitido).");
    }
}

}  // namespace detail

// -----------------------------------------------------------------------------
// Coeficiente de Bhattacharyya y distancia de Hellinger.
// -----------------------------------------------------------------------------

[[nodiscard]] inline double bhattacharyya_coefficient(const Distribution& p,
                                                      const Distribution& q) {
    detail::require_same_dim(p, q, "bhattacharyya");
    double bc = 0.0;
    for (std::size_t i = 0; i < p.dim(); ++i) {
        bc += std::sqrt(p[i] * q[i]);
    }
    // Pequeñas oscilaciones numéricas pueden dejarlo justo por encima de 1.
    if (bc > 1.0) bc = 1.0;
    if (bc < 0.0) bc = 0.0;
    return bc;
}

[[nodiscard]] inline double hellinger_distance(const Distribution& p,
                                               const Distribution& q) {
    const double bc = bhattacharyya_coefficient(p, q);
    return std::sqrt(std::max(0.0, 1.0 - bc));
}

// -----------------------------------------------------------------------------
// Distancia de Fisher-Rao.
// -----------------------------------------------------------------------------

[[nodiscard]] inline double fisher_rao_distance(const Distribution& p,
                                                const Distribution& q) {
    const double bc = bhattacharyya_coefficient(p, q);
    // arccos seguro en [-1, 1]
    const double c = std::clamp(bc, -1.0, 1.0);
    return 2.0 * std::acos(c);
}

// -----------------------------------------------------------------------------
// Divergencia de Kullback-Leibler.
// -----------------------------------------------------------------------------

[[nodiscard]] inline double kl_divergence(const Distribution& p,
                                          const Distribution& q) {
    detail::require_same_dim(p, q, "kl_divergence");
    detail::require_strictly_positive(q, "kl_divergence (q)");
    double kl = 0.0;
    for (std::size_t i = 0; i < p.dim(); ++i) {
        if (p[i] == 0.0) continue;  // 0 log 0 = 0 por continuidad
        kl += p[i] * std::log(p[i] / q[i]);
    }
    // KL es ≥ 0 teóricamente; el redondeo puede dar -1e-17.
    if (kl < 0.0 && kl > -1e-12) kl = 0.0;
    return kl;
}

// -----------------------------------------------------------------------------
// α-divergencia de Amari.
// -----------------------------------------------------------------------------
//
//   α  ∈ (-1, 1):  fórmula directa.
//   α = +1:        KL(p || q)
//   α = -1:        KL(q || p)

[[nodiscard]] inline double alpha_divergence(const Distribution& p,
                                             const Distribution& q,
                                             double alpha) {
    detail::require_same_dim(p, q, "alpha_divergence");

    constexpr double kEdgeTol = 1e-9;
    if (std::abs(alpha - 1.0) < kEdgeTol) {
        return kl_divergence(p, q);
    }
    if (std::abs(alpha + 1.0) < kEdgeTol) {
        return kl_divergence(q, p);
    }

    const double a = (1.0 - alpha) * 0.5;
    const double b = (1.0 + alpha) * 0.5;
    double s = 0.0;
    for (std::size_t i = 0; i < p.dim(); ++i) {
        const double pi = p[i];
        const double qi = q[i];
        if (pi == 0.0 || qi == 0.0) {
            // pow(0, positivo) = 0; la contribución es 0.
            continue;
        }
        s += std::pow(pi, a) * std::pow(qi, b);
    }
    const double factor = 4.0 / (1.0 - alpha * alpha);
    double d = factor * (1.0 - s);
    if (d < 0.0 && d > -1e-12) d = 0.0;
    return d;
}

// -----------------------------------------------------------------------------
// Métrica de Fisher (diagonal en coordenadas del símplex).
// -----------------------------------------------------------------------------
//
// g_ii(p) = 1 / p_i. Devolvemos el vector de elementos diagonales.

[[nodiscard]] inline std::vector<double> fisher_metric_diagonal(
    const Distribution& p) {
    detail::require_strictly_positive(p, "fisher_metric_diagonal");
    std::vector<double> g(p.dim());
    for (std::size_t i = 0; i < p.dim(); ++i) g[i] = 1.0 / p[i];
    return g;
}

/// Norma al cuadrado de un vector tangente v en el punto p:
///   ||v||_p^2 = sum_i v_i^2 / p_i
/// Para que v sea genuinamente tangente al símplex se requiere sum v_i = 0,
/// pero esta función no lo impone — devuelve la forma cuadrática evaluada.
[[nodiscard]] inline double fisher_norm_squared(const Distribution& p,
                                                const std::vector<double>& v) {
    detail::require_strictly_positive(p, "fisher_norm_squared");
    if (v.size() != p.dim()) {
        throw std::invalid_argument(
            "fisher_norm_squared: dimensión de v ≠ dim(p).");
    }
    double acc = 0.0;
    for (std::size_t i = 0; i < p.dim(); ++i) acc += (v[i] * v[i]) / p[i];
    return acc;
}

// -----------------------------------------------------------------------------
// Geodésica de Fisher-Rao (interpolación esférica de sqrt(p), sqrt(q)).
// -----------------------------------------------------------------------------

[[nodiscard]] inline Distribution fisher_rao_geodesic(const Distribution& p,
                                                      const Distribution& q,
                                                      double t) {
    detail::require_same_dim(p, q, "fisher_rao_geodesic");
    if (t < 0.0 || t > 1.0) {
        throw std::invalid_argument("fisher_rao_geodesic: t fuera de [0,1].");
    }
    const double bc = bhattacharyya_coefficient(p, q);
    const double theta = std::acos(std::clamp(bc, -1.0, 1.0));

    std::vector<double> r(p.dim(), 0.0);

    if (theta < 1e-9) {
        // Casi coincidentes: interpolación lineal de las probabilidades.
        for (std::size_t i = 0; i < p.dim(); ++i) {
            r[i] = (1.0 - t) * p[i] + t * q[i];
        }
        return Distribution(std::move(r));
    }

    const double s = std::sin(theta);
    const double a = std::sin((1.0 - t) * theta) / s;
    const double b = std::sin(t * theta) / s;
    double sum = 0.0;
    for (std::size_t i = 0; i < p.dim(); ++i) {
        const double sp = std::sqrt(p[i]);
        const double sq = std::sqrt(q[i]);
        const double ri = a * sp + b * sq;
        // r_i debe ser ≥ 0 (puede ser ligeramente negativo por redondeo).
        const double ri_clamped = (ri < 0.0) ? 0.0 : ri;
        const double prob = ri_clamped * ri_clamped;
        r[i] = prob;
        sum += prob;
    }
    // Renormalización defensiva contra error de redondeo.
    if (sum > 0.0) {
        for (double& x : r) x /= sum;
    }
    return Distribution(std::move(r), 1e-6);
}

}  // namespace easyatom::infogeo
