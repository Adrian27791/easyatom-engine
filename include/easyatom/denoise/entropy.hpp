// =============================================================================
// EasyAtom · Ladrillo 22 — Filtro de ruido y reduccion de entropia.
// =============================================================================
//
// Dado un estado `s` en H_D y un conjunto de "anclas" (subset relevante del
// codebook) {a_1, ..., a_M}, proyectamos s sobre el subespacio que generan
// (ortonormalizado por Gram-Schmidt modificado) y eliminamos las componentes
// debiles |c_i| < theta:
//
//      s_filtered = sum_{|c_i|>=theta}  c_i * e_i
//
//   donde {e_i} es la base ortonormal obtenida por GS sobre {a_i}.
//
// Esto cumple dos objetivos del usuario:
//
//   1. REDUCCION DE ENTROPIA:
//      la distribucion de Born sobre las anclas (p_i ∝ |c_i|^2) se vuelve
//      mas concentrada al eliminar componentes ruidosas. Reportamos la
//      entropia de Shannon antes y despues para que el llamador verifique
//      H_post <= H_pre.
//
//   2. FILTRADO DE RUIDO:
//      la componente residual de s ortogonal al subespacio queda descartada.
//      En particular: s' contiene SOLO informacion expresable en el lenguaje
//      del codebook anclado. Si la pregunta cae fuera del subespacio, el
//      motor lo detecta (componentes muy pequeñas en todos los anchors) en
//      vez de inventar.
//
// Sin SVD, sin BLAS, sin dependencias. Gram-Schmidt modificado en O(M^2 * D),
// reproyeccion en O(M * D). Para M <= 64 y D = 32768 es trivial en runtime.
//
// Insumo natural para L23 (coherencia: medir entropia tras añadir leyes) y
// L24 (lagunas: residual alto = pregunta fuera del soporte).

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "easyatom/hilbert/state.hpp"

namespace easyatom::denoise {

using easyatom::hilbert::Complex;
using easyatom::hilbert::State;

// -----------------------------------------------------------------------------
// Producto interno reutilizado de hilbert::inner (mismo convenio: <a,b> con
// conjugado en la izquierda).
// -----------------------------------------------------------------------------

using easyatom::hilbert::inner;

// -----------------------------------------------------------------------------
// Gram-Schmidt modificado sobre un conjunto de estados (orden conservado).
// Devuelve una base ortonormal del subespacio que generan; descarta vectores
// linealmente dependientes (norma residual < eps).
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::vector<State> gram_schmidt(
    const std::vector<State>& anchors, double eps = 1e-12) {
    std::vector<State> basis;
    basis.reserve(anchors.size());
    for (const auto& a : anchors) {
        if (basis.empty() && a.dim() == 0)
            throw std::invalid_argument("gram_schmidt: estado dim 0.");
        State u = a;
        for (const auto& e : basis) {
            const Complex c = inner(e, u);
            for (std::size_t i = 0; i < u.dim(); ++i)
                u[i] -= c * e[i];
        }
        const double n2 = u.norm_squared();
        if (n2 < eps) continue;
        const double inv = 1.0 / std::sqrt(n2);
        for (std::size_t i = 0; i < u.dim(); ++i)
            u[i] = u[i] * Complex{inv, 0.0};
        basis.push_back(std::move(u));
    }
    return basis;
}

// -----------------------------------------------------------------------------
// Resultado del denoising.
// -----------------------------------------------------------------------------

struct DenoiseResult {
    State        filtered;             // s'
    std::size_t  basis_size = 0;       // dim del subespacio anclado tras GS
    std::size_t  kept       = 0;       // componentes con |c_i| >= theta
    double       pre_entropy  = 0.0;   // H(p_i^pre)  con p_i^pre = |<e_i,s>|^2 / Z_pre
    double       post_entropy = 0.0;   // H(p_i^post) con p_i^post = |c_i'|^2  / Z_post
    double       residual_norm2 = 0.0; // ||s - sum c_i e_i||^2 ANTES del umbral
};

namespace detail {
[[nodiscard]] inline double shannon(const std::vector<double>& w) {
    double sum = 0.0;
    for (double x : w) sum += x;
    if (sum <= 0.0) return 0.0;
    double H = 0.0;
    for (double x : w) {
        if (x <= 0.0) continue;
        const double p = x / sum;
        H -= p * std::log(p);
    }
    return H;
}
}  // namespace detail

// -----------------------------------------------------------------------------
// denoise(s, anchors, theta) — filtro principal.
// -----------------------------------------------------------------------------

[[nodiscard]] inline DenoiseResult denoise(
    const State& s, const std::vector<State>& anchors, double theta) {
    if (anchors.empty())
        throw std::invalid_argument("denoise: anchors vacios.");
    if (theta < 0.0)
        throw std::invalid_argument("denoise: theta < 0.");

    auto basis = gram_schmidt(anchors);
    if (basis.empty())
        throw std::invalid_argument("denoise: subespacio vacio tras GS.");
    const std::size_t D = s.dim();
    if (basis.front().dim() != D)
        throw std::invalid_argument("denoise: dim incoherente.");

    // Coeficientes en la base ortonormal.
    std::vector<Complex> coef;
    coef.reserve(basis.size());
    std::vector<double> w_pre;
    w_pre.reserve(basis.size());
    for (const auto& e : basis) {
        const Complex c = inner(e, s);
        coef.push_back(c);
        w_pre.push_back(std::norm(c));
    }

    // Reconstruir la proyeccion (sin umbral) para residual.
    State proj(D);
    for (std::size_t k = 0; k < basis.size(); ++k) {
        const Complex c = coef[k];
        const auto& e = basis[k];
        for (std::size_t i = 0; i < D; ++i) proj[i] += c * e[i];
    }
    State residual(D);
    for (std::size_t i = 0; i < D; ++i) residual[i] = s[i] - proj[i];
    const double residual_n2 = residual.norm_squared();

    // Aplicar umbral.
    State filtered(D);
    std::size_t kept = 0;
    std::vector<double> w_post;
    w_post.reserve(basis.size());
    for (std::size_t k = 0; k < basis.size(); ++k) {
        const Complex c = coef[k];
        if (std::abs(c) < theta) {
            w_post.push_back(0.0);
            continue;
        }
        ++kept;
        w_post.push_back(std::norm(c));
        const auto& e = basis[k];
        for (std::size_t i = 0; i < D; ++i) filtered[i] += c * e[i];
    }

    DenoiseResult r;
    r.filtered       = std::move(filtered);
    r.basis_size     = basis.size();
    r.kept           = kept;
    r.pre_entropy    = detail::shannon(w_pre);
    r.post_entropy   = detail::shannon(w_post);
    r.residual_norm2 = residual_n2;
    return r;
}

}  // namespace easyatom::denoise
