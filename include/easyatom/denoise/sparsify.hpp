// =============================================================================
// easyatom/denoise/sparsify.hpp  --  L25
//
// Reduccion AGRESIVA de entropia. Resuelve la queja: "post_entropy 0.75-0.98
// es ruido residual demasiado alto -> el vector queda mezclado, sin
// proposicion robusta".
//
// L22 baja la entropia con un umbral fijo theta. Aqui:
//
//   1) quantile_theta(coefs, keep_ratio): elige theta automaticamente para
//      conservar exactamente top-(keep_ratio*M) componentes por magnitud.
//
//   2) denoise_sparse(s, anchors, keep_ratio): proyecta sobre el span de
//      anchors (Gram-Schmidt de L22), calcula coeficientes, escoge theta
//      por cuantil y aplica el filtrado. Garantiza concentracion fuerte.
//
//   3) sparsify_to_top_k(s, anchors, k): conserva exactamente las k
//      componentes mas fuertes. Para "enfocarse en senales fuertes".
//
//   4) effective_rank(coefs, eps): cuantas componentes superan eps. Mide
//      "cuantas direcciones reales tiene el state" en este span.
//
//   5) concentration(coefs): post_entropy / log2(M). En [0,1]; cerca de 0
//      = una sola direccion domina; cerca de 1 = ruido uniforme. Es el
//      indicador que se pide reportar para certificar "proposicion
//      robusta".
//
// Header-only, C++20 puro. Reusa Gram-Schmidt y proyeccion de L22.
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "easyatom/denoise/entropy.hpp"
#include "easyatom/hilbert/state.hpp"

namespace easyatom::denoise {

// -----------------------------------------------------------------------------
// quantile_theta
// -----------------------------------------------------------------------------
//
// Dado |coefs|, devuelve un theta tal que exactamente ceil(keep_ratio * M)
// magnitudes |c_k| son >= theta. Si keep_ratio*M no es entero, redondea
// hacia arriba (mejor conservar 1 mas que 1 menos).

[[nodiscard]] inline double quantile_theta(
    const std::vector<std::complex<double>>& coefs,
    double keep_ratio)
{
    if (coefs.empty())
        throw std::invalid_argument("quantile_theta: coefs vacio.");
    if (keep_ratio <= 0.0 || keep_ratio > 1.0)
        throw std::invalid_argument("quantile_theta: keep_ratio en (0,1].");

    const std::size_t M    = coefs.size();
    std::size_t       keep = static_cast<std::size_t>(
        std::ceil(keep_ratio * static_cast<double>(M)));
    if (keep < 1) keep = 1;
    if (keep > M) keep = M;

    std::vector<double> mags;
    mags.reserve(M);
    for (const auto& c : coefs) mags.push_back(std::abs(c));

    // Ordena descendente; theta = magnitud del keep-esimo (1-indexado).
    std::sort(mags.begin(), mags.end(), std::greater<double>());
    return mags[keep - 1];
}

// -----------------------------------------------------------------------------
// effective_rank y concentration
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::size_t effective_rank(
    const std::vector<std::complex<double>>& coefs, double eps = 1e-9)
{
    std::size_t r = 0;
    for (const auto& c : coefs) if (std::abs(c) > eps) ++r;
    return r;
}

// Entropia normalizada en [0,1]. 0 = una direccion domina (ideal).
// 1 = uniforme (ruido total). Evalua sobre |c_k|^2 normalizado.
[[nodiscard]] inline double concentration(
    const std::vector<std::complex<double>>& coefs)
{
    if (coefs.size() <= 1) return 0.0;
    double Z = 0.0;
    for (const auto& c : coefs) Z += std::norm(c);
    if (Z <= 0.0) return 0.0;
    double H = 0.0;
    for (const auto& c : coefs) {
        const double p = std::norm(c) / Z;
        if (p > 0.0) H -= p * std::log2(p);
    }
    const double Hmax = std::log2(static_cast<double>(coefs.size()));
    if (Hmax <= 0.0) return 0.0;
    return H / Hmax;
}

// -----------------------------------------------------------------------------
// SparsifyResult
// -----------------------------------------------------------------------------

struct SparsifyResult {
    hilbert::State filtered;
    std::size_t    basis_size;
    std::size_t    kept;
    double         theta_used;
    double         pre_entropy;       // Shannon sobre |coef|^2
    double         post_entropy;      // Shannon despues del filtrado
    double         pre_concentration;   // entropia normalizada en [0,1]
    double         post_concentration;
    double         residual_norm2;
};

namespace detail {

// Calcula coefs en la base ortonormal y reproyecta segun theta.
[[nodiscard]] inline SparsifyResult apply_with_theta(
    const hilbert::State&             s,
    const std::vector<hilbert::State>& anchors,
    double                            theta)
{
    auto basis = gram_schmidt(anchors);
    if (basis.empty())
        throw std::invalid_argument("sparsify: base vacia tras Gram-Schmidt.");

    const std::size_t M = basis.size();
    std::vector<std::complex<double>> coefs(M);
    for (std::size_t k = 0; k < M; ++k) coefs[k] = inner(basis[k], s);

    // Reconstruye proyeccion completa para residual.
    hilbert::State proj(s.dim());
    for (std::size_t k = 0; k < M; ++k) {
        const auto& a_e = basis[k].amplitudes();
        const auto  ck  = coefs[k];
        for (std::size_t i = 0; i < s.dim(); ++i)
            proj[i] = proj[i] + ck * a_e[i];
    }

    double residual_norm2 = 0.0;
    {
        const auto& a_s = s.amplitudes();
        for (std::size_t i = 0; i < s.dim(); ++i)
            residual_norm2 += std::norm(a_s[i] - proj[i]);
    }

    // Aplica umbral.
    std::vector<std::complex<double>> kept_coefs(M, std::complex<double>{0,0});
    std::size_t kept = 0;
    for (std::size_t k = 0; k < M; ++k) {
        if (std::abs(coefs[k]) >= theta) {
            kept_coefs[k] = coefs[k];
            ++kept;
        }
    }

    hilbert::State filt(s.dim());
    for (std::size_t k = 0; k < M; ++k) {
        if (kept_coefs[k] == std::complex<double>{0,0}) continue;
        const auto& a_e = basis[k].amplitudes();
        const auto  ck  = kept_coefs[k];
        for (std::size_t i = 0; i < s.dim(); ++i)
            filt[i] = filt[i] + ck * a_e[i];
    }

    // Pesos para Shannon.
    std::vector<double> w_pre(M), w_post(M);
    for (std::size_t k = 0; k < M; ++k) {
        w_pre[k]  = std::norm(coefs[k]);
        w_post[k] = std::norm(kept_coefs[k]);
    }

    SparsifyResult r;
    r.filtered           = std::move(filt);
    r.basis_size         = M;
    r.kept               = kept;
    r.theta_used         = theta;
    r.pre_entropy        = detail::shannon(w_pre);
    r.post_entropy       = detail::shannon(w_post);
    r.pre_concentration  = concentration(coefs);
    r.post_concentration = concentration(kept_coefs);
    r.residual_norm2     = residual_norm2;
    return r;
}

} // namespace detail

// -----------------------------------------------------------------------------
// denoise_sparse: theta auto por cuantil
// -----------------------------------------------------------------------------

[[nodiscard]] inline SparsifyResult denoise_sparse(
    const hilbert::State&              s,
    const std::vector<hilbert::State>& anchors,
    double                             keep_ratio)
{
    if (anchors.empty())
        throw std::invalid_argument("denoise_sparse: anchors vacio.");
    if (keep_ratio <= 0.0 || keep_ratio > 1.0)
        throw std::invalid_argument("denoise_sparse: keep_ratio en (0,1].");

    auto basis = gram_schmidt(anchors);
    if (basis.empty())
        throw std::invalid_argument("denoise_sparse: base vacia.");

    std::vector<std::complex<double>> coefs(basis.size());
    for (std::size_t k = 0; k < basis.size(); ++k)
        coefs[k] = inner(basis[k], s);

    const double theta = quantile_theta(coefs, keep_ratio);
    return detail::apply_with_theta(s, anchors, theta);
}

// -----------------------------------------------------------------------------
// sparsify_to_top_k: top-k EXACTO
// -----------------------------------------------------------------------------

[[nodiscard]] inline SparsifyResult sparsify_to_top_k(
    const hilbert::State&              s,
    const std::vector<hilbert::State>& anchors,
    std::size_t                        k)
{
    if (anchors.empty())
        throw std::invalid_argument("sparsify_to_top_k: anchors vacio.");
    if (k == 0)
        throw std::invalid_argument("sparsify_to_top_k: k=0.");

    auto basis = gram_schmidt(anchors);
    if (basis.empty())
        throw std::invalid_argument("sparsify_to_top_k: base vacia.");
    const std::size_t M = basis.size();
    if (k > M) k = M;

    std::vector<std::complex<double>> coefs(M);
    for (std::size_t i = 0; i < M; ++i) coefs[i] = inner(basis[i], s);

    // theta = k-esima magnitud (1-indexado).
    std::vector<double> mags(M);
    for (std::size_t i = 0; i < M; ++i) mags[i] = std::abs(coefs[i]);
    std::sort(mags.begin(), mags.end(), std::greater<double>());
    const double theta = mags[k - 1];

    return detail::apply_with_theta(s, anchors, theta);
}

} // namespace easyatom::denoise
