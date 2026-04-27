// =============================================================================
// EasyAtom · Ladrillo 10 — Decisor geométrico + validación lógica.
// =============================================================================
//
// El motor hasta aquí REPRESENTA. Este ladrillo le da capacidad de DECIDIR:
// dada una Distribution sobre candidatos nombrados, produce una Decision
// con clasificación (ACCEPT / AMBIGUOUS / ABSTAIN / DEGENERATE) y un
// conjunto de invariantes geométricos que justifican (o invalidan) la
// elección.
//
// FILOSOFÍA:
//   * 0 alucinaciones: si la geometría no soporta la decisión, se ABSTIENE.
//   * Toda decisión lleva su propio certificado numérico (confidence,
//     margin, entropy, effective_n, reason_codes).
//   * Sin umbrales mágicos enterrados: todos viven en DecisionPolicy.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "easyatom/infogeo/fisher.hpp"

namespace easyatom::decide {

using easyatom::infogeo::Distribution;

// -----------------------------------------------------------------------------
// Política — umbrales explícitos.
// -----------------------------------------------------------------------------
struct DecisionPolicy {
    /// Probabilidad mínima del top-1 para no ABSTENERSE.
    double min_confidence       = 0.35;
    /// Diferencia mínima top1 − top2 para no marcar AMBIGUOUS.
    double min_margin           = 0.05;
    /// Razón H/H_max máxima admisible (1.0 = uniforme = degenerado).
    double max_entropy_ratio    = 0.92;
    /// n_efectivo máximo (= exp(H)). Por defecto desactivado (∞).
    double max_effective_n      = 0.0;  // 0 ⇒ desactivado
    /// Si true, exige distribución estrictamente positiva (no NaN, no 0).
    bool   require_finite_probs = true;

    [[nodiscard]] static DecisionPolicy strict() {
        DecisionPolicy p;
        p.min_confidence    = 0.55;
        p.min_margin        = 0.10;
        p.max_entropy_ratio = 0.85;
        return p;
    }
    [[nodiscard]] static DecisionPolicy permissive() {
        DecisionPolicy p;
        p.min_confidence    = 0.20;
        p.min_margin        = 0.02;
        p.max_entropy_ratio = 0.97;
        return p;
    }
};

enum class DecisionKind : std::uint8_t {
    Accept,      // hay un ganador claro y la geometría lo certifica
    Ambiguous,   // top1 y top2 demasiado cerca
    Abstain,     // top1 por debajo de min_confidence
    Degenerate,  // distribución casi uniforme (entropía demasiado alta)
    Invalid      // probs no finitas / dim 0 / inputs corruptos
};

[[nodiscard]] inline const char* decision_kind_name(DecisionKind k) noexcept {
    switch (k) {
        case DecisionKind::Accept:     return "accept";
        case DecisionKind::Ambiguous:  return "ambiguous";
        case DecisionKind::Abstain:    return "abstain";
        case DecisionKind::Degenerate: return "degenerate";
        case DecisionKind::Invalid:    return "invalid";
    }
    return "?";
}

struct Decision {
    DecisionKind kind = DecisionKind::Invalid;
    std::size_t  winner_index = 0;
    std::string  winner_name;          // vacío si Invalid
    std::size_t  runner_up_index = 0;
    std::string  runner_up_name;

    // Certificado numérico:
    double confidence    = 0.0;        // p[top1]
    double margin        = 0.0;        // p[top1] − p[top2]
    double entropy       = 0.0;        // H(p) en nats
    double entropy_ratio = 0.0;        // H / log(n)
    double effective_n   = 1.0;        // exp(H) — tamaño efectivo
    double min_prob      = 0.0;
    double max_prob      = 0.0;

    // Códigos legibles para auditoría (e.g. "low_confidence", "high_entropy").
    std::vector<std::string> reason_codes;

    /// ¿La decisión es accionable? (Accept es la única "sí" pura).
    [[nodiscard]] bool is_actionable() const noexcept {
        return kind == DecisionKind::Accept;
    }
};

// -----------------------------------------------------------------------------
// Validación de la propia distribución antes de decidir.
// -----------------------------------------------------------------------------

/// Comprueba que p sea procesable: no NaN, no negativos, dim>0.
/// Devuelve cadena vacía si OK; código de error si no.
[[nodiscard]] inline std::string validate_distribution_payload(
    const Distribution& p) noexcept
{
    if (p.dim() == 0) return "empty_distribution";
    for (double x : p.probs()) {
        if (std::isnan(x) || std::isinf(x)) return "non_finite_prob";
        if (x < 0.0) return "negative_prob";
    }
    return {};
}

// -----------------------------------------------------------------------------
// Métricas escalares de la distribución.
// -----------------------------------------------------------------------------

/// Entropía de Shannon en nats. 0 si dim=1 o p degenerada en un punto.
[[nodiscard]] inline double shannon_entropy(const Distribution& p) noexcept {
    double h = 0.0;
    for (double x : p.probs()) {
        if (x > 0.0) h -= x * std::log(x);
    }
    return h;
}

/// Tamaño efectivo del soporte: exp(H). 1 = pico puro, n = uniforme.
[[nodiscard]] inline double effective_support(const Distribution& p) noexcept {
    return std::exp(shannon_entropy(p));
}

// -----------------------------------------------------------------------------
// El decisor — convierte Distribution + nombres + política → Decision.
// -----------------------------------------------------------------------------

[[nodiscard]] inline Decision decide(
    const Distribution& p,
    const std::vector<std::string>& names,
    const DecisionPolicy& pol = {})
{
    Decision d;
    if (names.size() != p.dim()) {
        d.kind = DecisionKind::Invalid;
        d.reason_codes.emplace_back("names_dim_mismatch");
        return d;
    }
    auto err = validate_distribution_payload(p);
    if (!err.empty()) {
        d.kind = DecisionKind::Invalid;
        d.reason_codes.push_back(std::move(err));
        return d;
    }

    // Top-1 y top-2.
    std::size_t i1 = 0, i2 = 0;
    double p1 = -1.0, p2 = -1.0;
    for (std::size_t i = 0; i < p.dim(); ++i) {
        const double pi = p[i];
        if (pi > p1) { p2 = p1; i2 = i1; p1 = pi; i1 = i; }
        else if (pi > p2) { p2 = pi; i2 = i; }
    }
    if (p2 < 0.0) { p2 = 0.0; i2 = i1; }  // dim==1

    d.winner_index    = i1;
    d.winner_name     = names[i1];
    d.runner_up_index = i2;
    d.runner_up_name  = names[i2];
    d.confidence      = p1;
    d.margin          = p1 - p2;

    d.entropy = shannon_entropy(p);
    const double h_max = (p.dim() > 1) ? std::log(static_cast<double>(p.dim())) : 1.0;
    d.entropy_ratio = (h_max > 0.0) ? (d.entropy / h_max) : 0.0;
    d.effective_n   = effective_support(p);

    auto [mn, mx] = std::minmax_element(p.probs().begin(), p.probs().end());
    d.min_prob = *mn;
    d.max_prob = *mx;

    if (pol.require_finite_probs && d.min_prob < 0.0) {
        d.kind = DecisionKind::Invalid;
        d.reason_codes.emplace_back("negative_prob");
        return d;
    }

    // Clasificación — orden estricto (el primero que dispara gana).
    if (p.dim() > 1 && d.entropy_ratio > pol.max_entropy_ratio) {
        d.kind = DecisionKind::Degenerate;
        d.reason_codes.emplace_back("high_entropy");
    }
    if (pol.max_effective_n > 0.0 && d.effective_n > pol.max_effective_n) {
        d.kind = DecisionKind::Degenerate;
        d.reason_codes.emplace_back("high_effective_n");
    }
    if (d.confidence < pol.min_confidence) {
        d.kind = DecisionKind::Abstain;
        d.reason_codes.emplace_back("low_confidence");
        return d;
    }
    if (p.dim() > 1 && d.margin < pol.min_margin) {
        d.kind = DecisionKind::Ambiguous;
        d.reason_codes.emplace_back("low_margin");
        return d;
    }
    if (d.kind == DecisionKind::Invalid) {
        // No disparó ninguna anomalía → decisión limpia.
        d.kind = DecisionKind::Accept;
    }
    return d;
}

// -----------------------------------------------------------------------------
// Validación adicional: distancia Fisher contra una distribución de
// referencia (p.ej. la "respuesta esperada" o un baseline). Útil para
// detectar drift de la geometría informacional.
// -----------------------------------------------------------------------------

struct FisherCheck {
    bool   passed = false;
    double distance = 0.0;
    double threshold = 0.0;
};

[[nodiscard]] inline FisherCheck fisher_consistency_check(
    const Distribution& p,
    const Distribution& reference,
    double max_distance)
{
    FisherCheck c;
    c.threshold = max_distance;
    if (p.dim() != reference.dim()) {
        c.passed = false;
        return c;
    }
    try {
        c.distance = easyatom::infogeo::fisher_rao_distance(p, reference);
    } catch (...) {
        c.passed = false;
        return c;
    }
    c.passed = (c.distance <= max_distance);
    return c;
}

}  // namespace easyatom::decide
