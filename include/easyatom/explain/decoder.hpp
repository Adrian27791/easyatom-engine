// =============================================================================
// EasyAtom · Ladrillo 11 — Decoder semántico (es-ES).
// =============================================================================
//
// Convierte una Decision + Distribution + (opcional) traza en una frase en
// español comprensible. NO reescribe ni inventa: cada frase es una
// proyección determinista del certificado numérico al lenguaje natural.
//
// REGLA CRÍTICA:
//   El decoder es PURO. Mismos números → misma frase. No hay aleatoriedad,
//   no hay "templates con variantes". Auditar la salida = auditar la
//   geometría.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "easyatom/decide/decisor.hpp"
#include "easyatom/infogeo/fisher.hpp"
#include "easyatom/introspect/trace.hpp"

namespace easyatom::explain {

using easyatom::decide::Decision;
using easyatom::decide::DecisionKind;
using easyatom::infogeo::Distribution;
using easyatom::introspect::Op;
using easyatom::introspect::TraceEvent;

// -----------------------------------------------------------------------------
// Helpers de formato puramente determinista.
// -----------------------------------------------------------------------------

namespace detail {

[[nodiscard]] inline std::string fmt_pct(double p) {
    if (std::isnan(p) || std::isinf(p)) return "indefinido";
    long long n = static_cast<long long>(std::round(p * 1000.0));
    if (n < 0) n = 0;
    if (n > 1000) n = 1000;
    const long long whole = n / 10;
    const long long frac  = n % 10;
    return std::to_string(whole) + "," + std::to_string(frac) + "%";
}

[[nodiscard]] inline std::string fmt_num(double v, int decimals = 3) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return (v < 0 ? "-inf" : "inf");
    double mul = 1.0;
    for (int i = 0; i < decimals; ++i) mul *= 10.0;
    long long n = static_cast<long long>(std::round(v * mul));
    std::string s = std::to_string(std::llabs(n));
    if (static_cast<int>(s.size()) <= decimals) {
        s = std::string(decimals + 1 - static_cast<int>(s.size()), '0') + s;
    }
    std::string out = s.substr(0, s.size() - decimals) + ","
                    + s.substr(s.size() - decimals);
    return (n < 0 ? "-" : "") + out;
}

[[nodiscard]] inline const char* confidence_word(double p) noexcept {
    if (std::isnan(p)) return "indefinida";
    if (p >= 0.85) return "muy alta";
    if (p >= 0.70) return "alta";
    if (p >= 0.50) return "media";
    if (p >= 0.35) return "baja";
    return "muy baja";
}

[[nodiscard]] inline std::string reasons_to_text(
    const std::vector<std::string>& codes)
{
    if (codes.empty()) return "ninguna anomalía";
    std::string out;
    for (std::size_t i = 0; i < codes.size(); ++i) {
        if (i) out += ", ";
        const auto& c = codes[i];
        if      (c == "low_confidence")     out += "confianza por debajo del umbral";
        else if (c == "low_margin")         out += "margen entre las dos primeras opciones demasiado estrecho";
        else if (c == "high_entropy")       out += "distribución demasiado dispersa (entropía alta)";
        else if (c == "high_effective_n")   out += "demasiados candidatos comparables";
        else if (c == "non_finite_prob")    out += "probabilidad no finita";
        else if (c == "negative_prob")      out += "probabilidad negativa";
        else if (c == "names_dim_mismatch") out += "nombres incompatibles con la distribución";
        else if (c == "empty_distribution") out += "distribución vacía";
        else                                out += c;
    }
    return out;
}

}  // namespace detail

// -----------------------------------------------------------------------------
// Decode de una Decision a frase humana.
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::string decode_decision(const Decision& d) {
    using detail::fmt_pct;
    using detail::fmt_num;
    using detail::confidence_word;

    const std::string conf  = fmt_pct(d.confidence);
    const std::string marg  = fmt_pct(d.margin);
    const std::string word  = confidence_word(d.confidence);
    const std::string reasons = detail::reasons_to_text(d.reason_codes);

    switch (d.kind) {
        case DecisionKind::Accept:
            return "El motor identifica '" + d.winner_name +
                   "' con confianza " + word + " (" + conf +
                   "), margen sobre la segunda opción ('" + d.runner_up_name +
                   "') de " + marg + ".";
        case DecisionKind::Ambiguous:
            return "El motor duda entre '" + d.winner_name + "' y '" +
                   d.runner_up_name + "'. La diferencia es de solo " + marg +
                   " (confianza del primero: " + conf + ", " + word + ").";
        case DecisionKind::Abstain:
            return "El motor se abstiene: ningún candidato alcanza la confianza "
                   "mínima (mejor opción '" + d.winner_name + "' con " + conf +
                   ", " + word + ").";
        case DecisionKind::Degenerate:
            return "La distribución es demasiado dispersa para decidir "
                   "(entropía relativa " + fmt_num(d.entropy_ratio, 3) +
                   ", n efectivo " + fmt_num(d.effective_n, 2) +
                   "). Motivos: " + reasons + ".";
        case DecisionKind::Invalid:
            return "Decisión inválida: " + reasons + ".";
    }
    return "Decisión desconocida.";
}

// -----------------------------------------------------------------------------
// Top-K legible.
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::string decode_topk(
    const Distribution& p,
    const std::vector<std::string>& names,
    std::size_t k = 3)
{
    if (names.size() != p.dim()) {
        throw std::invalid_argument(
            "decode_topk: nombres y distribución no concuerdan.");
    }
    if (k == 0) k = p.dim();
    if (k > p.dim()) k = p.dim();

    std::vector<std::size_t> idx(p.dim());
    for (std::size_t i = 0; i < p.dim(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + static_cast<long>(k), idx.end(),
        [&](std::size_t a, std::size_t b) { return p[a] > p[b]; });

    std::string out = "Top-" + std::to_string(k) + ": ";
    for (std::size_t r = 0; r < k; ++r) {
        if (r) out += "; ";
        out += std::to_string(r + 1) + ". '" + names[idx[r]] +
               "' " + detail::fmt_pct(p[idx[r]]);
    }
    out += ".";
    return out;
}

// -----------------------------------------------------------------------------
// Decode de un evento de traza.
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::string decode_event(const TraceEvent& e) {
    using detail::fmt_num;
    std::string head;
    switch (e.op) {
        case Op::Ingest:
            head = "Registró el concepto '" + e.output_label + "'.";
            break;
        case Op::Compose:
            head = "Compuso el rol '" + (e.inputs.size() > 0 ? e.inputs[0] : "?") +
                   "' con el valor '" + (e.inputs.size() > 1 ? e.inputs[1] : "?") + "'.";
            break;
        case Op::BundlePairs:
            head = "Combinó " + std::to_string(e.inputs.size() / 2) +
                   " parejas rol→valor en un único estado.";
            break;
        case Op::Query:
            head = "Consultó por el rol '" +
                   (e.inputs.size() > 0 ? e.inputs[0] : "?") + "'.";
            break;
        case Op::Collapse:
            head = "Colapsó el estado sobre " + std::to_string(e.inputs.size()) +
                   " candidatos.";
            break;
        case Op::Argmax:
            head = "Eligió '" + e.output_label + "'.";
            break;
        case Op::Custom:
            head = "Operación personalizada: " + e.output_label + ".";
            break;
    }
    std::string tail;
    if (!std::isnan(e.fidelity_prev)) {
        tail += " Similitud con el paso anterior: " + fmt_num(e.fidelity_prev, 3) + ".";
    }
    if (!std::isnan(e.fisher_step)) {
        tail += " Salto Fisher: " + fmt_num(e.fisher_step, 3) + ".";
    }
    return head + tail;
}

[[nodiscard]] inline std::string decode_trace(const std::vector<TraceEvent>& events) {
    std::string out;
    for (std::size_t i = 0; i < events.size(); ++i) {
        out += std::to_string(i + 1) + ". " + decode_event(events[i]) + "\n";
    }
    return out;
}

// -----------------------------------------------------------------------------
// Frase compacta unificada (lo que la app pinta en pantalla).
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::string decode_full(
    const Decision& d,
    const Distribution& p,
    const std::vector<std::string>& names,
    std::size_t topk = 3)
{
    std::string out = decode_decision(d);
    if (p.dim() == names.size() && p.dim() > 0) {
        out += " " + decode_topk(p, names, topk);
    }
    return out;
}

}  // namespace easyatom::explain
