// =============================================================================
// EasyAtom · Ladrillo 8 — Implementación del C ABI sobre QKernel.
// =============================================================================
//
// Este archivo es la ÚNICA unidad de traducción no header-only del motor.
// Razón: el C ABI necesita símbolos exportables. Todo lo demás sigue siendo
// header-only.
//
// Reglas:
//   * Toda excepción C++ se atrapa y se traduce a un eatom_status_t.
//   * No hay logging/IO desde la biblioteca: el caller decide qué hacer
//     con los códigos de error.

#include "easyatom/c_api.h"
#include "easyatom/qkernel/qkernel.hpp"
#include "easyatom/decide/decisor.hpp"
#include "easyatom/explain/decoder.hpp"

#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

using easyatom::qkernel::QKernel;
using easyatom::hilbert::State;
using easyatom::infogeo::Distribution;
using easyatom::decide::Decision;
using easyatom::decide::DecisionKind;
using easyatom::decide::DecisionPolicy;

struct eatom_kernel {
    QKernel impl;
    eatom_kernel(std::size_t dim, std::uint64_t seed) : impl(dim, seed) {}
};

extern "C" {

size_t eatom_recommend_dim(int tier) {
    switch (tier) {
        case EATOM_TIER_LOW:   return static_cast<size_t>(1) << 14;  //  16384
        case EATOM_TIER_MID:   return static_cast<size_t>(1) << 15;  //  32768
        case EATOM_TIER_HIGH:  return static_cast<size_t>(1) << 16;  //  65536
        case EATOM_TIER_ULTRA: return static_cast<size_t>(1) << 17;  // 131072
        default:               return 0;
    }
}

eatom_kernel_t* eatom_kernel_create(size_t dim, uint64_t seed) {
    if (dim == 0) return nullptr;
    try {
        return new eatom_kernel(dim, seed);
    } catch (...) {
        return nullptr;
    }
}

void eatom_kernel_destroy(eatom_kernel_t* k) {
    delete k;
}

size_t eatom_kernel_dim(const eatom_kernel_t* k) {
    if (!k) return 0;
    return k->impl.dim();
}

size_t eatom_kernel_codebook_size(const eatom_kernel_t* k) {
    if (!k) return 0;
    return k->impl.codebook_size();
}

int eatom_kernel_ingest(eatom_kernel_t* k, const char* name) {
    if (!k || !name) return EATOM_ERR_NULL;
    try {
        (void)k->impl.ingest(std::string(name));
        return EATOM_OK;
    } catch (const std::invalid_argument&) {
        return EATOM_ERR_INVALID_ARG;
    } catch (...) {
        return EATOM_ERR_INTERNAL;
    }
}

namespace {

// Helper: ingiere y devuelve el State, lanzando si autoingest=0 y no existe.
const State& fetch_state(QKernel& k, const char* name, bool autoingest) {
    std::string s(name);
    if (!autoingest && !k.contains(s)) {
        throw std::invalid_argument("nombre no presente: " + s);
    }
    return k.ingest(s);
}

// Construye composite + query_state + lista de candidatos.
// Lanza std::invalid_argument en caso de input inválido.
std::pair<State, std::vector<std::string>> build_query(
    QKernel& k,
    const char* const* roles,        size_t n_pairs,
    const char* const* fillers,
    const char* query_role,
    const char* const* candidates,   size_t n_candidates,
    bool autoingest)
{
    if (n_pairs == 0) {
        throw std::invalid_argument("n_pairs = 0");
    }
    if (n_candidates == 0) {
        throw std::invalid_argument("n_candidates = 0");
    }
    if (!roles || !fillers || !query_role || !candidates) {
        throw std::invalid_argument("null pointer en arrays");
    }
    std::vector<std::pair<State, State>> pairs;
    pairs.reserve(n_pairs);
    for (size_t i = 0; i < n_pairs; ++i) {
        if (!roles[i] || !fillers[i]) {
            throw std::invalid_argument("role o filler nulo");
        }
        const State& r = fetch_state(k, roles[i], autoingest);
        const State& f = fetch_state(k, fillers[i], autoingest);
        pairs.emplace_back(r, f);
    }
    State composite = k.bundle_pairs(pairs);
    const State& qrole = fetch_state(k, query_role, autoingest);
    State guess = k.query(composite, qrole);

    std::vector<std::string> cand_names;
    cand_names.reserve(n_candidates);
    for (size_t i = 0; i < n_candidates; ++i) {
        if (!candidates[i]) throw std::invalid_argument("candidato nulo");
        // Asegura que están en la codebook.
        (void)fetch_state(k, candidates[i], autoingest);
        cand_names.emplace_back(candidates[i]);
    }
    return {std::move(guess), std::move(cand_names)};
}

}  // namespace

int eatom_kernel_query_pairs_argmax(
    eatom_kernel_t* k,
    const char* const* roles, size_t n_pairs,
    const char* const* fillers,
    const char* query_role,
    const char* const* candidates, size_t n_candidates,
    int autoingest,
    size_t* out_winner_index)
{
    if (!k || !out_winner_index) return EATOM_ERR_NULL;
    try {
        auto [guess, cand_names] = build_query(
            k->impl, roles, n_pairs, fillers, query_role,
            candidates, n_candidates, autoingest != 0);
        std::string winner = k->impl.argmax_collapse(guess, cand_names);
        for (size_t i = 0; i < n_candidates; ++i) {
            if (cand_names[i] == winner) {
                *out_winner_index = i;
                return EATOM_OK;
            }
        }
        return EATOM_ERR_INTERNAL;  // no debería pasar
    } catch (const std::invalid_argument&) {
        return EATOM_ERR_INVALID_ARG;
    } catch (...) {
        return EATOM_ERR_INTERNAL;
    }
}

int eatom_kernel_query_pairs_probs(
    eatom_kernel_t* k,
    const char* const* roles, size_t n_pairs,
    const char* const* fillers,
    const char* query_role,
    const char* const* candidates, size_t n_candidates,
    int autoingest,
    double* out_probs)
{
    if (!k || !out_probs) return EATOM_ERR_NULL;
    try {
        auto [guess, cand_names] = build_query(
            k->impl, roles, n_pairs, fillers, query_role,
            candidates, n_candidates, autoingest != 0);
        auto d = k->impl.collapse(guess, cand_names);
        for (size_t i = 0; i < n_candidates; ++i) {
            out_probs[i] = d[i];
        }
        return EATOM_OK;
    } catch (const std::invalid_argument&) {
        return EATOM_ERR_INVALID_ARG;
    } catch (...) {
        return EATOM_ERR_INTERNAL;
    }
}

namespace {

inline int decision_kind_to_c(DecisionKind k) noexcept {
    switch (k) {
        case DecisionKind::Accept:     return EATOM_DECISION_ACCEPT;
        case DecisionKind::Ambiguous:  return EATOM_DECISION_AMBIGUOUS;
        case DecisionKind::Abstain:    return EATOM_DECISION_ABSTAIN;
        case DecisionKind::Degenerate: return EATOM_DECISION_DEGENERATE;
        case DecisionKind::Invalid:    return EATOM_DECISION_INVALID;
    }
    return EATOM_DECISION_INVALID;
}

inline DecisionPolicy policy_from_c(const eatom_policy_t* p) {
    if (!p) return {};
    DecisionPolicy out;
    out.min_confidence       = p->min_confidence;
    out.min_margin           = p->min_margin;
    out.max_entropy_ratio    = p->max_entropy_ratio;
    out.max_effective_n      = p->max_effective_n;
    out.require_finite_probs = (p->require_finite_probs != 0);
    return out;
}

inline void copy_string_safe(const std::string& s, char* buf, size_t cap,
                             size_t* needed)
{
    if (needed) *needed = s.size() + 1;
    if (!buf || cap == 0) return;
    const size_t to_copy = (s.size() + 1 <= cap) ? s.size() : (cap - 1);
    std::memcpy(buf, s.data(), to_copy);
    buf[to_copy] = '\0';
}

}  // namespace

int eatom_kernel_decide_pairs(
    eatom_kernel_t* k,
    const char* const* roles, size_t n_pairs,
    const char* const* fillers,
    const char* query_role,
    const char* const* candidates, size_t n_candidates,
    int autoingest,
    const eatom_policy_t* policy,
    int*    out_kind,
    size_t* out_winner_index,
    size_t* out_runner_up_index,
    double* out_confidence,
    double* out_margin,
    double* out_entropy,
    double* out_entropy_ratio,
    double* out_effective_n,
    double* out_probs,
    char*   out_explanation,
    size_t  out_explanation_capacity,
    size_t* out_explanation_needed)
{
    if (!k) return EATOM_ERR_NULL;
    try {
        auto [guess, cand_names] = build_query(
            k->impl, roles, n_pairs, fillers, query_role,
            candidates, n_candidates, autoingest != 0);
        Distribution dist = k->impl.collapse(guess, cand_names);
        Decision dec = easyatom::decide::decide(
            dist, cand_names, policy_from_c(policy));

        if (out_kind)            *out_kind            = decision_kind_to_c(dec.kind);
        if (out_winner_index)    *out_winner_index    = dec.winner_index;
        if (out_runner_up_index) *out_runner_up_index = dec.runner_up_index;
        if (out_confidence)      *out_confidence      = dec.confidence;
        if (out_margin)          *out_margin          = dec.margin;
        if (out_entropy)         *out_entropy         = dec.entropy;
        if (out_entropy_ratio)   *out_entropy_ratio   = dec.entropy_ratio;
        if (out_effective_n)     *out_effective_n     = dec.effective_n;
        if (out_probs) {
            for (size_t i = 0; i < n_candidates; ++i) out_probs[i] = dist[i];
        }
        std::string text = easyatom::explain::decode_full(dec, dist, cand_names, 3);
        copy_string_safe(text, out_explanation, out_explanation_capacity,
                         out_explanation_needed);
        return EATOM_OK;
    } catch (const std::invalid_argument&) {
        return EATOM_ERR_INVALID_ARG;
    } catch (...) {
        return EATOM_ERR_INTERNAL;
    }
}

}  // extern "C"
