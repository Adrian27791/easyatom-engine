// =============================================================================
// EasyAtom · Ladrillo 7 — Q-Kernel: API pública (ingest / compose / collapse).
// =============================================================================
//
// Esta es la fachada que une los ladrillos previos en un motor utilizable:
//
//   Codebook            : diccionario nombrado de estados |a⟩ ∈ H_D, donde
//                         cada nombre se mapea de forma determinista a un
//                         estado de fase aleatoria (random_phase_state) usando
//                         hash(nombre) como semilla.
//
//   ingest(name)        : devuelve el estado asociado a `name`, creándolo
//                         si no existe. Idempotente y reproducible:
//                         dos llamadas con el mismo nombre devuelven el mismo
//                         estado (y dos kernels con la misma seed master
//                         devuelven los mismos estados para los mismos nombres).
//
//   compose(role,filler): bind(role, filler) — vincula un rol con su valor.
//
//   bundle_pairs(pairs) : superpone varios pares role:filler en un único
//                         estado vía bundle de los binds.
//
//   query(state, role)  : unbind(state, role) — recupera (aproximadamente)
//                         el filler que se vinculó al rol.
//
//   collapse(state, names): proyecta `state` sobre el subconjunto de la
//                         codebook etiquetado por `names`, calcula la
//                         FIDELIDAD con cada uno, y devuelve una
//                         Distribution (Ladrillo 3) sobre esos nombres.
//                         Es la "lectura cuántica": de un superpuesto a una
//                         distribución de probabilidad calibrada.
//
// CONSTRUCCIÓN MATEMÁTICA:
//   * Los estados base son vectores de fase aleatoria → casi-ortogonales en
//     dimensiones grandes (<φ_i|φ_j> → 0 con D → ∞), lo que garantiza que
//     bind/unbind sean ruidosos pero recuperables.
//   * collapse usa fidelidad F = |⟨ψ|φ⟩|² (Ladrillo 1) como "probabilidad
//     de proyección" — es exactamente la regla de Born en mecánica cuántica.
//   * La distribución resultante vive en el símplex y es procesable con
//     Fisher-Rao / α-divergencias (Ladrillo 3).
//
// REGLAS:
//   * Hash determinista propio (FNV-1a 64-bit) → no dependemos de la
//     implementación de std::hash<string> (que varía entre stdlibs).
//   * Sin estado global: cada QKernel es autónomo.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "easyatom/hilbert/state.hpp"
#include "easyatom/infogeo/fisher.hpp"
#include "easyatom/ops/fundamental.hpp"

namespace easyatom::qkernel {

using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;
using easyatom::infogeo::Distribution;

// -----------------------------------------------------------------------------
// Hash determinista FNV-1a 64-bit. Estable entre plataformas.
// -----------------------------------------------------------------------------
namespace detail {

[[nodiscard]] inline std::uint64_t fnv1a_64(const std::string& s) noexcept {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

}  // namespace detail

// -----------------------------------------------------------------------------
// QKernel — fachada del motor.
// -----------------------------------------------------------------------------
class QKernel {
public:
    /// Construye un kernel con dimensión D y semilla maestra.
    QKernel(std::size_t dim, std::uint64_t master_seed)
        : dim_(dim), master_seed_(master_seed) {
        if (dim == 0) {
            throw std::invalid_argument("QKernel: dim = 0.");
        }
    }

    [[nodiscard]] std::size_t dim() const noexcept { return dim_; }

    /// Devuelve (creando si hace falta) el estado base asociado a `name`.
    /// Idempotente: misma string → mismo estado.
    [[nodiscard]] const State& ingest(const std::string& name) {
        auto it = codebook_.find(name);
        if (it != codebook_.end()) return it->second;
        const std::uint64_t seed = master_seed_ ^ detail::fnv1a_64(name);
        State s = easyatom::ops::random_phase_state(dim_, seed);
        auto [ins, _] = codebook_.emplace(name, std::move(s));
        return ins->second;
    }

    [[nodiscard]] bool contains(const std::string& name) const {
        return codebook_.find(name) != codebook_.end();
    }

    [[nodiscard]] std::size_t codebook_size() const noexcept {
        return codebook_.size();
    }

    /// bind(role, filler).
    [[nodiscard]] State compose(const State& role, const State& filler) const {
        return easyatom::ops::bind(role, filler);
    }

    /// Bundle de binds: Σ bind(role_i, filler_i).
    [[nodiscard]] State bundle_pairs(
        const std::vector<std::pair<State, State>>& pairs) const {
        if (pairs.empty()) {
            throw std::invalid_argument("bundle_pairs: vacío.");
        }
        std::vector<State> bound;
        bound.reserve(pairs.size());
        for (const auto& [r, f] : pairs) {
            bound.push_back(easyatom::ops::bind(r, f));
        }
        return easyatom::ops::bundle(bound);
    }

    /// Recupera (aproximadamente) el filler vinculado a `role`.
    [[nodiscard]] State query(const State& composite, const State& role) const {
        return easyatom::ops::unbind(composite, role);
    }

    /// Colapsa `state` sobre el sub-codebook etiquetado por `names`.
    /// Devuelve una Distribution sobre `names` (mismo orden) con
    /// p_i = F(state, codebook[names[i]]) / Σ_j F(state, codebook[names[j]]).
    [[nodiscard]] Distribution collapse(
        const State& state, const std::vector<std::string>& names) const {
        if (names.empty()) {
            throw std::invalid_argument("collapse: sin candidatos.");
        }
        std::vector<double> scores;
        scores.reserve(names.size());
        for (const auto& n : names) {
            auto it = codebook_.find(n);
            if (it == codebook_.end()) {
                throw std::invalid_argument(
                    "collapse: nombre no presente en codebook: '" + n + "'.");
            }
            scores.push_back(fidelity(state, it->second));
        }
        return Distribution::from_scores(scores);
    }

    /// Devuelve el nombre del candidato con mayor fidelidad.
    /// Empate: gana el primero por orden de inserción en `names`.
    [[nodiscard]] std::string argmax_collapse(
        const State& state, const std::vector<std::string>& names) const {
        if (names.empty()) {
            throw std::invalid_argument("argmax_collapse: sin candidatos.");
        }
        const Distribution d = collapse(state, names);
        std::size_t best = 0;
        double best_p = d[0];
        for (std::size_t i = 1; i < names.size(); ++i) {
            if (d[i] > best_p) { best_p = d[i]; best = i; }
        }
        return names[best];
    }

private:
    std::size_t dim_;
    std::uint64_t master_seed_;
    std::unordered_map<std::string, State> codebook_;
};

}  // namespace easyatom::qkernel
