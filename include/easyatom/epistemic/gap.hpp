// =============================================================================
// easyatom/epistemic/gap.hpp  --  L24
//
// Detecta lagunas (gaps) en el conocimiento del kernel y propone leyes nuevas
// por recombinacion local (sin RAG, sin red, sin texto externo).
//
// Idea fundamental:
//
//   Dado un codebook de leyes ya compiladas (vector<CompiledLaw>) y una nube
//   de "consultas" expresadas como States, decimos que una consulta esta en
//   un "hueco" si su densidad de soporte es baja:
//
//     density(q, codebook) = mean_i fidelity(q, codebook[i].state)
//
//   Si density(q) < theta -> hay un gap.
//
// try_fill construye una propuesta sintetica para llenar ese hueco usando
// solo lo que ya esta en el kernel: toma los top-k vecinos por fidelity con
// la query y produce una CompiledLaw cuya state es el bundle (superposicion
// normalizada por norma) de esos vecinos. La tripleta sintetica conserva
// la relacion del vecino mas afin y usa etiquetas sintetizadas con prefijo
// "gap__" para marcar provenance.
//
// La ley propuesta NO se inserta automaticamente: debe pasar por la capa
// de coherencia (L23) antes de ser aceptada. Esto cierra el lazo
// detectar -> proponer -> validar -> aceptar/rechazar sin alucinacion.
//
// Header-only, C++20 puro.
// =============================================================================

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "easyatom/cst/compile.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"

namespace easyatom::epistemic {

using easyatom::cst::CompiledLaw;
using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;

// -----------------------------------------------------------------------------
// Densidad de soporte
// -----------------------------------------------------------------------------

[[nodiscard]] inline double density(const State& q,
                                    const std::vector<CompiledLaw>& codebook) {
    if (codebook.empty()) return 0.0;
    double acc = 0.0;
    for (const auto& law : codebook)
        acc += fidelity(q, law.state);
    return acc / static_cast<double>(codebook.size());
}

// -----------------------------------------------------------------------------
// Deteccion de huecos
// -----------------------------------------------------------------------------

struct Gap {
    std::size_t query_index;   // indice en queries
    double      density;       // densidad observada (< theta)
};

[[nodiscard]] inline std::vector<Gap>
find_gaps(const std::vector<State>& queries,
          const std::vector<CompiledLaw>& codebook,
          double theta) {
    if (theta < 0.0)
        throw std::invalid_argument("find_gaps: theta < 0.");
    std::vector<Gap> out;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const double d = density(queries[i], codebook);
        if (d < theta) out.push_back({i, d});
    }
    return out;
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

namespace detail {

[[nodiscard]] inline std::uint64_t fnv1a64_str(std::string_view s) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

[[nodiscard]] inline std::string hex16(std::uint64_t v) {
    static const char* dig = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) { out[i] = dig[v & 0xF]; v >>= 4; }
    return out;
}

} // namespace detail

// -----------------------------------------------------------------------------
// Propuesta sintetica para llenar un hueco
// -----------------------------------------------------------------------------
//
// codebook    : leyes ya compiladas (vecindario potencial).
// query       : state que define el hueco.
// k_top       : cuantos vecinos mas afines combinar (>=1).
// Retorna nullopt si codebook esta vacio.

[[nodiscard]] inline std::optional<CompiledLaw>
try_fill(const std::vector<CompiledLaw>& codebook,
         const State& query,
         std::size_t  k_top = 3) {
    if (codebook.empty()) return std::nullopt;
    if (k_top == 0) k_top = 1;

    // 1) ranking por fidelity con la query.
    std::vector<std::pair<double, std::size_t>> rank;
    rank.reserve(codebook.size());
    for (std::size_t i = 0; i < codebook.size(); ++i)
        rank.emplace_back(fidelity(query, codebook[i].state), i);
    std::sort(rank.begin(), rank.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    const std::size_t k = std::min(k_top, rank.size());

    // 2) bundle de los top-k states (operacion fundamental de la capa A).
    std::vector<State> top;
    top.reserve(k);
    for (std::size_t i = 0; i < k; ++i)
        top.push_back(codebook[rank[i].second].state);
    State acc = easyatom::ops::bundle(top);

    // 3) Tripleta sintetica: hereda relacion del vecino mas afin, sujetos
    //    y objetos se marcan con prefijo "gap__" + hash de la query.
    const auto& best_t = codebook[rank[0].second].triplet;
    const std::string tag = "gap__" + detail::hex16(
        detail::fnv1a64_str(best_t.subject + "|" + best_t.object));
    Triplet t_syn{tag + "__s", best_t.relation, tag + "__o"};

    // 4) Empaquetado.
    CompiledLaw out;
    out.triplet         = std::move(t_syn);
    out.state           = std::move(acc);
    out.fingerprint     = detail::fnv1a64_str(out.triplet.subject + "|" +
                                              out.triplet.object);
    out.provenance_hash = 0;  // sintetico; no proviene de texto.
    return out;
}

} // namespace easyatom::epistemic
