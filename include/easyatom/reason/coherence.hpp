// =============================================================================
// easyatom/reason/coherence.hpp  --  L23
//
// Capa de coherencia logica y auto-evaluacion. NO escribe en el kernel; solo
// inspecciona conjuntos de tripletas y nubes de estados ya existentes.
//
// Dos mecanismos complementarios:
//
//  1) Contradicciones explicitas a nivel simbolico:
//       - Causes(X,Y)    + Inhibits(X,Y)
//       - Increases(X,Y) + Decreases(X,Y)
//       - IsA(X,Y) + IsA(Y,X)  con  X != Y   (ciclo de identidad)
//
//  2) Validacion topologica via beta_1 (L19):
//       Proyectamos cada State a R^k tomando los primeros k coeficientes
//       reales y calculamos beta_1 con Vietoris-Rips. Si anadir una ley
//       nueva incrementa beta_1 (aparece un agujero), se considera que
//       introduce inconsistencia geometrica -> se rechaza.
//
// Header-only, C++20 puro, sin RAG, sin red, sin estado global.
// =============================================================================

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/topo/persistent.hpp"

namespace easyatom::reason {

using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::hilbert::State;

// -----------------------------------------------------------------------------
// Contradicciones explicitas
// -----------------------------------------------------------------------------

enum class ContradictionKind {
    CausesVsInhibits,
    IncreasesVsDecreases,
    IsACycle,
};

struct Contradiction {
    std::size_t        i;       // indice de la primera tripleta
    std::size_t        j;       // indice de la segunda
    ContradictionKind  kind;
};

namespace detail {

[[nodiscard]] inline bool same_so(const Triplet& a, const Triplet& b) {
    return a.subject == b.subject && a.object == b.object;
}

[[nodiscard]] inline bool reversed_so(const Triplet& a, const Triplet& b) {
    return a.subject == b.object && a.object == b.subject &&
           a.subject != a.object;
}

[[nodiscard]] inline bool pair_opposes(Relation r1, Relation r2,
                                       ContradictionKind& out) {
    if ((r1 == Relation::Causes    && r2 == Relation::Inhibits) ||
        (r1 == Relation::Inhibits  && r2 == Relation::Causes)) {
        out = ContradictionKind::CausesVsInhibits;
        return true;
    }
    if ((r1 == Relation::Increases && r2 == Relation::Decreases) ||
        (r1 == Relation::Decreases && r2 == Relation::Increases)) {
        out = ContradictionKind::IncreasesVsDecreases;
        return true;
    }
    return false;
}

} // namespace detail

[[nodiscard]] inline std::vector<Contradiction>
contradictions(const std::vector<Triplet>& ts) {
    std::vector<Contradiction> out;
    for (std::size_t i = 0; i < ts.size(); ++i) {
        if (!ts[i].valid()) continue;
        for (std::size_t j = i + 1; j < ts.size(); ++j) {
            if (!ts[j].valid()) continue;
            const Triplet& a = ts[i];
            const Triplet& b = ts[j];
            // Mismo (S,O) con relaciones opuestas.
            if (detail::same_so(a, b)) {
                ContradictionKind k;
                if (detail::pair_opposes(a.relation, b.relation, k))
                    out.push_back({i, j, k});
            }
            // IsA(X,Y) + IsA(Y,X) con X != Y -> ciclo de identidad.
            if (a.relation == Relation::IsA && b.relation == Relation::IsA &&
                detail::reversed_so(a, b))
                out.push_back({i, j, ContradictionKind::IsACycle});
        }
    }
    return out;
}

[[nodiscard]] inline bool is_coherent(const std::vector<Triplet>& ts) {
    return contradictions(ts).empty();
}

// -----------------------------------------------------------------------------
// Validacion topologica via beta_1 (L19)
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::vector<std::vector<double>>
project_states(const std::vector<State>& states, std::size_t k) {
    if (k == 0) throw std::invalid_argument("project_states: k=0.");
    std::vector<std::vector<double>> pts;
    pts.reserve(states.size());
    for (const State& s : states) {
        if (s.dim() < k)
            throw std::invalid_argument("project_states: dim<k.");
        std::vector<double> p(k);
        const auto& amp = s.amplitudes();
        for (std::size_t i = 0; i < k; ++i)
            p[i] = amp[i].real();
        pts.push_back(std::move(p));
    }
    return pts;
}

[[nodiscard]] inline std::size_t
betti1_of_states(const std::vector<State>& states, std::size_t k, double eps) {
    if (states.empty()) return 0;
    auto pts = project_states(states, k);
    auto r = easyatom::topo::vietoris_rips_betti(pts, eps);
    return r.beta_1;
}

// -----------------------------------------------------------------------------
// Reporte de evaluacion para una adicion candidata
// -----------------------------------------------------------------------------

struct CoherenceReport {
    bool                       has_contradiction;  // contradiccion simbolica
    std::size_t                beta1_before;       // beta_1 antes de anadir
    std::size_t                beta1_after;        // beta_1 despues de anadir
    bool                       accepted;           // veredicto final
};

// existing_triplets/states deben tener el mismo tamano y representar el cuerpo
// de leyes ya aceptado. La nueva ley se evalua sin mutar nada.
[[nodiscard]] inline CoherenceReport
evaluate_addition(const std::vector<Triplet>& existing_triplets,
                  const std::vector<State>&   existing_states,
                  const Triplet&              new_triplet,
                  const State&                new_state,
                  std::size_t                 k,
                  double                      eps) {
    if (existing_triplets.size() != existing_states.size())
        throw std::invalid_argument("evaluate_addition: tamanos distintos.");

    CoherenceReport rep{};

    // 1) Contradiccion simbolica con cualquiera ya aceptada.
    std::vector<Triplet> probe = existing_triplets;
    probe.push_back(new_triplet);
    auto cs = contradictions(probe);
    rep.has_contradiction = false;
    for (const auto& c : cs)
        if (c.j == probe.size() - 1) { rep.has_contradiction = true; break; }

    // 2) Topologia: beta_1 antes y despues.
    rep.beta1_before = betti1_of_states(existing_states, k, eps);
    std::vector<State> after = existing_states;
    after.push_back(new_state);
    rep.beta1_after = betti1_of_states(after, k, eps);

    rep.accepted = !rep.has_contradiction &&
                   rep.beta1_after <= rep.beta1_before;
    return rep;
}

} // namespace easyatom::reason
