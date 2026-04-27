// =============================================================================
// easyatom/auto/loop.hpp  --  L32
//
// Circulo autonomo de aprendizaje:
//
//   1) find_gaps(queries, codebook, theta)
//        -> queries cuyo density(q, codebook) < theta.
//   2) try_fill(codebook, q, k_top)
//        -> propuesta sintetica (bundle de top-k vecinos).
//   3) evaluate_addition(...)  (L23: contradiccion + beta_1 no aumenta)
//        -> aceptar o rechazar.
//   4) Si aceptada -> codebook.push_back(propuesta)  (la siguiente iteracion
//      ve la nueva ley; aprendizaje incremental).
//   5) Repetir hasta max_iters o hasta no detectar mas gaps.
//
// Sin red, sin alucinacion: solo recombinacion local validada formalmente.
// =============================================================================

#ifndef EASYATOM_AUTO_LOOP_HPP
#define EASYATOM_AUTO_LOOP_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "easyatom/cst/compile.hpp"
#include "easyatom/epistemic/gap.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/reason/coherence.hpp"

namespace easyatom::autoloop {

using easyatom::cst::CompiledLaw;
using easyatom::cst::Triplet;
using easyatom::epistemic::find_gaps;
using easyatom::epistemic::try_fill;
using easyatom::hilbert::State;
using easyatom::reason::CoherenceReport;
using easyatom::reason::evaluate_addition;

struct LoopConfig {
    double      gap_theta         = 0.3;   // umbral de density para gap
    std::size_t k_top             = 3;     // vecinos a fundir en try_fill
    std::size_t max_iters         = 8;     // tope de iteraciones
    std::size_t coherence_k       = 4;     // k vecinos para Betti1
    double      coherence_eps     = 0.5;   // umbral Vietoris-Rips
};

struct LoopReport {
    std::size_t iters              = 0;
    std::size_t gaps_detected      = 0;    // suma sobre todas las iteraciones
    std::size_t proposals          = 0;    // try_fill que devolvieron algo
    std::size_t accepted           = 0;    // pasaron coherence
    std::size_t rejected_coherence = 0;    // contradiccion o beta1 subio
};

// run_auto_loop muta `codebook` agregando solo leyes aceptadas formalmente.
inline LoopReport run_auto_loop(std::vector<CompiledLaw>& codebook,
                                const std::vector<State>& queries,
                                const LoopConfig&         cfg = {}) {
    if (cfg.gap_theta < 0.0)
        throw std::invalid_argument("run_auto_loop: gap_theta < 0.");
    LoopReport rep{};

    for (std::size_t it = 0; it < cfg.max_iters; ++it) {
        ++rep.iters;
        auto gaps = find_gaps(queries, codebook, cfg.gap_theta);
        if (gaps.empty()) break;
        rep.gaps_detected += gaps.size();

        bool any_accepted_this_iter = false;
        for (const auto& g : gaps) {
            auto cand = try_fill(codebook, queries[g.query_index], cfg.k_top);
            if (!cand) continue;
            ++rep.proposals;

            // Snapshot del cuerpo aceptado para evaluar la adicion.
            std::vector<Triplet> ts;
            std::vector<State>   ss;
            ts.reserve(codebook.size());
            ss.reserve(codebook.size());
            for (const auto& law : codebook) {
                ts.push_back(law.triplet);
                ss.push_back(law.state);
            }

            CoherenceReport cr = evaluate_addition(
                ts, ss, cand->triplet, cand->state,
                cfg.coherence_k, cfg.coherence_eps);

            if (cr.accepted) {
                codebook.push_back(*cand);
                ++rep.accepted;
                any_accepted_this_iter = true;
            } else {
                ++rep.rejected_coherence;
            }
        }

        // Si esta iteracion no acepto nada nuevo, no hay forma de progresar.
        if (!any_accepted_this_iter) break;
    }

    return rep;
}

}  // namespace easyatom::autoloop

#endif  // EASYATOM_AUTO_LOOP_HPP
