// =============================================================================
// easyatom/auto/loop_ranked.hpp  --  L40
//
// Auto-loop con ranking formal de candidatas (extiende L32).
//
// Para cada gap detectado:
//   1) Genera N candidatas variando k_top en [k_min, k_max] (try_fill).
//   2) Las rankea por:
//        - Mode::Energy    -> menor E_total (L37) primero  (MDL ascendente).
//        - Mode::Discovery -> mayor discovery_score (L36) primero
//                             (alpha*novelty + beta*coherence + gamma*cross).
//   3) Recorre el ranking best-first y ACEPTA la primera candidata que pasa
//      evaluate_addition (L23).  El resto se descartan en esa iteracion del
//      gap; en la siguiente vuelta del loop el codebook ya contiene la nueva
//      ley y los gaps se reevaluan.
//
// Beneficio sobre L32 puro:
//   - L32 acepta la PRIMERA candidata que cumple coherencia, sea cual sea su
//     novedad o su costo descriptivo.
//   - L40 elige la MEJOR candidata segun un criterio formal antes de gastar
//     un slot en el codebook.
//
// Sin red, sin alucinacion: 100% recombinacion local + validacion formal.
// =============================================================================

#ifndef EASYATOM_AUTO_LOOP_RANKED_HPP
#define EASYATOM_AUTO_LOOP_RANKED_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "easyatom/auto/energy.hpp"
#include "easyatom/auto/loop.hpp"  // LoopReport, find_gaps, try_fill, ...
#include "easyatom/cst/compile.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/reason/coherence.hpp"
#include "easyatom/score/discovery.hpp"

namespace easyatom::autoloop {

using easyatom::cst::CompiledLaw;
using easyatom::cst::Triplet;
using easyatom::epistemic::find_gaps;
using easyatom::epistemic::try_fill;
using easyatom::hilbert::State;
using easyatom::reason::CoherenceReport;
using easyatom::reason::evaluate_addition;
using easyatom::score::discovery_score;
using easyatom::score::ScoreConfig;

enum class RankMode {
    Energy    = 0,  // L37: menor E_total = mejor
    Discovery = 1   // L36: mayor score    = mejor
};

struct LoopRankedConfig {
    double      gap_theta     = 0.3;
    std::size_t k_min         = 2;
    std::size_t k_max         = 4;
    std::size_t max_iters     = 8;
    std::size_t coherence_k   = 4;
    double      coherence_eps = 0.5;
    RankMode    mode          = RankMode::Energy;
    EnergyConfig energy;        // usado si mode == Energy
    ScoreConfig  discovery;     // usado si mode == Discovery
};

struct LoopRankedReport {
    std::size_t iters              = 0;
    std::size_t gaps_detected      = 0;
    std::size_t candidates_total   = 0;  // suma de propuestas generadas
    std::size_t candidates_unique  = 0;  // tras dedupe por fingerprint
    std::size_t accepted           = 0;
    std::size_t rejected_coherence = 0;
};

namespace detail_ranked {

inline std::vector<CompiledLaw>
gen_candidates(const std::vector<CompiledLaw>& codebook,
               const State&                    query,
               std::size_t                     k_min,
               std::size_t                     k_max) {
    std::vector<CompiledLaw> out;
    if (codebook.empty()) return out;
    if (k_min == 0) k_min = 1;
    if (k_max < k_min) k_max = k_min;
    out.reserve(k_max - k_min + 1);
    for (std::size_t k = k_min; k <= k_max; ++k) {
        auto cand = try_fill(codebook, query, k);
        if (cand) out.push_back(*std::move(cand));
    }
    // Dedupe por fingerprint (try_fill con k distintos puede colisionar).
    std::sort(out.begin(), out.end(),
              [](const CompiledLaw& a, const CompiledLaw& b) {
                  return a.fingerprint < b.fingerprint;
              });
    out.erase(std::unique(out.begin(), out.end(),
                          [](const CompiledLaw& a, const CompiledLaw& b) {
                              return a.fingerprint == b.fingerprint;
                          }),
              out.end());
    return out;
}

}  // namespace detail_ranked

inline LoopRankedReport
run_auto_loop_ranked(std::vector<CompiledLaw>& codebook,
                     const std::vector<State>& queries,
                     const LoopRankedConfig&   cfg = {}) {
    if (cfg.gap_theta < 0.0)
        throw std::invalid_argument("run_auto_loop_ranked: gap_theta < 0.");
    if (cfg.k_min == 0)
        throw std::invalid_argument("run_auto_loop_ranked: k_min == 0.");
    if (cfg.k_max < cfg.k_min)
        throw std::invalid_argument("run_auto_loop_ranked: k_max < k_min.");

    LoopRankedReport rep{};

    for (std::size_t it = 0; it < cfg.max_iters; ++it) {
        ++rep.iters;
        auto gaps = find_gaps(queries, codebook, cfg.gap_theta);
        if (gaps.empty()) break;
        rep.gaps_detected += gaps.size();

        bool any_accepted = false;
        for (const auto& g : gaps) {
            // Numero de variaciones de k_top intentadas (antes de dedupe).
            const std::size_t tried = (cfg.k_max >= cfg.k_min)
                                          ? (cfg.k_max - cfg.k_min + 1)
                                          : 0;
            rep.candidates_total += tried;

            auto cands = detail_ranked::gen_candidates(
                codebook, queries[g.query_index], cfg.k_min, cfg.k_max);
            rep.candidates_unique += cands.size();
            if (cands.empty()) continue;

            // Ranking best-first segun mode.
            std::vector<std::size_t> order;
            order.reserve(cands.size());
            if (cfg.mode == RankMode::Energy) {
                std::vector<std::pair<double, std::size_t>> rs;
                rs.reserve(cands.size());
                for (std::size_t i = 0; i < cands.size(); ++i) {
                    const double e = compute_energy(cands[i], codebook,
                                                    cfg.energy).e_total;
                    rs.emplace_back(e, i);
                }
                std::sort(rs.begin(), rs.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });
                for (const auto& [_, i] : rs) order.push_back(i);
            } else {
                std::vector<std::pair<double, std::size_t>> rs;
                rs.reserve(cands.size());
                for (std::size_t i = 0; i < cands.size(); ++i) {
                    const double s =
                        discovery_score(cands[i], codebook, cfg.discovery).total;
                    rs.emplace_back(s, i);
                }
                std::sort(rs.begin(), rs.end(),
                          [](const auto& a, const auto& b) {
                              return a.first > b.first;
                          });
                for (const auto& [_, i] : rs) order.push_back(i);
            }

            // Snapshot del codebook para evaluate_addition (no muta hasta aceptar).
            std::vector<Triplet> ts;
            std::vector<State>   ss;
            ts.reserve(codebook.size());
            ss.reserve(codebook.size());
            for (const auto& law : codebook) {
                ts.push_back(law.triplet);
                ss.push_back(law.state);
            }

            bool accepted_this_gap = false;
            for (std::size_t idx : order) {
                CoherenceReport cr = evaluate_addition(
                    ts, ss, cands[idx].triplet, cands[idx].state,
                    cfg.coherence_k, cfg.coherence_eps);
                if (cr.accepted) {
                    codebook.push_back(cands[idx]);
                    ++rep.accepted;
                    accepted_this_gap = true;
                    any_accepted      = true;
                    break;  // best-first: nos quedamos con la mejor que pasa.
                } else {
                    ++rep.rejected_coherence;
                }
            }
            (void)accepted_this_gap;
        }

        if (!any_accepted) break;
    }

    return rep;
}

}  // namespace easyatom::autoloop

#endif  // EASYATOM_AUTO_LOOP_RANKED_HPP
