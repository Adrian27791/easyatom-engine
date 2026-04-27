// =============================================================================
// easyatom/score/discovery.hpp  --  L36
//
// DiscoveryScore: metrica formal para rankear propuestas de leyes nuevas.
//
//   score(law, codebook) = alpha * novelty + beta * coherence
//                        + gamma * cross_domain
//
// donde:
//   novelty       = 1 - max_i fidelity(law.state, codebook[i].state)
//                   (1 = totalmente nueva, 0 = ya existe igual)
//   coherence     = 1 si evaluate_addition().accepted else 0       (L23)
//   cross_domain  = entropia normalizada de las relaciones (Triplet.relation)
//                   entre los top-k vecinos de law.state, en [0,1].
//                   1 = vecinos diversos (cruza dominios), 0 = todos del mismo
//                   tipo (intra-dominio).
//
// Score en [0, alpha+beta+gamma]; si los pesos suman 1, score en [0,1].
// =============================================================================

#ifndef EASYATOM_SCORE_DISCOVERY_HPP
#define EASYATOM_SCORE_DISCOVERY_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "easyatom/cst/compile.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/reason/coherence.hpp"

namespace easyatom::score {

using easyatom::cst::CompiledLaw;
using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::hilbert::fidelity;
using easyatom::hilbert::State;
using easyatom::reason::CoherenceReport;
using easyatom::reason::evaluate_addition;

struct ScoreWeights {
    double alpha = 1.0 / 3.0;   // novelty
    double beta  = 1.0 / 3.0;   // coherence
    double gamma = 1.0 / 3.0;   // cross_domain
};

struct ScoreConfig {
    std::size_t  k_neighbors    = 5;
    std::size_t  coherence_k    = 4;
    double       coherence_eps  = 0.5;
    ScoreWeights weights;
};

struct ScoreReport {
    double novelty      = 0.0;
    double coherence    = 0.0;   // 0 o 1
    double cross_domain = 0.0;
    double total        = 0.0;
};

namespace detail {

[[nodiscard]] inline double novelty_of(const State&                    s,
                                       const std::vector<CompiledLaw>& cb) {
    if (cb.empty()) return 1.0;
    double max_f = 0.0;
    for (const auto& law : cb) {
        const double f = fidelity(s, law.state);
        if (f > max_f) max_f = f;
    }
    return 1.0 - max_f;
}

[[nodiscard]] inline double cross_domain_of(
    const State&                    s,
    const std::vector<CompiledLaw>& cb,
    std::size_t                     k_neighbors) {
    if (cb.empty() || k_neighbors == 0) return 0.0;
    std::vector<std::pair<double, std::size_t>> rank;
    rank.reserve(cb.size());
    for (std::size_t i = 0; i < cb.size(); ++i)
        rank.emplace_back(fidelity(s, cb[i].state), i);
    const std::size_t k = std::min(k_neighbors, rank.size());
    std::partial_sort(rank.begin(), rank.begin() + k, rank.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;
                      });
    std::unordered_map<int, std::size_t> hist;
    for (std::size_t i = 0; i < k; ++i)
        ++hist[static_cast<int>(cb[rank[i].second].triplet.relation)];
    if (hist.size() <= 1) return 0.0;
    double H = 0.0;
    for (const auto& [_, c] : hist) {
        const double p = static_cast<double>(c) / static_cast<double>(k);
        if (p > 0.0) H -= p * std::log2(p);
    }
    const double Hmax = std::log2(static_cast<double>(hist.size()));
    return Hmax > 0.0 ? H / Hmax : 0.0;
}

}  // namespace detail

[[nodiscard]] inline ScoreReport
discovery_score(const CompiledLaw&              candidate,
                const std::vector<CompiledLaw>& codebook,
                const ScoreConfig&              cfg = {}) {
    if (cfg.weights.alpha < 0.0 || cfg.weights.beta < 0.0 ||
        cfg.weights.gamma < 0.0)
        throw std::invalid_argument("discovery_score: pesos negativos.");

    ScoreReport r;
    r.novelty      = detail::novelty_of(candidate.state, codebook);
    r.cross_domain = detail::cross_domain_of(candidate.state, codebook,
                                             cfg.k_neighbors);

    std::vector<Triplet> ts;
    std::vector<State>   ss;
    ts.reserve(codebook.size());
    ss.reserve(codebook.size());
    for (const auto& law : codebook) {
        ts.push_back(law.triplet);
        ss.push_back(law.state);
    }
    CoherenceReport cr = evaluate_addition(
        ts, ss, candidate.triplet, candidate.state,
        cfg.coherence_k, cfg.coherence_eps);
    r.coherence = cr.accepted ? 1.0 : 0.0;

    r.total = cfg.weights.alpha * r.novelty +
              cfg.weights.beta  * r.coherence +
              cfg.weights.gamma * r.cross_domain;
    return r;
}

[[nodiscard]] inline std::vector<std::size_t>
rank_candidates(const std::vector<CompiledLaw>& candidates,
                const std::vector<CompiledLaw>& codebook,
                const ScoreConfig&              cfg = {}) {
    std::vector<std::pair<double, std::size_t>> rs;
    rs.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double s = discovery_score(candidates[i], codebook, cfg).total;
        rs.emplace_back(s, i);
    }
    std::sort(rs.begin(), rs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::size_t> out;
    out.reserve(rs.size());
    for (const auto& [_, idx] : rs) out.push_back(idx);
    return out;
}

}  // namespace easyatom::score

#endif  // EASYATOM_SCORE_DISCOVERY_HPP
