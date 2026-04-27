// =============================================================================
// easyatom/auto/energy.hpp  --  L37
//
// Energia compuesta E_total para rankear adiciones candidatas al codebook.
// Inspirada en el principio de minima descripcion (MDL): preferimos leyes
// que (a) reducen la entropia/redundancia del codebook y (b) no anaden
// contradicciones ni complican la topologia simbolica.
//
//   E_repr(law)   = 1 - density(law.state, codebook)        (en [0,1])
//                   "cuanto cuesta REPRESENTAR la ley dado lo que ya hay"
//                   alta densidad => ya esta soportada => bajo costo
//
//   E_const(law)  = +inf si has_contradiction
//                   = max(0, beta1_after - beta1_before)    (penalizacion
//                     suave por ciclos topologicos nuevos)
//
//   E_total(law)  = E_repr + lambda * E_const
//
// Menor E_total = mejor candidata. rank ascending.
// =============================================================================

#ifndef EASYATOM_AUTO_ENERGY_HPP
#define EASYATOM_AUTO_ENERGY_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "easyatom/cst/compile.hpp"
#include "easyatom/epistemic/gap.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/reason/coherence.hpp"

namespace easyatom::autoloop {

using easyatom::cst::CompiledLaw;
using easyatom::cst::Triplet;
using easyatom::epistemic::density;
using easyatom::hilbert::State;
using easyatom::reason::CoherenceReport;
using easyatom::reason::evaluate_addition;

struct EnergyConfig {
    double      lambda        = 1.0;
    std::size_t coherence_k   = 4;
    double      coherence_eps = 0.5;
};

struct EnergyReport {
    double e_repr  = 0.0;
    double e_const = 0.0;
    double e_total = 0.0;
};

[[nodiscard]] inline EnergyReport
compute_energy(const CompiledLaw&              candidate,
               const std::vector<CompiledLaw>& codebook,
               const EnergyConfig&             cfg = {}) {
    if (cfg.lambda < 0.0)
        throw std::invalid_argument("compute_energy: lambda < 0.");

    EnergyReport r;
    r.e_repr = 1.0 - density(candidate.state, codebook);
    if (r.e_repr < 0.0) r.e_repr = 0.0;

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

    if (cr.has_contradiction) {
        r.e_const = std::numeric_limits<double>::infinity();
        r.e_total = std::numeric_limits<double>::infinity();
        return r;
    }
    const long long delta =
        static_cast<long long>(cr.beta1_after) -
        static_cast<long long>(cr.beta1_before);
    r.e_const = delta > 0 ? static_cast<double>(delta) : 0.0;
    r.e_total = r.e_repr + cfg.lambda * r.e_const;
    return r;
}

[[nodiscard]] inline std::vector<std::size_t>
rank_by_energy(const std::vector<CompiledLaw>& candidates,
               const std::vector<CompiledLaw>& codebook,
               const EnergyConfig&             cfg = {}) {
    std::vector<std::pair<double, std::size_t>> rs;
    rs.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double e = compute_energy(candidates[i], codebook, cfg).e_total;
        rs.emplace_back(e, i);
    }
    std::sort(rs.begin(), rs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::size_t> out;
    out.reserve(rs.size());
    for (const auto& [_, idx] : rs) out.push_back(idx);
    return out;
}

}  // namespace easyatom::autoloop

#endif  // EASYATOM_AUTO_ENERGY_HPP
