// =============================================================================
// easyatom/dynamics/multiprobe.hpp  --  L38
//
// Colapso multi-probe: lanzamos N copias perturbadas de la misma query y
// medimos cuanto coinciden los atractores. Si todos convergen al mismo
// punto fijo, la respuesta es estable; si se dispersan, la query cae en
// una zona ambigua del paisaje energetico (frontera entre cuencas).
//
//   1) genera N perturbaciones de q (ruido gaussiano determinista por seed)
//   2) corre Hopfield moderno (L35) sobre cada una
//   3) mide acuerdo = min_{i,j} fidelity(xi_i, xi_j)
//   4) consensus = bundle normalizado de los xi_i
//   5) stable = acuerdo >= agreement_tol
//
// 0 alucinaciones: si stable=false, el cliente sabe que la respuesta NO se
// puede dar con certeza y debe escalar (gap o ingest externo, L26+L33).
// =============================================================================

#ifndef EASYATOM_DYNAMICS_MULTIPROBE_HPP
#define EASYATOM_DYNAMICS_MULTIPROBE_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "easyatom/dynamics/modern_hopfield.hpp"
#include "easyatom/hilbert/state.hpp"

namespace easyatom::dynamics::multiprobe {

using easyatom::dynamics::modern::HopfieldConfig;
using easyatom::dynamics::modern::run;
using easyatom::hilbert::Complex;
using easyatom::hilbert::fidelity;
using easyatom::hilbert::State;

struct MultiProbeConfig {
    std::size_t    n_probes        = 8;
    double         perturbation    = 0.05;   // sigma del ruido por componente
    double         agreement_tol   = 0.95;   // fidelity minima entre probes
    std::uint64_t  seed            = 42;
    HopfieldConfig hopfield;
};

struct MultiProbeReport {
    State                consensus;
    double               agreement = 0.0;     // min fidelity entre probes
    bool                 stable    = false;
    std::vector<State>   probes;              // atractores individuales
};

namespace detail {

[[nodiscard]] inline State perturb(const State&         q,
                                   double               sigma,
                                   std::mt19937_64&     rng) {
    std::normal_distribution<double> nd(0.0, sigma);
    const std::size_t D = q.dim();
    std::vector<Complex> v(D);
    for (std::size_t i = 0; i < D; ++i) {
        v[i] = q[i] + Complex{nd(rng), nd(rng)};
    }
    return State(std::move(v));
}

[[nodiscard]] inline State bundle_norm(const std::vector<State>& xs) {
    if (xs.empty()) throw std::invalid_argument("bundle_norm: vacio.");
    const std::size_t D = xs.front().dim();
    std::vector<Complex> acc(D, Complex{0.0, 0.0});
    for (const auto& s : xs)
        for (std::size_t i = 0; i < D; ++i) acc[i] += s[i];
    double n2 = 0.0;
    for (const auto& c : acc) n2 += std::norm(c);
    if (n2 > 0.0) {
        const double inv = 1.0 / std::sqrt(n2);
        for (auto& c : acc) c *= inv;
    }
    return State(std::move(acc));
}

}  // namespace detail

[[nodiscard]] inline MultiProbeReport
multi_probe_collapse(const State&              query,
                     const std::vector<State>& memories,
                     const MultiProbeConfig&   cfg = {}) {
    if (memories.empty())
        throw std::invalid_argument("multi_probe_collapse: memorias vacias.");
    if (cfg.n_probes == 0)
        throw std::invalid_argument("multi_probe_collapse: n_probes == 0.");
    if (cfg.perturbation < 0.0)
        throw std::invalid_argument("multi_probe_collapse: perturbation < 0.");

    std::mt19937_64 rng(cfg.seed);
    MultiProbeReport rep;
    rep.probes.reserve(cfg.n_probes);
    for (std::size_t i = 0; i < cfg.n_probes; ++i) {
        State q_i = (cfg.perturbation == 0.0)
                        ? query
                        : detail::perturb(query, cfg.perturbation, rng);
        auto r = run(q_i, memories, cfg.hopfield);
        rep.probes.push_back(std::move(r.xi));
    }
    rep.consensus = detail::bundle_norm(rep.probes);

    double min_f = 1.0;
    for (std::size_t i = 0; i < rep.probes.size(); ++i)
        for (std::size_t j = i + 1; j < rep.probes.size(); ++j) {
            const double f = fidelity(rep.probes[i], rep.probes[j]);
            if (f < min_f) min_f = f;
        }
    rep.agreement = (rep.probes.size() < 2) ? 1.0 : min_f;
    rep.stable    = rep.agreement >= cfg.agreement_tol;
    return rep;
}

}  // namespace easyatom::dynamics::multiprobe

#endif  // EASYATOM_DYNAMICS_MULTIPROBE_HPP
