// =============================================================================
// easyatom/dynamics/modern_hopfield.hpp  --  L35
//
// Hopfield moderno (Krotov-Hopfield 2016 / Ramsauer 2020 "Hopfield is all
// you need"). Sobre nuestro espacio H_D complejo:
//
//   E(xi)        = -(1/beta) * log Sum_mu exp(beta * Re<x_mu, xi>)
//   update(xi)   = Sum_mu softmax(beta * Re<x_mu, xi>)_mu * x_mu
//
// Capacidad de almacenamiento exponencial en D (vs lineal del Hopfield
// clasico) y convergencia en una sola actualizacion sincrona si beta es
// suficientemente grande (atractor unico = memoria mas similar). Todo se
// expresa como atencion sobre el codebook de memorias x_mu (= leyes
// compiladas L20). NO toca QKernel; es complementario.
//
// Garantia: la energia E es monotonamente decreciente bajo update sincrono
// para beta > 0 (Ramsauer et al., Theorem 3). Aqui se itera hasta que el
// cambio de estado <= tol o se alcanza max_iters.
// =============================================================================

#ifndef EASYATOM_DYNAMICS_MODERN_HOPFIELD_HPP
#define EASYATOM_DYNAMICS_MODERN_HOPFIELD_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "easyatom/hilbert/state.hpp"

namespace easyatom::dynamics::modern {

using easyatom::hilbert::Complex;
using easyatom::hilbert::inner;
using easyatom::hilbert::State;

struct HopfieldConfig {
    double      beta      = 8.0;     // temperatura inversa (mayor -> mas afinada)
    std::size_t max_iters = 32;
    double      tol       = 1e-9;    // cambio relativo de norma para parar
};

struct HopfieldResult {
    State       xi;                   // estado convergido (atractor)
    std::size_t iters       = 0;
    double      energy      = 0.0;    // E(xi) final
    double      last_delta  = 0.0;    // ||xi_t - xi_{t-1}||^2 final
    bool        converged   = false;
};

namespace detail {

[[nodiscard]] inline double energy_of(const std::vector<double>& re_inner,
                                      double                    beta) {
    if (re_inner.empty()) return 0.0;
    const double m = *std::max_element(re_inner.begin(), re_inner.end());
    double sum_exp = 0.0;
    for (double s : re_inner) sum_exp += std::exp(beta * (s - m));
    return -(m + std::log(sum_exp) / beta);
}

[[nodiscard]] inline std::vector<double>
softmax_weights(const std::vector<double>& re_inner, double beta) {
    std::vector<double> w(re_inner.size(), 0.0);
    if (re_inner.empty()) return w;
    const double m = *std::max_element(re_inner.begin(), re_inner.end());
    double sum_exp = 0.0;
    for (std::size_t i = 0; i < re_inner.size(); ++i) {
        w[i]    = std::exp(beta * (re_inner[i] - m));
        sum_exp += w[i];
    }
    if (sum_exp == 0.0) return w;
    for (auto& v : w) v /= sum_exp;
    return w;
}

[[nodiscard]] inline double delta_norm2(const State& a, const State& b) {
    const std::size_t D = a.dim();
    double s = 0.0;
    for (std::size_t i = 0; i < D; ++i) {
        const Complex d = a[i] - b[i];
        s += std::norm(d);
    }
    return s;
}

}  // namespace detail

// Recorre el codebook de memorias x_mu y calcula la energia y los pesos
// softmax sobre Re<x_mu, xi>.
[[nodiscard]] inline double energy(const State&              xi,
                                   const std::vector<State>& memories,
                                   double                    beta) {
    if (memories.empty()) return 0.0;
    if (beta <= 0.0)
        throw std::invalid_argument("modern_hopfield::energy: beta <= 0.");
    std::vector<double> r;
    r.reserve(memories.size());
    for (const auto& m : memories) {
        if (m.dim() != xi.dim())
            throw std::invalid_argument(
                "modern_hopfield::energy: dim mismatch.");
        r.push_back(inner(m, xi).real());
    }
    return detail::energy_of(r, beta);
}

// Una actualizacion sincrona: xi' = sum_mu softmax(beta * Re<x_mu, xi>) * x_mu
[[nodiscard]] inline State step(const State&              xi,
                                const std::vector<State>& memories,
                                double                    beta) {
    if (memories.empty())
        throw std::invalid_argument("modern_hopfield::step: memorias vacias.");
    if (beta <= 0.0)
        throw std::invalid_argument("modern_hopfield::step: beta <= 0.");
    const std::size_t D = xi.dim();
    std::vector<double> r;
    r.reserve(memories.size());
    for (const auto& m : memories) {
        if (m.dim() != D)
            throw std::invalid_argument(
                "modern_hopfield::step: dim mismatch.");
        r.push_back(inner(m, xi).real());
    }
    auto w = detail::softmax_weights(r, beta);
    std::vector<Complex> out(D, Complex{0.0, 0.0});
    for (std::size_t mu = 0; mu < memories.size(); ++mu) {
        const Complex wc{w[mu], 0.0};
        for (std::size_t i = 0; i < D; ++i)
            out[i] += wc * memories[mu][i];
    }
    return State(std::move(out));
}

// Converge desde xi0 hacia el atractor; retorna estado, energia y trayecto.
[[nodiscard]] inline HopfieldResult run(const State&              xi0,
                                        const std::vector<State>& memories,
                                        const HopfieldConfig&     cfg = {}) {
    if (memories.empty())
        throw std::invalid_argument("modern_hopfield::run: memorias vacias.");
    if (cfg.beta <= 0.0)
        throw std::invalid_argument("modern_hopfield::run: beta <= 0.");
    if (cfg.max_iters == 0)
        throw std::invalid_argument("modern_hopfield::run: max_iters == 0.");

    HopfieldResult res;
    res.xi = xi0;
    State prev = xi0;
    for (std::size_t it = 0; it < cfg.max_iters; ++it) {
        State next = step(res.xi, memories, cfg.beta);
        ++res.iters;
        const double d = detail::delta_norm2(next, res.xi);
        res.last_delta = d;
        prev   = std::move(res.xi);
        res.xi = std::move(next);
        if (d <= cfg.tol) { res.converged = true; break; }
    }
    res.energy = energy(res.xi, memories, cfg.beta);
    return res;
}

}  // namespace easyatom::dynamics::modern

#endif  // EASYATOM_DYNAMICS_MODERN_HOPFIELD_HPP
