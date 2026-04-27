// =============================================================================
// easyatom/decide/counterfactual.hpp  --  L30
//
// Certificado contrafactual:
//
//   Dada una funcion arbitraria winner_fn(State) -> size_t (cualquier
//   regla de decision externa) y un estado base s con ganador W = winner_fn(s),
//   buscamos la perturbacion DELTA de norma minima que haga
//   winner_fn(s + DELTA) != W.
//
//   La busqueda se restringe a direcciones explicitas (states "directions"
//   provistos por el caller). Para cada direccion d, hacemos bisqueda
//   binaria sobre alpha en (0, alpha_max] hasta encontrar el menor alpha
//   tal que winner_fn(s + alpha*d) != W (si existe). Se reporta la
//   (direccion, alpha) con la perturbacion delta de menor norma2.
//
//   Esto da explicabilidad real "que cambio de input minimal habria
//   cambiado la decision".
// =============================================================================

#ifndef EASYATOM_DECIDE_COUNTERFACTUAL_HPP
#define EASYATOM_DECIDE_COUNTERFACTUAL_HPP

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "easyatom/hilbert/state.hpp"

namespace easyatom::decide {

using easyatom::hilbert::Complex;
using easyatom::hilbert::State;

struct Counterfactual {
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();
    State       delta;                // perturbacion (mismo dim que s)
    std::size_t direction_index = npos;
    std::size_t new_winner      = npos;
    double      delta_norm2     = std::numeric_limits<double>::infinity();
    bool        found           = false;
};

namespace detail {

inline State scale_add(const State& s, double alpha, const State& d) {
    State out(s.dim());
    const auto& a_s = s.amplitudes();
    const auto& a_d = d.amplitudes();
    for (std::size_t i = 0; i < s.dim(); ++i)
        out[i] = a_s[i] + Complex{alpha, 0.0} * a_d[i];
    return out;
}

inline double norm2_of(const State& d, double alpha) {
    double n2 = 0.0;
    const auto& a_d = d.amplitudes();
    for (std::size_t i = 0; i < d.dim(); ++i)
        n2 += std::norm(a_d[i]);
    return alpha * alpha * n2;
}

}  // namespace detail

[[nodiscard]] inline Counterfactual find_counterfactual(
    const State&                                       s,
    const std::vector<State>&                          directions,
    const std::function<std::size_t(const State&)>&    winner_fn,
    double                                             alpha_max  = 1.0,
    int                                                bisect_iters = 32) {
    if (directions.empty())
        throw std::invalid_argument(
            "find_counterfactual: directions vacios.");
    if (alpha_max <= 0.0)
        throw std::invalid_argument(
            "find_counterfactual: alpha_max debe ser > 0.");
    for (const auto& d : directions) {
        if (d.dim() != s.dim())
            throw std::invalid_argument(
                "find_counterfactual: dim de direccion != dim(s).");
    }

    const std::size_t W = winner_fn(s);
    Counterfactual best;

    for (std::size_t k = 0; k < directions.size(); ++k) {
        const State& d = directions[k];
        // Si en alpha_max no cambia, esta direccion no sirve.
        const std::size_t w_max = winner_fn(detail::scale_add(s, alpha_max, d));
        if (w_max == W) continue;

        // Bisqueda binaria: invariante: lo no cambia (w==W), hi cambia.
        double lo = 0.0;
        double hi = alpha_max;
        std::size_t w_hi = w_max;
        for (int it = 0; it < bisect_iters; ++it) {
            const double mid = 0.5 * (lo + hi);
            const std::size_t wm = winner_fn(detail::scale_add(s, mid, d));
            if (wm == W) lo = mid;
            else { hi = mid; w_hi = wm; }
        }

        const double n2 = detail::norm2_of(d, hi);
        if (n2 < best.delta_norm2) {
            best.found           = true;
            best.direction_index = k;
            best.new_winner      = w_hi;
            best.delta_norm2     = n2;

            // Calcula delta = hi * d como State.
            State delta(s.dim());
            const auto& a_d = d.amplitudes();
            for (std::size_t i = 0; i < s.dim(); ++i)
                delta[i] = Complex{hi, 0.0} * a_d[i];
            best.delta = std::move(delta);
        }
    }
    return best;
}

}  // namespace easyatom::decide

#endif  // EASYATOM_DECIDE_COUNTERFACTUAL_HPP
