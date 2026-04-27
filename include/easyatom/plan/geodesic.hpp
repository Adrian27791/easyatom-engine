// =============================================================================
// easyatom/plan/geodesic.hpp  --  L31
//
// Planner geodesico sobre el grafo de leyes:
//
//   - Nodo i = laws[i].
//   - Arista (i, j) con peso w_ij = 1 - fidelity(laws[i].state, laws[j].state).
//   - Filtro tipo persistent-homology beta_0: descartamos aristas con
//     w_ij > max_eps (equivalente a separar componentes conexas en la
//     escala max_eps).
//   - Dijkstra clasico O((V+E) log V) sobre el grafo filtrado.
//
//   find_path(laws, src, dst, max_eps) -> vector<size_t>
//     ruta como secuencia de indices (incluye src y dst), vacia si no hay
//     camino dentro de la misma componente.
// =============================================================================

#ifndef EASYATOM_PLAN_GEODESIC_HPP
#define EASYATOM_PLAN_GEODESIC_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

#include "easyatom/cst/compile.hpp"
#include "easyatom/hilbert/state.hpp"

namespace easyatom::plan {

using easyatom::cst::CompiledLaw;
using easyatom::hilbert::fidelity;

[[nodiscard]] inline std::vector<std::size_t> find_path(
    const std::vector<CompiledLaw>& laws,
    std::size_t                     src,
    std::size_t                     dst,
    double                          max_eps) {
    if (laws.empty())
        throw std::invalid_argument("find_path: laws vacio.");
    if (src >= laws.size() || dst >= laws.size())
        throw std::out_of_range("find_path: src/dst fuera de rango.");
    if (max_eps < 0.0 || max_eps > 1.0)
        throw std::invalid_argument(
            "find_path: max_eps debe estar en [0,1].");
    if (src == dst) return {src};

    const std::size_t N = laws.size();
    const double      INF = std::numeric_limits<double>::infinity();
    std::vector<double>      dist(N, INF);
    std::vector<std::size_t> prev(N, std::numeric_limits<std::size_t>::max());

    using Item = std::pair<double, std::size_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    dist[src] = 0.0;
    pq.push({0.0, src});

    while (!pq.empty()) {
        auto [du, u] = pq.top();
        pq.pop();
        if (du > dist[u]) continue;
        if (u == dst) break;

        for (std::size_t v = 0; v < N; ++v) {
            if (v == u) continue;
            const double w = 1.0 - fidelity(laws[u].state, laws[v].state);
            if (w > max_eps) continue;     // filtro beta_0
            const double nd = du + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                prev[v] = u;
                pq.push({nd, v});
            }
        }
    }

    if (dist[dst] == INF) return {};

    std::vector<std::size_t> path;
    for (std::size_t at = dst; at != std::numeric_limits<std::size_t>::max();
         at = prev[at]) {
        path.push_back(at);
        if (at == src) break;
    }
    std::reverse(path.begin(), path.end());
    if (path.empty() || path.front() != src) return {};
    return path;
}

}  // namespace easyatom::plan

#endif  // EASYATOM_PLAN_GEODESIC_HPP
