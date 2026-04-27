// =============================================================================
// easyatom/memory/episodic.hpp  --  L29
//
// Memoria episodica geometrica.
//
// Modelo: cada episodio empareja un timestamp (uint64) con una CompiledLaw.
// La "geometria" se usa para recall por contenido: top-k por fidelity entre
// el state de la query y el state del episodio. Para ventanas temporales
// se filtra por [t0, t1] inclusive.
//
// API:
//   struct Episode { uint64_t ts; CompiledLaw law; };
//   class EpisodicStore {
//       void append(Episode);
//       size_t size() const noexcept;
//       const std::vector<Episode>& all() const noexcept;
//       std::vector<Episode> recall_window(uint64_t t0, uint64_t t1) const;
//       std::vector<size_t>  recall_by_content(const State& q, size_t k) const;
//       std::vector<size_t>  recall_by_content_in_window(const State& q,
//                              uint64_t t0, uint64_t t1, size_t k) const;
//   };
//
// recall_by_content devuelve indices ordenados por fidelity descendente.
// =============================================================================

#ifndef EASYATOM_MEMORY_EPISODIC_HPP
#define EASYATOM_MEMORY_EPISODIC_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "easyatom/cst/compile.hpp"
#include "easyatom/hilbert/state.hpp"

namespace easyatom::memory {

using easyatom::cst::CompiledLaw;
using easyatom::hilbert::fidelity;
using easyatom::hilbert::State;

struct Episode {
    std::uint64_t ts = 0;
    CompiledLaw   law;
};

class EpisodicStore {
public:
    void append(Episode e) { episodes_.push_back(std::move(e)); }
    [[nodiscard]] std::size_t size() const noexcept { return episodes_.size(); }
    [[nodiscard]] const std::vector<Episode>& all() const noexcept {
        return episodes_;
    }
    void clear() noexcept { episodes_.clear(); }

    [[nodiscard]] std::vector<Episode>
    recall_window(std::uint64_t t0, std::uint64_t t1) const {
        if (t0 > t1)
            throw std::invalid_argument("recall_window: t0 > t1.");
        std::vector<Episode> out;
        for (const auto& e : episodes_)
            if (e.ts >= t0 && e.ts <= t1) out.push_back(e);
        return out;
    }

    [[nodiscard]] std::vector<std::size_t>
    recall_by_content(const State& q, std::size_t k) const {
        return top_k_indices_(q, k, [](const Episode&) { return true; });
    }

    [[nodiscard]] std::vector<std::size_t>
    recall_by_content_in_window(const State& q,
                                std::uint64_t t0,
                                std::uint64_t t1,
                                std::size_t   k) const {
        if (t0 > t1)
            throw std::invalid_argument(
                "recall_by_content_in_window: t0 > t1.");
        return top_k_indices_(q, k,
            [t0, t1](const Episode& e) {
                return e.ts >= t0 && e.ts <= t1;
            });
    }

private:
    template <typename Pred>
    std::vector<std::size_t>
    top_k_indices_(const State& q, std::size_t k, Pred pred) const {
        if (k == 0)
            throw std::invalid_argument("recall_by_content: k = 0.");
        std::vector<std::pair<double, std::size_t>> scored;
        scored.reserve(episodes_.size());
        for (std::size_t i = 0; i < episodes_.size(); ++i) {
            if (!pred(episodes_[i])) continue;
            const double f = fidelity(q, episodes_[i].law.state);
            scored.emplace_back(f, i);
        }
        const std::size_t kk = std::min(k, scored.size());
        std::partial_sort(scored.begin(), scored.begin() + kk, scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::size_t> out;
        out.reserve(kk);
        for (std::size_t i = 0; i < kk; ++i) out.push_back(scored[i].second);
        return out;
    }

    std::vector<Episode> episodes_;
};

}  // namespace easyatom::memory

#endif  // EASYATOM_MEMORY_EPISODIC_HPP
