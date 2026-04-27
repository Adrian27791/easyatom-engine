// =============================================================================
// easyatom/hypothesis/engine.hpp  --  L27
//
// Iteracion sobre hipotesis con validacion cruzada por holdout.
//
// Idea:
//   - Particion holdout determinista del corpus en (train, test).
//   - Score de cobertura: para cada law del test, density(law.state,
//     train_codebook) >= threshold cuenta como "explicada" por el train.
//   - validate(train, test, threshold) -> ValidationReport{accuracy, hits, total}.
//   - iterate_hypothesis(train, test, candidates, threshold) prueba cada
//     candidato, lo agrega temporalmente al train, recalcula accuracy y se
//     queda con el que mas la sube. Devuelve el indice y el delta. Si nada
//     mejora estrictamente, devuelve {npos, 0.0, baseline}.
//
// Header-only. Reusa easyatom::epistemic::density (L24).
// =============================================================================

#ifndef EASYATOM_HYPOTHESIS_ENGINE_HPP
#define EASYATOM_HYPOTHESIS_ENGINE_HPP

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "easyatom/cst/compile.hpp"
#include "easyatom/epistemic/gap.hpp"

namespace easyatom::hypothesis {

using easyatom::cst::CompiledLaw;
using easyatom::epistemic::density;

struct Holdout {
    std::vector<CompiledLaw> train;
    std::vector<CompiledLaw> test;
};

[[nodiscard]] inline Holdout split_holdout(
    const std::vector<CompiledLaw>& corpus, std::size_t train_count) {
    if (corpus.empty()) {
        throw std::invalid_argument("split_holdout: corpus vacio.");
    }
    if (train_count == 0 || train_count >= corpus.size()) {
        throw std::invalid_argument(
            "split_holdout: train_count debe estar en (0, corpus.size()).");
    }
    Holdout h;
    h.train.assign(corpus.begin(), corpus.begin() + train_count);
    h.test.assign(corpus.begin() + train_count, corpus.end());
    return h;
}

struct ValidationReport {
    std::size_t hits     = 0;
    std::size_t total    = 0;
    double      accuracy = 0.0;   // hits / total, 0 si total==0
};

[[nodiscard]] inline ValidationReport validate(
    const std::vector<CompiledLaw>& train,
    const std::vector<CompiledLaw>& test,
    double                          threshold) {
    if (threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("validate: threshold fuera de [0,1].");
    }
    ValidationReport r;
    r.total = test.size();
    if (train.empty() || test.empty()) return r;

    for (const auto& law : test) {
        const double d = density(law.state, train);
        if (d >= threshold) ++r.hits;
    }
    r.accuracy = static_cast<double>(r.hits) /
                 static_cast<double>(r.total);
    return r;
}

struct HypothesisChoice {
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();
    std::size_t      candidate_index   = npos;
    double           delta_accuracy    = 0.0;
    ValidationReport baseline_report;
    ValidationReport best_report;
};

[[nodiscard]] inline HypothesisChoice iterate_hypothesis(
    const std::vector<CompiledLaw>& train,
    const std::vector<CompiledLaw>& test,
    const std::vector<CompiledLaw>& candidates,
    double                          threshold) {
    HypothesisChoice choice;
    choice.baseline_report = validate(train, test, threshold);
    choice.best_report     = choice.baseline_report;

    for (std::size_t k = 0; k < candidates.size(); ++k) {
        std::vector<CompiledLaw> extended = train;
        extended.push_back(candidates[k]);
        const auto rep = validate(extended, test, threshold);
        const double delta = rep.accuracy - choice.baseline_report.accuracy;
        if (delta > choice.delta_accuracy) {
            choice.delta_accuracy  = delta;
            choice.candidate_index = k;
            choice.best_report     = rep;
        }
    }
    return choice;
}

}  // namespace easyatom::hypothesis

#endif  // EASYATOM_HYPOTHESIS_ENGINE_HPP
