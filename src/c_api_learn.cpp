// =============================================================================
// easyatom/src/c_api_learn.cpp  --  L39
//
// Implementacion del C ABI de aprendizaje. UNICA unidad no header-only ademas
// de c_api.cpp. Atrapa toda excepcion C++ y la traduce a eatom_status_t.
// =============================================================================

#include "easyatom/c_api_learn.h"

#include "easyatom/auto/loop.hpp"
#include "easyatom/auto/loop_ranked.hpp"
#include "easyatom/facade/learn.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/qkernel/qkernel.hpp"

#include <exception>
#include <string>
#include <vector>

using easyatom::autoloop::LoopConfig;
using easyatom::autoloop::LoopRankedConfig;
using easyatom::autoloop::LoopRankedReport;
using easyatom::autoloop::LoopReport;
using easyatom::autoloop::RankMode;
using easyatom::autoloop::run_auto_loop;
using easyatom::autoloop::run_auto_loop_ranked;
using easyatom::facade::detect_and_enqueue;
using easyatom::facade::IngestTextsReport;
using easyatom::facade::LearnSession;
using easyatom::facade::ingest_texts;
using easyatom::hilbert::State;
using easyatom::qkernel::QKernel;

struct eatom_learn {
    QKernel       kernel;
    LearnSession  session;
    eatom_learn(std::size_t dim, std::uint64_t seed) : kernel(dim, seed) {
        session.kernel = &kernel;
    }
};

namespace {

// Convierte topics[i] en un probe State pidiendolo al kernel.
// El kernel garantiza estados deterministas, casi-ortogonales entre nombres
// distintos. Se ingiere sobre el propio kernel de la session, lo que ademas
// asegura que ese nombre existe como anchor para futuras compilaciones.
std::vector<State> probes_from_topics(QKernel& k,
                                      const char* const* topics,
                                      std::size_t        n) {
    std::vector<State> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const char* t = topics ? topics[i] : nullptr;
        if (!t) { out.emplace_back(k.dim()); continue; }
        out.push_back(k.ingest(std::string(t)));  // copia
    }
    return out;
}

}  // namespace

extern "C" {

eatom_learn_t* eatom_learn_create(size_t dim, uint64_t seed) {
    if (dim == 0) return nullptr;
    try {
        return new eatom_learn(dim, seed);
    } catch (...) {
        return nullptr;
    }
}

void eatom_learn_destroy(eatom_learn_t* s) {
    delete s;
}

size_t eatom_learn_codebook_size(const eatom_learn_t* s) {
    if (!s) return 0;
    return s->session.codebook.size();
}

size_t eatom_learn_pending_count(const eatom_learn_t* s) {
    if (!s) return 0;
    return s->session.queue.pending().size();
}

int eatom_learn_ingest_texts(
    eatom_learn_t* s,
    const char* const* texts, size_t n,
    size_t* out_total,
    size_t* out_compiled,
    size_t* out_accepted,
    size_t* out_rejected,
    size_t* out_failed) {
    if (!s || (!texts && n != 0)) return EATOM_ERR_NULL;
    try {
        std::vector<std::string> ts;
        ts.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            ts.emplace_back(texts[i] ? texts[i] : "");
        IngestTextsReport rep = ingest_texts(s->session, ts);
        if (out_total)    *out_total    = rep.total;
        if (out_compiled) *out_compiled = rep.compiled;
        if (out_accepted) *out_accepted = rep.accepted;
        if (out_rejected) *out_rejected = rep.rejected;
        if (out_failed)   *out_failed   = rep.failed;
        return EATOM_OK;
    } catch (const std::invalid_argument&) {
        return EATOM_ERR_INVALID_ARG;
    } catch (...) {
        return EATOM_ERR_INTERNAL;
    }
}

int eatom_learn_detect_topics_and_enqueue(
    eatom_learn_t* s,
    const char* const* topics, size_t n,
    double theta,
    size_t* out_enqueued) {
    if (!s) return EATOM_ERR_NULL;
    if (theta < 0.0) return EATOM_ERR_INVALID_ARG;
    try {
        auto probes = probes_from_topics(s->kernel, topics, n);
        std::vector<std::string> tnames;
        tnames.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            tnames.emplace_back(topics && topics[i] ? topics[i] : "");
        const std::size_t enq = detect_and_enqueue(
            s->session, probes, tnames, theta);
        if (out_enqueued) *out_enqueued = enq;
        return EATOM_OK;
    } catch (const std::invalid_argument&) {
        return EATOM_ERR_INVALID_ARG;
    } catch (...) {
        return EATOM_ERR_INTERNAL;
    }
}

int eatom_learn_run_autoloop(
    eatom_learn_t* s,
    const char* const* probe_topics, size_t n_queries,
    double      gap_theta,
    size_t      k_top,
    size_t      max_iters,
    size_t* out_iters,
    size_t* out_gaps_detected,
    size_t* out_proposals,
    size_t* out_accepted,
    size_t* out_rejected_coherence) {
    if (!s) return EATOM_ERR_NULL;
    if (gap_theta < 0.0 || max_iters == 0) return EATOM_ERR_INVALID_ARG;
    try {
        auto qs = probes_from_topics(s->kernel, probe_topics, n_queries);
        LoopConfig cfg;
        cfg.gap_theta     = gap_theta;
        cfg.k_top         = k_top == 0 ? 3 : k_top;
        cfg.max_iters     = max_iters;
        cfg.coherence_k   = s->session.coherence_k;
        cfg.coherence_eps = s->session.coherence_eps;
        LoopReport rep = run_auto_loop(s->session.codebook, qs, cfg);
        if (out_iters)              *out_iters              = rep.iters;
        if (out_gaps_detected)      *out_gaps_detected      = rep.gaps_detected;
        if (out_proposals)          *out_proposals          = rep.proposals;
        if (out_accepted)           *out_accepted           = rep.accepted;
        if (out_rejected_coherence) *out_rejected_coherence = rep.rejected_coherence;
        return EATOM_OK;
    } catch (const std::invalid_argument&) {
        return EATOM_ERR_INVALID_ARG;
    } catch (...) {
        return EATOM_ERR_INTERNAL;
    }
}

int eatom_learn_run_autoloop_ranked(
    eatom_learn_t* s,
    const char* const* probe_topics, size_t n_queries,
    double      gap_theta,
    size_t      k_min,
    size_t      k_max,
    size_t      max_iters,
    int         mode,
    size_t* out_iters,
    size_t* out_gaps_detected,
    size_t* out_candidates_total,
    size_t* out_candidates_unique,
    size_t* out_accepted,
    size_t* out_rejected_coherence) {
    if (!s) return EATOM_ERR_NULL;
    if (gap_theta < 0.0 || max_iters == 0 || k_min == 0 || k_max < k_min)
        return EATOM_ERR_INVALID_ARG;
    try {
        auto qs = probes_from_topics(s->kernel, probe_topics, n_queries);
        LoopRankedConfig cfg;
        cfg.gap_theta     = gap_theta;
        cfg.k_min         = k_min;
        cfg.k_max         = k_max;
        cfg.max_iters     = max_iters;
        cfg.coherence_k   = s->session.coherence_k;
        cfg.coherence_eps = s->session.coherence_eps;
        cfg.mode          = (mode == 1) ? RankMode::Discovery : RankMode::Energy;
        LoopRankedReport rep = run_auto_loop_ranked(
            s->session.codebook, qs, cfg);
        if (out_iters)              *out_iters              = rep.iters;
        if (out_gaps_detected)      *out_gaps_detected      = rep.gaps_detected;
        if (out_candidates_total)   *out_candidates_total   = rep.candidates_total;
        if (out_candidates_unique)  *out_candidates_unique  = rep.candidates_unique;
        if (out_accepted)           *out_accepted           = rep.accepted;
        if (out_rejected_coherence) *out_rejected_coherence = rep.rejected_coherence;
        return EATOM_OK;
    } catch (const std::invalid_argument&) {
        return EATOM_ERR_INVALID_ARG;
    } catch (...) {
        return EATOM_ERR_INTERNAL;
    }
}

}  // extern "C"
