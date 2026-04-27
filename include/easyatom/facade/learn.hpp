// =============================================================================
// easyatom/facade/learn.hpp  --  L34
//
// Fachada de alto nivel que une L20 (compile_law), L23 (coherence),
// L24 (gap detection), L32 (auto_loop) y L33 (ingest queue).
//
// Diseñada para ser el punto de entrada que envolvera el bridge JNI/RN:
// el caller de TypeScript puede mantener un LearnSession opaco y llamar
// las 4 operaciones de alto nivel sin saber de States ni Triplets.
//
//   ingest_texts(session, texts)           -> intenta compilar+aceptar cada texto
//   detect_and_enqueue(session, probes, theta) -> encola gaps externos
//   learn_locally(session, queries, cfg)   -> autoloop por recombinacion
//   learn_externally(session, perm, ...)   -> drain de cola con permiso
// =============================================================================

#ifndef EASYATOM_FACADE_LEARN_HPP
#define EASYATOM_FACADE_LEARN_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "easyatom/auto/loop.hpp"
#include "easyatom/cst/compile.hpp"
#include "easyatom/epistemic/gap.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ingest/external.hpp"
#include "easyatom/qkernel/qkernel.hpp"
#include "easyatom/reason/coherence.hpp"

namespace easyatom::facade {

using easyatom::autoloop::LoopConfig;
using easyatom::autoloop::LoopReport;
using easyatom::autoloop::run_auto_loop;
using easyatom::cst::compile_law;
using easyatom::cst::CompiledLaw;
using easyatom::cst::CompileError;
using easyatom::cst::Triplet;
using easyatom::epistemic::find_gaps;
using easyatom::hilbert::State;
using easyatom::ingest::DrainReport;
using easyatom::ingest::IngestQueue;
using easyatom::ingest::Permission;
using easyatom::qkernel::QKernel;
using easyatom::reason::CoherenceReport;
using easyatom::reason::evaluate_addition;

struct LearnSession {
    QKernel*                  kernel = nullptr;   // no posee
    std::vector<CompiledLaw>  codebook;
    IngestQueue               queue;
    std::size_t               coherence_k   = 4;
    double                    coherence_eps = 0.5;
};

struct IngestTextsReport {
    std::size_t total      = 0;
    std::size_t compiled   = 0;     // compile_law no lanzo
    std::size_t accepted   = 0;     // paso coherence
    std::size_t rejected   = 0;     // contradiccion o beta1 subio
    std::size_t failed     = 0;     // compile_law lanzo CompileError
};

// Compila cada texto, lo evalua contra el cuerpo actual y lo acepta si no
// hay contradiccion ni se rompe la topologia.
inline IngestTextsReport ingest_texts(LearnSession&                  s,
                                      const std::vector<std::string>& texts) {
    if (s.kernel == nullptr)
        throw std::invalid_argument("ingest_texts: session.kernel == null.");
    IngestTextsReport rep{};
    rep.total = texts.size();

    for (const auto& text : texts) {
        CompiledLaw cand;
        try {
            cand = compile_law(*s.kernel, text);
        } catch (const CompileError&) {
            ++rep.failed;
            continue;
        }
        ++rep.compiled;

        std::vector<Triplet> ts;
        std::vector<State>   ss;
        ts.reserve(s.codebook.size());
        ss.reserve(s.codebook.size());
        for (const auto& law : s.codebook) {
            ts.push_back(law.triplet);
            ss.push_back(law.state);
        }
        CoherenceReport cr = evaluate_addition(
            ts, ss, cand.triplet, cand.state,
            s.coherence_k, s.coherence_eps);
        if (cr.accepted) {
            s.codebook.push_back(std::move(cand));
            ++rep.accepted;
        } else {
            ++rep.rejected;
        }
    }
    return rep;
}

// Por cada gap detectado, encola un PendingQuery con el topic_hint dado.
// Devuelve cuantos items se agregaron a la cola.
inline std::size_t detect_and_enqueue(LearnSession&                   s,
                                      const std::vector<State>&       probes,
                                      const std::vector<std::string>& topics,
                                      double                          theta) {
    if (probes.size() != topics.size())
        throw std::invalid_argument(
            "detect_and_enqueue: probes y topics distinto tamano.");
    auto gaps = find_gaps(probes, s.codebook, theta);
    for (const auto& g : gaps)
        s.queue.enqueue(topics[g.query_index]);
    return gaps.size();
}

// Atajo: ejecuta el circulo autonomo local sobre el codebook actual.
inline LoopReport learn_locally(LearnSession&             s,
                                const std::vector<State>& queries,
                                const LoopConfig&         cfg = {}) {
    LoopConfig local = cfg;
    local.coherence_k   = s.coherence_k;
    local.coherence_eps = s.coherence_eps;
    return run_auto_loop(s.codebook, queries, local);
}

// Drain de la cola externa: por cada texto que retorne el fetcher, intenta
// ingerirlo via compile+coherence. No requiere red en este header.
template <class Authorize, class Fetcher>
inline DrainReport learn_externally(LearnSession& s,
                                    Permission    perm,
                                    Authorize&&   authorize,
                                    Fetcher&&     fetcher) {
    return s.queue.drain(perm,
        std::forward<Authorize>(authorize),
        std::forward<Fetcher>(fetcher),
        [&s](const std::string& text) {
            (void)ingest_texts(s, {text});
        });
}

}  // namespace easyatom::facade

#endif  // EASYATOM_FACADE_LEARN_HPP
