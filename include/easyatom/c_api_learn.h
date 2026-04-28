// =============================================================================
// easyatom/c_api_learn.h  --  L39
//
// C ABI para la fachada LearnSession (L34) + auto-loop ranked (L40).
// Pensada para el bridge JNI de Android / iOS y consumible desde TS via
// React Native NativeModule. NO hace red: la ingesta externa la maneja
// el caller (TS) llamando a eatom_learn_ingest_texts con los textos ya
// descargados.
//
// Convencion: int de retorno = eatom_status_t. 0 = OK; punteros opacos
// para handles.
// =============================================================================

#ifndef EASYATOM_C_API_LEARN_H
#define EASYATOM_C_API_LEARN_H

#include <stddef.h>
#include <stdint.h>

#include "easyatom/c_api.h"  // reutiliza eatom_status_t y eatom_kernel_t

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eatom_learn eatom_learn_t;

/// Crea una sesion de aprendizaje propietaria de su propio kernel.
/// Devuelve NULL si dim == 0 o por OOM.
eatom_learn_t* eatom_learn_create(size_t dim, uint64_t seed);

/// Destruye la sesion (acepta NULL).
void eatom_learn_destroy(eatom_learn_t* s);

/// Tamano actual del codebook compilado.
size_t eatom_learn_codebook_size(const eatom_learn_t* s);

/// Items pendientes en la cola de ingesta externa.
size_t eatom_learn_pending_count(const eatom_learn_t* s);

/// Ingiere N textos: cada uno se compila (compile_law) y se valida
/// contra el codebook (evaluate_addition). Devuelve metricas en los
/// out-params (cualquiera puede ser NULL).
int eatom_learn_ingest_texts(
    eatom_learn_t* s,
    const char* const* texts, size_t n,
    size_t* out_total,
    size_t* out_compiled,
    size_t* out_accepted,
    size_t* out_rejected,
    size_t* out_failed);

/// Encola N topics como pendientes externos *si* su correspondiente
/// probe esta por debajo del umbral theta de densidad. Cada topic[i]
/// se asocia a un probe sintetico derivado del mismo string (hash
/// estable -> indice de base canonica del kernel), por lo que la
/// llamada NO requiere que el caller construya States.
int eatom_learn_detect_topics_and_enqueue(
    eatom_learn_t* s,
    const char* const* topics, size_t n,
    double theta,
    size_t* out_enqueued);

/// Ejecuta un autoloop sobre `n_queries` probes sinteticos (mismo
/// hash -> base canonica) con configuracion basica L32. Sin red.
int eatom_learn_run_autoloop(
    eatom_learn_t* s,
    const char* const* probe_topics, size_t n_queries,
    double      gap_theta,
    size_t      k_top,
    size_t      max_iters,
    /* out: */
    size_t* out_iters,
    size_t* out_gaps_detected,
    size_t* out_proposals,
    size_t* out_accepted,
    size_t* out_rejected_coherence);

/// Variante con ranking formal L40. mode: 0 = Energy (L37, asc.),
/// 1 = Discovery (L36, desc.).
int eatom_learn_run_autoloop_ranked(
    eatom_learn_t* s,
    const char* const* probe_topics, size_t n_queries,
    double      gap_theta,
    size_t      k_min,
    size_t      k_max,
    size_t      max_iters,
    int         mode,
    /* out: */
    size_t* out_iters,
    size_t* out_gaps_detected,
    size_t* out_candidates_total,
    size_t* out_candidates_unique,
    size_t* out_accepted,
    size_t* out_rejected_coherence);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EASYATOM_C_API_LEARN_H
