// =============================================================================
// EasyAtom · Ladrillo 8 — C ABI público.
// =============================================================================
//
// Esta cabecera define la INTERFAZ ESTABLE de C que cualquier lenguaje
// (Java/Kotlin vía JNI, Python vía ctypes/cffi, Rust vía FFI, JavaScript
// vía N-API/WebAssembly, etc.) puede consumir sin tocar C++.
//
// Garantías:
//   * Solo tipos C: punteros opacos + escalares + arrays planos.
//   * Sin excepciones cruzando la frontera: todo se convierte en `int`
//     de error (eatom_status). 0 = OK, negativos = error.
//   * Sin asignación dinámica oculta entrante: el caller pasa buffers de
//     salida y nosotros escribimos en ellos.
//   * Reentrante por handle: cada eatom_kernel_t es independiente; el caller
//     gestiona la sincronización entre threads sobre el MISMO handle.
//
// Modelo:
//   1. eatom_kernel_create(dim, seed)     → handle.
//   2. eatom_kernel_ingest_many(...)      → registra nombres en la codebook.
//   3. eatom_kernel_query_role_filler(...) → consulta directa role:filler.
//   4. eatom_kernel_argmax(...)           → colapsa sobre candidatos.
//   5. eatom_kernel_destroy(handle).

#ifndef EASYATOM_C_API_H
#define EASYATOM_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eatom_kernel eatom_kernel_t;

typedef enum eatom_status {
    EATOM_OK                  =  0,
    EATOM_ERR_NULL            = -1,
    EATOM_ERR_INVALID_ARG     = -2,
    EATOM_ERR_NOT_FOUND       = -3,
    EATOM_ERR_BUFFER_TOO_SMALL= -4,
    EATOM_ERR_INTERNAL        = -5
} eatom_status_t;

/// Crea un kernel con dimensión `dim` y semilla maestra `seed`.
/// Devuelve NULL si dim==0 o por OOM.
eatom_kernel_t* eatom_kernel_create(size_t dim, uint64_t seed);

/// Destruye el kernel. Acepta NULL.
void eatom_kernel_destroy(eatom_kernel_t* k);

/// Devuelve la dimensión del kernel.
size_t eatom_kernel_dim(const eatom_kernel_t* k);

/// Devuelve el tamaño actual de la codebook.
size_t eatom_kernel_codebook_size(const eatom_kernel_t* k);

/// Registra un nombre (idempotente). Devuelve EATOM_OK / error.
int eatom_kernel_ingest(eatom_kernel_t* k, const char* name);

/// Compone bundle de pares role:filler y consulta `query_role`.
/// Devuelve, vía argmax sobre `candidates`, el índice ganador en
/// `*out_winner_index`.
///
///   roles[i]      , fillers[i]      i=0..n_pairs-1
///   query_role    : rol a consultar
///   candidates[i] , i=0..n_candidates-1
///
/// Todos los nombres deben estar previamente en la codebook (o serán
/// añadidos vía ingest implícito si autoingest != 0).
int eatom_kernel_query_pairs_argmax(
    eatom_kernel_t* k,
    const char* const* roles,        size_t n_pairs,
    const char* const* fillers,
    const char* query_role,
    const char* const* candidates,   size_t n_candidates,
    int autoingest,
    size_t* out_winner_index);

/// Compone bundle, consulta y devuelve la distribución de probabilidad
/// (p_0,...,p_{n_candidates-1}) en `out_probs` (debe tener al menos
/// n_candidates floats reservados).
int eatom_kernel_query_pairs_probs(
    eatom_kernel_t* k,
    const char* const* roles,        size_t n_pairs,
    const char* const* fillers,
    const char* query_role,
    const char* const* candidates,   size_t n_candidates,
    int autoingest,
    double* out_probs);

// ---------------------------------------------------------------------------
// Ladrillos 10 + 11: decide + decoder semántico.
// ---------------------------------------------------------------------------

typedef enum eatom_decision_kind {
    EATOM_DECISION_ACCEPT     = 0,
    EATOM_DECISION_AMBIGUOUS  = 1,
    EATOM_DECISION_ABSTAIN    = 2,
    EATOM_DECISION_DEGENERATE = 3,
    EATOM_DECISION_INVALID    = 4
} eatom_decision_kind_t;

/// Política de decisión. Pasa NULL para usar la por defecto (balanced).
typedef struct eatom_policy {
    double min_confidence;
    double min_margin;
    double max_entropy_ratio;
    double max_effective_n;     // 0 = desactivado
    int    require_finite_probs;
} eatom_policy_t;

/// Resultado del decisor + decoder.
///
/// `out_explanation` es un buffer UTF-8 que el caller proporciona. Si no
/// cabe la frase completa se trunca con '\0' final y se devuelve el tamaño
/// que se necesitaba en `out_explanation_needed` (si != NULL).
int eatom_kernel_decide_pairs(
    eatom_kernel_t* k,
    const char* const* roles,        size_t n_pairs,
    const char* const* fillers,
    const char* query_role,
    const char* const* candidates,   size_t n_candidates,
    int autoingest,
    const eatom_policy_t* policy,        // NULL → default
    /* salidas: */
    int*    out_kind,                    // eatom_decision_kind_t
    size_t* out_winner_index,
    size_t* out_runner_up_index,
    double* out_confidence,
    double* out_margin,
    double* out_entropy,
    double* out_entropy_ratio,
    double* out_effective_n,
    double* out_probs,                   // n_candidates doubles, opcional (NULL ok)
    char*   out_explanation,             // buffer del caller
    size_t  out_explanation_capacity,
    size_t* out_explanation_needed);     // opcional

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EASYATOM_C_API_H
