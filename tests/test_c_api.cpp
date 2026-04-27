// Tests del Ladrillo 8 — C ABI público.
//
// Estos tests son C++ pero usan EXCLUSIVAMENTE las funciones extern "C" de
// `easyatom/c_api.h`, demostrando que el motor es invocable desde un
// consumidor que solo conoce esa interfaz.

#include "test_framework.hpp"
#include "easyatom/c_api.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

EATEST_CASE(c_api_create_destroy_basico) {
    auto* k = eatom_kernel_create(64, 42);
    EATEST_REQUIRE(k != nullptr);
    EATEST_REQUIRE(eatom_kernel_dim(k) == 64);
    EATEST_REQUIRE(eatom_kernel_codebook_size(k) == 0);
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_recommend_dim_tiers_exactos) {
    EATEST_REQUIRE(eatom_recommend_dim(EATOM_TIER_LOW)   == 16384u);
    EATEST_REQUIRE(eatom_recommend_dim(EATOM_TIER_MID)   == 32768u);
    EATEST_REQUIRE(eatom_recommend_dim(EATOM_TIER_HIGH)  == 65536u);
    EATEST_REQUIRE(eatom_recommend_dim(EATOM_TIER_ULTRA) == 131072u);
    EATEST_REQUIRE(eatom_recommend_dim(-1) == 0u);
    EATEST_REQUIRE(eatom_recommend_dim(99) == 0u);
}

EATEST_CASE(c_api_create_dim_cero_devuelve_null) {
    auto* k = eatom_kernel_create(0, 1);
    EATEST_REQUIRE(k == nullptr);
}

EATEST_CASE(c_api_destroy_null_es_seguro) {
    eatom_kernel_destroy(nullptr);
    EATEST_REQUIRE(true);
}

EATEST_CASE(c_api_ingest_es_idempotente) {
    auto* k = eatom_kernel_create(32, 7);
    EATEST_REQUIRE(eatom_kernel_ingest(k, "alpha") == EATOM_OK);
    EATEST_REQUIRE(eatom_kernel_ingest(k, "alpha") == EATOM_OK);
    EATEST_REQUIRE(eatom_kernel_codebook_size(k) == 1);
    EATEST_REQUIRE(eatom_kernel_ingest(k, "beta") == EATOM_OK);
    EATEST_REQUIRE(eatom_kernel_codebook_size(k) == 2);
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_ingest_null_devuelve_error) {
    auto* k = eatom_kernel_create(8, 1);
    EATEST_REQUIRE(eatom_kernel_ingest(k, nullptr) == EATOM_ERR_NULL);
    EATEST_REQUIRE(eatom_kernel_ingest(nullptr, "x") == EATOM_ERR_NULL);
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_query_argmax_recupera_filler) {
    auto* k = eatom_kernel_create(1024, 2024);
    const char* roles[]   = {"color", "tamano"};
    const char* fillers[] = {"rojo", "grande"};
    const char* cands[]   = {"rojo", "grande", "azul", "pequeno"};
    size_t winner = 999;
    int rc = eatom_kernel_query_pairs_argmax(
        k, roles, 2, fillers, "color", cands, 4,
        /*autoingest*/ 1, &winner);
    EATEST_REQUIRE(rc == EATOM_OK);
    EATEST_REQUIRE(winner == 0);  // "rojo"

    rc = eatom_kernel_query_pairs_argmax(
        k, roles, 2, fillers, "tamano", cands, 4, 1, &winner);
    EATEST_REQUIRE(rc == EATOM_OK);
    EATEST_REQUIRE(winner == 1);  // "grande"
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_query_probs_es_distribucion_normalizada) {
    auto* k = eatom_kernel_create(512, 11);
    const char* roles[]   = {"k"};
    const char* fillers[] = {"v"};
    const char* cands[]   = {"v", "w", "z"};
    double probs[3] = {0, 0, 0};
    int rc = eatom_kernel_query_pairs_probs(
        k, roles, 1, fillers, "k", cands, 3, 1, probs);
    EATEST_REQUIRE(rc == EATOM_OK);
    double s = probs[0] + probs[1] + probs[2];
    EATEST_REQUIRE_NEAR(s, 1.0, 1e-9);
    EATEST_REQUIRE(probs[0] > probs[1]);
    EATEST_REQUIRE(probs[0] > probs[2]);
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_autoingest_off_lanza_si_falta_nombre) {
    auto* k = eatom_kernel_create(32, 1);
    const char* roles[]   = {"r"};
    const char* fillers[] = {"f"};
    const char* cands[]   = {"f"};
    size_t winner = 0;
    int rc = eatom_kernel_query_pairs_argmax(
        k, roles, 1, fillers, "r", cands, 1,
        /*autoingest*/ 0, &winner);
    EATEST_REQUIRE(rc == EATOM_ERR_INVALID_ARG);
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_n_pairs_cero_lanza) {
    auto* k = eatom_kernel_create(16, 1);
    const char* cands[] = {"x"};
    size_t winner = 0;
    int rc = eatom_kernel_query_pairs_argmax(
        k, nullptr, 0, nullptr, "r", cands, 1, 1, &winner);
    EATEST_REQUIRE(rc == EATOM_ERR_INVALID_ARG);
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_handle_null_devuelve_error) {
    size_t w = 0;
    const char* x[] = {"a"};
    int rc = eatom_kernel_query_pairs_argmax(
        nullptr, x, 1, x, "a", x, 1, 1, &w);
    EATEST_REQUIRE(rc == EATOM_ERR_NULL);
    double p[1];
    rc = eatom_kernel_query_pairs_probs(
        nullptr, x, 1, x, "a", x, 1, 1, p);
    EATEST_REQUIRE(rc == EATOM_ERR_NULL);
}

EATEST_CASE(c_api_decide_pairs_acepta_y_explica) {
    auto* k = eatom_kernel_create(1024, 99);
    const char* roles[]   = {"color", "tamano"};
    const char* fillers[] = {"rojo",  "grande"};
    const char* cands[]   = {"rojo", "azul", "verde", "grande"};
    int    kind = -1;
    size_t winner = 0, runner = 0;
    double conf = 0, marg = 0, ent = 0, ratio = 0, neff = 0;
    double probs[4] = {0,0,0,0};
    char   text[512] = {0};
    size_t needed = 0;
    int rc = eatom_kernel_decide_pairs(
        k, roles, 2, fillers, "color", cands, 4, /*autoingest*/1,
        /*policy*/nullptr,
        &kind, &winner, &runner, &conf, &marg, &ent, &ratio, &neff,
        probs, text, sizeof(text), &needed);
    EATEST_REQUIRE(rc == EATOM_OK);
    EATEST_REQUIRE(kind == EATOM_DECISION_ACCEPT);
    EATEST_REQUIRE(winner == 0);          // 'rojo' es el primero
    EATEST_REQUIRE(conf > 0.30);
    EATEST_REQUIRE(marg > 0.0);
    EATEST_REQUIRE(needed > 1);
    // La frase contiene el ganador.
    std::string s(text);
    EATEST_REQUIRE(s.find("'rojo'") != std::string::npos);
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_decide_pairs_buffer_pequeno_trunca_pero_no_falla) {
    auto* k = eatom_kernel_create(512, 1);
    const char* roles[]   = {"a"};
    const char* fillers[] = {"x"};
    const char* cands[]   = {"x", "y"};
    char text[8] = {1,1,1,1,1,1,1,1};
    size_t needed = 0;
    int rc = eatom_kernel_decide_pairs(
        k, roles, 1, fillers, "a", cands, 2, 1, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        text, sizeof(text), &needed);
    EATEST_REQUIRE(rc == EATOM_OK);
    EATEST_REQUIRE(text[7] == '\0');     // siempre terminado
    EATEST_REQUIRE(needed > sizeof(text));
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_decide_pairs_policy_strict_puede_abstener) {
    auto* k = eatom_kernel_create(64, 5);
    const char* roles[]   = {"r"};
    const char* fillers[] = {"f"};
    // Consultamos por un rol DISTINTO al ingerido → ruido puro.
    const char* cands[]   = {"f","g","h","i","j","k","l","m"};
    eatom_policy_t pol;
    pol.min_confidence       = 0.50;
    pol.min_margin           = 0.30;
    pol.max_entropy_ratio    = 0.50;
    pol.max_effective_n      = 0.0;
    pol.require_finite_probs = 1;
    int kind = -1;
    int rc = eatom_kernel_decide_pairs(
        k, roles, 1, fillers, /*query_role*/"otro_rol", cands, 8, 1, &pol,
        &kind, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, 0, nullptr);
    EATEST_REQUIRE(rc == EATOM_OK);
    // Consultar por un rol no compuesto → distribución dispersa.
    EATEST_REQUIRE(kind != EATOM_DECISION_ACCEPT);
    eatom_kernel_destroy(k);
}

// ---------------------------------------------------------------------------
// Ladrillo 15 — unbind standalone.
// ---------------------------------------------------------------------------

EATEST_CASE(c_api_unbind_argmax_recupera_partner_exacto) {
    auto* k = eatom_kernel_create(1024, 31337);
    // Inverso EXACTO del bind Hadamard-conjugado:
    //   unbind(bind(insulina, glucosa), insulina) == glucosa
    // El argmax sobre los candidatos debe colapsar a "glucosa" (idx 1).
    const char* cands[] = {"agua", "glucosa", "grasa", "sal"};
    size_t winner = 999;
    int rc = eatom_kernel_unbind_argmax(
        k, "insulina", "glucosa", cands, 4, /*autoingest*/ 1, &winner);
    EATEST_REQUIRE(rc == EATOM_OK);
    EATEST_REQUIRE(winner == 1);

    // Cambiar el partner cambia el ganador (simetría del operador).
    rc = eatom_kernel_unbind_argmax(
        k, "insulina", "agua", cands, 4, 1, &winner);
    EATEST_REQUIRE(rc == EATOM_OK);
    EATEST_REQUIRE(winner == 0);
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_unbind_argmax_handle_null_devuelve_error) {
    const char* cands[] = {"a"};
    size_t winner = 0;
    int rc = eatom_kernel_unbind_argmax(
        nullptr, "k", "p", cands, 1, 1, &winner);
    EATEST_REQUIRE(rc == EATOM_ERR_NULL);
}

EATEST_CASE(c_api_unbind_argmax_n_candidates_cero_lanza) {
    auto* k = eatom_kernel_create(64, 7);
    size_t winner = 0;
    int rc = eatom_kernel_unbind_argmax(
        k, "k", "p", nullptr, 0, 1, &winner);
    EATEST_REQUIRE(rc == EATOM_ERR_INVALID_ARG);
    eatom_kernel_destroy(k);
}

EATEST_CASE(c_api_unbind_argmax_autoingest_off_lanza_si_falta_nombre) {
    auto* k = eatom_kernel_create(64, 9);
    const char* cands[] = {"x"};
    size_t winner = 0;
    int rc = eatom_kernel_unbind_argmax(
        k, "k_no_ingerido", "p_no_ingerido", cands, 1,
        /*autoingest*/ 0, &winner);
    EATEST_REQUIRE(rc == EATOM_ERR_INVALID_ARG);
    eatom_kernel_destroy(k);
}
