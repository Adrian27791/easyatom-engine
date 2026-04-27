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
