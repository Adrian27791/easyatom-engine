// =============================================================================
// tests/test_c_api_learn.cpp  --  L39
//
// Verifica el C ABI de aprendizaje (eatom_learn_*). El proposito es asegurar
// que el bridge JNI ve una superficie estable y libre de excepciones C++.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "easyatom/c_api.h"
#include "easyatom/c_api_learn.h"
#include "test_framework.hpp"

EATEST_CASE(capi_learn_create_destroy) {
    auto* s = eatom_learn_create(64, 0xC0FFEEULL);
    EATEST_REQUIRE(s != nullptr);
    EATEST_REQUIRE(eatom_learn_codebook_size(s) == 0u);
    EATEST_REQUIRE(eatom_learn_pending_count(s) == 0u);
    eatom_learn_destroy(s);
}

EATEST_CASE(capi_learn_create_dim_cero_es_null) {
    EATEST_REQUIRE(eatom_learn_create(0, 0) == nullptr);
}

EATEST_CASE(capi_learn_destroy_null_no_explota) {
    eatom_learn_destroy(nullptr);
    EATEST_REQUIRE(true);
}

EATEST_CASE(capi_learn_codebook_size_null_es_cero) {
    EATEST_REQUIRE(eatom_learn_codebook_size(nullptr) == 0u);
    EATEST_REQUIRE(eatom_learn_pending_count(nullptr) == 0u);
}

EATEST_CASE(capi_learn_ingest_textos_vacios_no_explota) {
    auto* s = eatom_learn_create(64, 1);
    size_t total = 99, compiled = 99, accepted = 99, rejected = 99, failed = 99;
    int rc = eatom_learn_ingest_texts(s, nullptr, 0,
                                       &total, &compiled,
                                       &accepted, &rejected, &failed);
    EATEST_REQUIRE(rc == EATOM_OK);
    EATEST_REQUIRE(total == 0u);
    EATEST_REQUIRE(compiled == 0u);
    EATEST_REQUIRE(accepted == 0u);
    eatom_learn_destroy(s);
}

EATEST_CASE(capi_learn_ingest_un_texto_intenta_compilar) {
    auto* s = eatom_learn_create(64, 7);
    const char* texts[] = { "aspirina causa alivio." };
    size_t total = 0, compiled = 0, accepted = 0, rejected = 0, failed = 0;
    int rc = eatom_learn_ingest_texts(s, texts, 1,
                                       &total, &compiled,
                                       &accepted, &rejected, &failed);
    EATEST_REQUIRE(rc == EATOM_OK);
    EATEST_REQUIRE(total == 1u);
    // No exigimos accepted == 1 (compile_law puede fallar segun parser); solo
    // que la suma de outcomes sea coherente con total.
    EATEST_REQUIRE(compiled + failed == total);
    EATEST_REQUIRE(accepted + rejected == compiled);
    eatom_learn_destroy(s);
}

EATEST_CASE(capi_learn_detect_topics_y_encolar) {
    auto* s = eatom_learn_create(64, 11);
    const char* topics[] = { "neuroplasticidad", "criotermia", "hipoxia" };
    size_t enq = 0;
    int rc = eatom_learn_detect_topics_and_enqueue(
        s, topics, 3, /*theta=*/0.99, &enq);
    EATEST_REQUIRE(rc == EATOM_OK);
    // Codebook vacio => density==0 para todos => todos se encolan.
    EATEST_REQUIRE(enq == 3u);
    EATEST_REQUIRE(eatom_learn_pending_count(s) == 3u);
    eatom_learn_destroy(s);
}

EATEST_CASE(capi_learn_detect_theta_negativo_es_invalid_arg) {
    auto* s = eatom_learn_create(64, 12);
    size_t enq = 0;
    int rc = eatom_learn_detect_topics_and_enqueue(
        s, nullptr, 0, -0.1, &enq);
    EATEST_REQUIRE(rc == EATOM_ERR_INVALID_ARG);
    eatom_learn_destroy(s);
}

EATEST_CASE(capi_learn_run_autoloop_sin_codebook_no_explota) {
    auto* s = eatom_learn_create(64, 13);
    const char* qs[] = { "alpha", "beta" };
    size_t it = 0, gd = 0, prop = 0, acc = 0, rej = 0;
    int rc = eatom_learn_run_autoloop(
        s, qs, 2, /*gap_theta=*/0.5, /*k_top=*/3, /*max_iters=*/2,
        &it, &gd, &prop, &acc, &rej);
    EATEST_REQUIRE(rc == EATOM_OK);
    EATEST_REQUIRE(it >= 1u);
    eatom_learn_destroy(s);
}

EATEST_CASE(capi_learn_run_autoloop_max_iters_cero_invalid) {
    auto* s = eatom_learn_create(64, 14);
    int rc = eatom_learn_run_autoloop(
        s, nullptr, 0, 0.5, 3, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
    EATEST_REQUIRE(rc == EATOM_ERR_INVALID_ARG);
    eatom_learn_destroy(s);
}

EATEST_CASE(capi_learn_run_autoloop_ranked_modos_no_explotan) {
    auto* s = eatom_learn_create(64, 15);
    const char* qs[] = { "x", "y" };
    size_t it = 0, gd = 0, ct = 0, cu = 0, acc = 0, rej = 0;
    int rc1 = eatom_learn_run_autoloop_ranked(
        s, qs, 2, 0.5, /*k_min=*/1, /*k_max=*/3, /*max_iters=*/2, /*mode=*/0,
        &it, &gd, &ct, &cu, &acc, &rej);
    EATEST_REQUIRE(rc1 == EATOM_OK);
    int rc2 = eatom_learn_run_autoloop_ranked(
        s, qs, 2, 0.5, 1, 3, 2, /*mode=*/1,
        &it, &gd, &ct, &cu, &acc, &rej);
    EATEST_REQUIRE(rc2 == EATOM_OK);
    eatom_learn_destroy(s);
}

EATEST_CASE(capi_learn_run_autoloop_ranked_kmax_menor_kmin_invalid) {
    auto* s = eatom_learn_create(64, 16);
    int rc = eatom_learn_run_autoloop_ranked(
        s, nullptr, 0, 0.5, /*k_min=*/4, /*k_max=*/2, 2, 0,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    EATEST_REQUIRE(rc == EATOM_ERR_INVALID_ARG);
    eatom_learn_destroy(s);
}

EATEST_CASE(capi_learn_handles_son_independientes) {
    auto* s1 = eatom_learn_create(32, 100);
    auto* s2 = eatom_learn_create(32, 200);
    EATEST_REQUIRE(s1 != nullptr && s2 != nullptr && s1 != s2);
    const char* topics[] = { "abc" };
    (void)eatom_learn_detect_topics_and_enqueue(s1, topics, 1, 0.99, nullptr);
    EATEST_REQUIRE(eatom_learn_pending_count(s1) == 1u);
    EATEST_REQUIRE(eatom_learn_pending_count(s2) == 0u);
    eatom_learn_destroy(s1);
    eatom_learn_destroy(s2);
}
