// =============================================================================
// tests/test_facade.cpp  --  L34 fachada de aprendizaje (alto nivel)
// =============================================================================

#include <stdexcept>
#include <string>
#include <vector>

#include "easyatom/facade/learn.hpp"
#include "test_framework.hpp"

using easyatom::facade::ingest_texts;
using easyatom::facade::LearnSession;
using easyatom::facade::detect_and_enqueue;
using easyatom::facade::learn_externally;
using easyatom::facade::learn_locally;
using easyatom::ingest::Permission;
using easyatom::qkernel::QKernel;

EATEST_CASE(facade_ingest_texts_acepta_primera_ley) {
    QKernel k(64, 42);
    LearnSession s; s.kernel = &k;
    auto rep = ingest_texts(s, {"insulina causa baja-glucosa"});
    EATEST_REQUIRE(rep.total == 1);
    EATEST_REQUIRE(rep.compiled == 1);
    EATEST_REQUIRE(rep.accepted == 1);
    EATEST_REQUIRE(s.codebook.size() == 1);
}

EATEST_CASE(facade_ingest_texts_falla_compile_se_cuenta) {
    QKernel k(64, 42);
    LearnSession s; s.kernel = &k;
    // Texto sin verbo reconocible -> CompileError -> failed.
    auto rep = ingest_texts(s, {"xyzzy plugh quux"});
    EATEST_REQUIRE(rep.failed == 1);
    EATEST_REQUIRE(rep.accepted == 0);
}

EATEST_CASE(facade_ingest_texts_session_sin_kernel_lanza) {
    LearnSession s;
    bool t = false;
    try { (void)ingest_texts(s, {"x is y"}); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(facade_detect_and_enqueue_encola_gaps) {
    QKernel k(64, 42);
    LearnSession s; s.kernel = &k;
    (void)ingest_texts(s, {"perro es mamifero"});
    EATEST_REQUIRE(s.codebook.size() == 1);

    // Probe ortogonal -> density baja -> gap.
    easyatom::hilbert::State probe(64);
    probe[7] = easyatom::hilbert::Complex{1.0, 0.0};
    auto added = detect_and_enqueue(s, {probe}, {"genoma humano"}, 0.9);
    EATEST_REQUIRE(added == 1);
    EATEST_REQUIRE(s.queue.size() == 1);
}

EATEST_CASE(facade_detect_and_enqueue_tamanos_distintos_lanza) {
    QKernel k(64, 42);
    LearnSession s; s.kernel = &k;
    bool t = false;
    try {
        (void)detect_and_enqueue(s, {easyatom::hilbert::State(64)}, {}, 0.5);
    } catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(facade_learn_externally_never_no_consume) {
    QKernel k(64, 42);
    LearnSession s; s.kernel = &k;
    s.queue.enqueue("topic-a");
    auto rep = learn_externally(s, Permission::Never,
        [](const std::string&){ return true; },
        [](const std::string&){ return std::vector<std::string>{}; });
    EATEST_REQUIRE(rep.skipped == 1);
    EATEST_REQUIRE(s.queue.size() == 1);
}

EATEST_CASE(facade_learn_externally_always_pasa_text_a_ingest) {
    QKernel k(64, 42);
    LearnSession s; s.kernel = &k;
    s.queue.enqueue("topic-a");
    auto rep = learn_externally(s, Permission::Always,
        [](const std::string&){ return false; },
        [](const std::string&){
            return std::vector<std::string>{"insulina causa baja-glucosa"};
        });
    EATEST_REQUIRE(rep.authorized == 1);
    EATEST_REQUIRE(rep.ingested == 1);
    // Y la ley debio entrar al codebook (compile + coherence aceptan
    // cuerpo vacio).
    EATEST_REQUIRE(s.codebook.size() == 1);
}

EATEST_CASE(facade_learn_locally_invoca_autoloop_y_puede_crecer) {
    QKernel k(64, 42);
    LearnSession s; s.kernel = &k;
    (void)ingest_texts(s, {"perro es mamifero", "gato es mamifero"});
    const std::size_t before = s.codebook.size();
    easyatom::hilbert::State q(64);
    q[3] = easyatom::hilbert::Complex{1.0, 0.0};
    easyatom::autoloop::LoopConfig cfg;
    cfg.gap_theta = 0.5; cfg.max_iters = 2;
    auto rep = learn_locally(s, {q}, cfg);
    EATEST_REQUIRE(s.codebook.size() == before + rep.accepted);
}
