// =============================================================================
// tests/test_ingest.cpp  --  L33 cola de ingesta autorizada
// =============================================================================

#include <stdexcept>
#include <string>
#include <vector>

#include "easyatom/ingest/external.hpp"
#include "test_framework.hpp"

using easyatom::ingest::DrainReport;
using easyatom::ingest::IngestQueue;
using easyatom::ingest::Permission;

EATEST_CASE(ingest_enqueue_aumenta_size) {
    IngestQueue q;
    EATEST_REQUIRE(q.size() == 0);
    auto id1 = q.enqueue("kalman filter");
    auto id2 = q.enqueue("rao blackwell");
    EATEST_REQUIRE(q.size() == 2);
    EATEST_REQUIRE(id1 != id2);
}

EATEST_CASE(ingest_enqueue_topic_vacio_lanza) {
    IngestQueue q;
    bool t = false;
    try { (void)q.enqueue(""); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(ingest_drain_never_no_consume_pero_skipea) {
    IngestQueue q;
    q.enqueue("a"); q.enqueue("b");
    int fetched_calls = 0;
    auto rep = q.drain(Permission::Never,
        [](const std::string&){ return true; },
        [&](const std::string&){ ++fetched_calls;
                                 return std::vector<std::string>{}; },
        [](const std::string&){});
    EATEST_REQUIRE(rep.considered == 2);
    EATEST_REQUIRE(rep.authorized == 0);
    EATEST_REQUIRE(rep.skipped == 2);
    EATEST_REQUIRE(fetched_calls == 0);
    EATEST_REQUIRE(q.size() == 2);  // se preservan
}

EATEST_CASE(ingest_drain_always_procesa_todo) {
    IngestQueue q;
    q.enqueue("doi:10.1/x"); q.enqueue("arxiv:2401.0001");
    int ingested = 0;
    auto rep = q.drain(Permission::Always,
        [](const std::string&){ return false; },  // ignorado
        [](const std::string&){ return std::vector<std::string>{"abstract"}; },
        [&](const std::string&){ ++ingested; });
    EATEST_REQUIRE(rep.authorized == 2);
    EATEST_REQUIRE(rep.fetched == 2);
    EATEST_REQUIRE(rep.ingested == 2);
    EATEST_REQUIRE(ingested == 2);
    EATEST_REQUIRE(q.size() == 0);  // todos consumidos
}

EATEST_CASE(ingest_drain_ask_filtra_por_authorize) {
    IngestQueue q;
    q.enqueue("medline:1"); q.enqueue("blogspam:2"); q.enqueue("crossref:3");
    auto allow = [](const std::string& topic) {
        return topic.rfind("blogspam:", 0) != 0;   // bloquea blogs
    };
    auto rep = q.drain(Permission::Ask,
        allow,
        [](const std::string&){ return std::vector<std::string>{"t"}; },
        [](const std::string&){});
    EATEST_REQUIRE(rep.authorized == 2);
    EATEST_REQUIRE(rep.skipped == 1);
    EATEST_REQUIRE(q.size() == 1);
    EATEST_REQUIRE(q.pending()[0].topic.rfind("blogspam:", 0) == 0);
}

EATEST_CASE(ingest_fetcher_vacio_marca_authorized_sin_ingested) {
    IngestQueue q;
    q.enqueue("topic");
    auto rep = q.drain(Permission::Always,
        [](const std::string&){ return true; },
        [](const std::string&){ return std::vector<std::string>{}; },
        [](const std::string&){});
    EATEST_REQUIRE(rep.authorized == 1);
    EATEST_REQUIRE(rep.fetched == 0);
    EATEST_REQUIRE(rep.ingested == 0);
    EATEST_REQUIRE(q.size() == 0);  // intentado, removido igual
}

EATEST_CASE(ingest_clear_vacia_la_cola) {
    IngestQueue q;
    q.enqueue("a"); q.enqueue("b");
    q.clear();
    EATEST_REQUIRE(q.size() == 0);
}

EATEST_CASE(ingest_ids_son_estrictamente_crecientes) {
    IngestQueue q;
    auto a = q.enqueue("x");
    auto b = q.enqueue("y");
    auto c = q.enqueue("z");
    EATEST_REQUIRE(a < b);
    EATEST_REQUIRE(b < c);
}
