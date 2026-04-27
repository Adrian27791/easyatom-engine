// =============================================================================
// easyatom/ingest/external.hpp  --  L33
//
// Cola de pendientes para ingesta externa autorizada.
//
//   - El motor detecta una laguna (L24/L32) y encola un PendingQuery.
//   - Cuando hay autorizacion del usuario (Permission::Always o callback
//     authorize(topic) -> bool), drain() llama al fetcher externo (HTTP a
//     fuentes academicas/cientificas, lo provee la app), pasa cada texto
//     resultante a on_text(text) para que la app lo compile (compile_law)
//     y lo valide (L23/L26) antes de acceptarlo.
//
// Este header NO hace red ni IO. Es el orquestador puro de la cola y el
// chequeo de permiso. El adaptador HTTP vive en la capa TS/RN o JNI, fuera
// del motor.
// =============================================================================

#ifndef EASYATOM_INGEST_EXTERNAL_HPP
#define EASYATOM_INGEST_EXTERNAL_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace easyatom::ingest {

enum class Permission : std::uint8_t {
    Ask    = 0,   // por cada item llamamos authorize(topic)
    Always = 1,   // se procesa todo sin preguntar
    Never  = 2,   // nada se procesa, los items quedan en cola
};

struct PendingQuery {
    std::uint64_t id    = 0;
    std::string   topic;        // lo que el motor quiere aprender
};

struct DrainReport {
    std::size_t considered = 0;   // items vistos en este drain
    std::size_t authorized = 0;   // pasaron el chequeo de permiso
    std::size_t fetched    = 0;   // texts devueltos por el fetcher
    std::size_t ingested   = 0;   // on_text() llamado sobre cada uno
    std::size_t skipped    = 0;   // permission denied / dejados en cola
};

class IngestQueue {
public:
    IngestQueue() = default;

    std::uint64_t enqueue(std::string topic) {
        if (topic.empty())
            throw std::invalid_argument("IngestQueue::enqueue: topic vacio.");
        const std::uint64_t id = ++next_id_;
        pending_.push_back(PendingQuery{id, std::move(topic)});
        return id;
    }

    [[nodiscard]] const std::vector<PendingQuery>& pending() const noexcept {
        return pending_;
    }

    [[nodiscard]] std::size_t size() const noexcept { return pending_.size(); }

    void clear() noexcept { pending_.clear(); }

    // drain procesa los items en orden FIFO.
    //   perm:      politica de permiso global
    //   authorize: bool(const std::string& topic) — solo se invoca con perm==Ask
    //   fetcher:   std::vector<std::string>(const std::string& topic) — sin red
    //              en este header; lo conecta la capa exterior
    //   on_text:   void(const std::string& text) — se llama por cada documento
    //
    // Los items NO autorizados se preservan en la cola; los autorizados se
    // remueven (incluso si el fetcher devuelve vacio: el intento ya se hizo).
    template <class Authorize, class Fetcher, class OnText>
    DrainReport drain(Permission perm,
                      Authorize&& authorize,
                      Fetcher&&   fetcher,
                      OnText&&    on_text) {
        DrainReport rep{};
        std::vector<PendingQuery> kept;
        kept.reserve(pending_.size());

        for (auto& pq : pending_) {
            ++rep.considered;
            bool ok = false;
            switch (perm) {
                case Permission::Always: ok = true; break;
                case Permission::Never:  ok = false; break;
                case Permission::Ask:    ok = static_cast<bool>(
                                              authorize(pq.topic)); break;
            }
            if (!ok) {
                ++rep.skipped;
                kept.push_back(std::move(pq));
                continue;
            }
            ++rep.authorized;
            const auto texts = fetcher(pq.topic);
            rep.fetched += texts.size();
            for (const auto& t : texts) {
                on_text(t);
                ++rep.ingested;
            }
        }
        pending_ = std::move(kept);
        return rep;
    }

private:
    std::vector<PendingQuery> pending_;
    std::uint64_t             next_id_ = 0;
};

}  // namespace easyatom::ingest

#endif  // EASYATOM_INGEST_EXTERNAL_HPP
