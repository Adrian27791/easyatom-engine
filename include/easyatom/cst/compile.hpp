// =============================================================================
// EasyAtom · Ladrillo 20 — CST etapa 3: orquestador compile_law(text).
// =============================================================================
//
// Punto unico de entrada del subsistema simbolico:
//
//   string crudo
//       |
//       v   L16: extract(text) -> Triplet(subject, R, object)
//       |
//       v   QKernel::ingest(subject), ingest(object)  (idempotente)
//       |
//       v   relation_key(R, dim)
//       |
//       v   L17: apply_relation(R, S, O, K_R) -> State law
//       |
//       v   fingerprint  = FNV-1a sobre el contenido binario de `law`
//       |   provenance   = FNV-1a sobre el TEXTO ORIGINAL (que se descarta)
//       |
//       v   CompiledLaw{ triplet, state, fingerprint, provenance_hash }
//
// EL TEXTO ORIGINAL NO SE GUARDA. El motor solo conserva:
//   * la tripleta normalizada (subject, R, object) — necesaria para que L21
//     (teoremas) y L23 (coherencia) puedan razonar simbolicamente,
//   * el estado compilado en H_D — la "ley" en su forma manipulable,
//   * dos hashes — fingerprint del estado y provenance del texto.
//
// Los hashes son la unica conexion con la fuente externa: si dos textos
// distintos compilan al mismo estado, sus fingerprints coinciden y se
// detecta deduplicacion automatica. Si el mismo texto vuelve a entrar, se
// detecta por provenance_hash.
//
// Asi se cumplen las dos directivas operativas:
//   * "no es guardar datos es guardar leyes"  -> guardamos State + Triplet,
//     no el texto.
//   * "0 alucinaciones, sin caer en RAG"      -> nada se recupera del texto:
//     todo razonamiento posterior trabaja sobre el grafo simbolico de
//     CompiledLaw + sus States.
//
// Header-only. Sin dependencias externas.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "easyatom/cst/operator_map.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/qkernel/qkernel.hpp"

namespace easyatom::cst {

using easyatom::hilbert::State;
using easyatom::qkernel::QKernel;

// -----------------------------------------------------------------------------
// FNV-1a 64-bit sobre bytes arbitrarios.
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::uint64_t fnv1a_bytes(const void* data,
                                               std::size_t n) noexcept {
    constexpr std::uint64_t kOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kPrime  = 0x100000001b3ULL;
    std::uint64_t h = kOffset;
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kPrime;
    }
    return h;
}

[[nodiscard]] inline std::uint64_t fnv1a_text(std::string_view s) noexcept {
    return fnv1a_bytes(s.data(), s.size());
}

[[nodiscard]] inline std::uint64_t fingerprint(const State& s) noexcept {
    return fnv1a_bytes(s.amplitudes().data(),
                       s.amplitudes().size() * sizeof(easyatom::hilbert::Complex));
}

// -----------------------------------------------------------------------------
// CompiledLaw — la unidad atomica de conocimiento del motor.
// -----------------------------------------------------------------------------
//
// `state` ES la ley. `triplet` es su forma simbolica para razonar. Los hashes
// son metadatos.

struct CompiledLaw {
    Triplet       triplet;
    State         state;
    std::uint64_t fingerprint     = 0;
    std::uint64_t provenance_hash = 0;
};

// -----------------------------------------------------------------------------
// compile_law(kernel, text) — entrada principal.
// -----------------------------------------------------------------------------

struct CompileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[nodiscard]] inline CompiledLaw compile_law(QKernel& kernel,
                                             std::string_view text) {
    Triplet t = extract(text);
    if (!t.valid()) {
        throw CompileError(
            "compile_law: no se pudo extraer una tripleta declarativa.");
    }
    const State& S   = kernel.ingest(t.subject);
    const State& O   = kernel.ingest(t.object);
    const State  K_R = relation_key(t.relation, kernel.dim());

    CompiledLaw L;
    L.state           = apply_relation(t.relation, S, O, K_R);
    L.triplet         = std::move(t);
    L.fingerprint     = fingerprint(L.state);
    L.provenance_hash = fnv1a_text(text);
    return L;
}

// -----------------------------------------------------------------------------
// compile_corpus(kernel, text) — multiples leyes desde un texto multifrase.
// -----------------------------------------------------------------------------
//
// Aplica L16::extract_all (separadores '.', ';', '\n') y compila cada
// tripleta valida. Las invalidas se omiten silenciosamente (NO se inventan).

[[nodiscard]] inline std::vector<CompiledLaw> compile_corpus(
    QKernel& kernel, std::string_view text) {
    std::vector<CompiledLaw> out;
    for (const Triplet& tt : extract_all(text)) {
        if (!tt.valid()) continue;
        const State& S   = kernel.ingest(tt.subject);
        const State& O   = kernel.ingest(tt.object);
        const State  K_R = relation_key(tt.relation, kernel.dim());
        CompiledLaw L;
        L.state           = apply_relation(tt.relation, S, O, K_R);
        L.triplet         = tt;
        L.fingerprint     = fingerprint(L.state);
        L.provenance_hash = 0;   // texto compuesto: provenance global no aplica
        out.push_back(std::move(L));
    }
    return out;
}

}  // namespace easyatom::cst
