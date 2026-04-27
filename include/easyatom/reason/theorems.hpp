// =============================================================================
// EasyAtom · Ladrillo 21 — Generador de teoremas (forward chaining).
// =============================================================================
//
// Dada una base de leyes compiladas {L_i} (L20), genera proposiciones nuevas
// por COMPOSICION SILOGISTICA. No es busqueda estadistica: cada teorema se
// produce por una regla de inferencia exacta sobre la tabla `infer(R1,R2)`.
//
// Reglas (todas con sujeto/objeto encadenados: A -R1- B  ,  B -R2- C  =>  A -R- C):
//
//   transitividad estricta:
//     IsA           + IsA           -> IsA
//     PartOf        + PartOf        -> PartOf
//     LocatedIn     + LocatedIn     -> LocatedIn
//     PrecedesTime  + PrecedesTime  -> PrecedesTime
//     Causes        + Causes        -> Causes
//     Increases     + Increases     -> Increases
//     Decreases     + Decreases     -> Increases     (doble negacion)
//     Increases     + Decreases     -> Decreases
//     Decreases     + Increases     -> Decreases
//     Causes        + Inhibits      -> Inhibits
//     Inhibits      + Causes        -> Inhibits
//
//   substitucion por clase (IsA propaga propiedades):
//     IsA           + R*            -> R*    (para R* en {Causes, Inhibits,
//                                              Increases, Decreases, Treats,
//                                              HasProperty, LocatedIn, PartOf})
//
//   sustitucion por equivalencia:
//     Equivalent    + R             -> R
//     R             + Equivalent    -> R
//
// Las composiciones que NO estan en la tabla devuelven Relation::Unknown y se
// descartan: el motor NO inventa relaciones que no haya respaldado por una
// regla explicita. (0 alucinaciones.)
//
// Salida:
//   * Theorem { Triplet derivada, sources={i,j}, depth }.
//   * derive_theorems(laws, max_depth) ejecuta hasta max_depth iteraciones,
//     en cada una compone las leyes actuales con la base original, y deduplica
//     por (subject, R, object) y por tripletas ya presentes en la base.
//   * compile_theorems(kernel, theorems) materializa cada teorema como
//     CompiledLaw "fresca" usando el mismo apply_relation.
//
// Esto cierra el bucle: el motor produce conocimiento NUEVO sin internet, sin
// LLM, sin RAG. Solo composicion algebraico-simbolica sobre lo ya compilado.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "easyatom/cst/compile.hpp"
#include "easyatom/cst/operator_map.hpp"
#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/qkernel/qkernel.hpp"

namespace easyatom::reason {

using easyatom::cst::CompiledLaw;
using easyatom::cst::Relation;
using easyatom::cst::Triplet;
using easyatom::qkernel::QKernel;

// -----------------------------------------------------------------------------
// Tabla de inferencia.
// -----------------------------------------------------------------------------

[[nodiscard]] inline Relation infer(Relation a, Relation b) noexcept {
    if (a == Relation::Unknown || b == Relation::Unknown) return Relation::Unknown;

    // Equivalent actua como identidad sustitucional por cualquier lado.
    if (a == Relation::Equivalent) return b;
    if (b == Relation::Equivalent) return a;

    // Transitividad estricta del mismo tipo.
    if (a == b) {
        switch (a) {
            case Relation::IsA:          return Relation::IsA;
            case Relation::PartOf:       return Relation::PartOf;
            case Relation::LocatedIn:    return Relation::LocatedIn;
            case Relation::PrecedesTime: return Relation::PrecedesTime;
            case Relation::Causes:       return Relation::Causes;
            case Relation::Increases:    return Relation::Increases;
            case Relation::Decreases:    return Relation::Increases;   // -- = +
            default: break;
        }
    }

    // Mezcla de monotonia.
    if (a == Relation::Increases && b == Relation::Decreases) return Relation::Decreases;
    if (a == Relation::Decreases && b == Relation::Increases) return Relation::Decreases;
    if (a == Relation::Causes    && b == Relation::Inhibits)  return Relation::Inhibits;
    if (a == Relation::Inhibits  && b == Relation::Causes)    return Relation::Inhibits;

    // Substitucion por clase: IsA propaga propiedades del padre al hijo.
    if (a == Relation::IsA) {
        switch (b) {
            case Relation::Causes:
            case Relation::Inhibits:
            case Relation::Increases:
            case Relation::Decreases:
            case Relation::Treats:
            case Relation::HasProperty:
            case Relation::LocatedIn:
            case Relation::PartOf:
                return b;
            default: break;
        }
    }
    return Relation::Unknown;
}

// -----------------------------------------------------------------------------
// Theorem — proposicion derivada con su trazabilidad.
// -----------------------------------------------------------------------------

struct Theorem {
    Triplet                  triplet;
    std::vector<std::size_t> source_indices;   // indices en la base original
    std::uint32_t            depth = 1;
};

// -----------------------------------------------------------------------------
// derive_theorems — forward chaining hasta max_depth.
// -----------------------------------------------------------------------------

namespace detail {

inline std::string key(const Triplet& t) {
    std::string s;
    s.reserve(t.subject.size() + t.object.size() + 8);
    s += t.subject;
    s.push_back('|');
    s += std::to_string(static_cast<int>(t.relation));
    s.push_back('|');
    s += t.object;
    return s;
}

}  // namespace detail

[[nodiscard]] inline std::vector<Theorem> derive_theorems(
    const std::vector<CompiledLaw>& base,
    std::uint32_t                   max_depth = 2) {
    std::vector<Theorem> out;
    if (base.empty() || max_depth == 0) return out;

    std::unordered_set<std::string> known;
    known.reserve(base.size() * 4);
    for (const auto& L : base) known.insert(detail::key(L.triplet));

    // Capa actual: pares (triplet, sources, depth). Empezamos con la base
    // como "depth 0".
    struct Frontier { Triplet t; std::vector<std::size_t> src; std::uint32_t depth; };
    std::vector<Frontier> current;
    current.reserve(base.size());
    for (std::size_t i = 0; i < base.size(); ++i) {
        current.push_back({base[i].triplet, {i}, 0});
    }

    for (std::uint32_t d = 1; d <= max_depth; ++d) {
        std::vector<Frontier> next;
        // Componer cada (frontier) con cada base.
        for (const auto& f : current) {
            for (std::size_t j = 0; j < base.size(); ++j) {
                if (f.t.object != base[j].triplet.subject) continue;
                const Relation r =
                    infer(f.t.relation, base[j].triplet.relation);
                if (r == Relation::Unknown) continue;
                Triplet nt;
                nt.subject  = f.t.subject;
                nt.relation = r;
                nt.object   = base[j].triplet.object;
                if (nt.subject == nt.object) continue;        // sin auto-loops
                const std::string k = detail::key(nt);
                if (known.count(k)) continue;
                known.insert(k);
                std::vector<std::size_t> src = f.src;
                src.push_back(j);
                Theorem th;
                th.triplet = nt;
                th.source_indices = src;
                th.depth = d;
                out.push_back(th);
                next.push_back({nt, std::move(src), d});
            }
        }
        if (next.empty()) break;
        current = std::move(next);
    }
    return out;
}

// -----------------------------------------------------------------------------
// compile_theorems — materializa cada teorema como CompiledLaw.
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::vector<CompiledLaw> compile_theorems(
    QKernel& kernel, const std::vector<Theorem>& theorems) {
    std::vector<CompiledLaw> out;
    out.reserve(theorems.size());
    for (const auto& th : theorems) {
        const auto& S = kernel.ingest(th.triplet.subject);
        const auto& O = kernel.ingest(th.triplet.object);
        const auto  K = easyatom::cst::relation_key(th.triplet.relation,
                                                    kernel.dim());
        CompiledLaw L;
        L.triplet         = th.triplet;
        L.state           = easyatom::cst::apply_relation(
                                th.triplet.relation, S, O, K);
        L.fingerprint     = easyatom::cst::fingerprint(L.state);
        L.provenance_hash = 0;   // teorema derivado: sin texto fuente
        out.push_back(std::move(L));
    }
    return out;
}

}  // namespace easyatom::reason
