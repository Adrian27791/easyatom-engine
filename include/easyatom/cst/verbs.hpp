// =============================================================================
// EasyAtom · Ladrillo 16 — Catálogo de verbos canónicos del CST.
// =============================================================================
//
// El "Compilador Simbólico-Topológico" (CST) parte de texto natural y produce
// LEYES (operadores parametrizados) en lugar de embeddings. Su primera etapa
// es identificar el VERBO RELACIONAL que une dos conceptos.
//
// En vez de aceptar el lenguaje completo (intratable), reducimos cualquier
// verbo a uno de ~30 verbos canónicos agrupados por familia de relación.
// Inspirado en UMLS Semantic Network (54 relaciones) y SNOMED-CT.
//
// Cada Relation define qué FAMILIA de operador HDC instanciará el L17
// (`operator_map`). El parser del L16 (`triplet`) sólo reconoce el verbo;
// es el L17 el que decide cómo se ejecuta sobre el codebook.
//
// Sin dependencias externas. Datos como constexpr arrays.

#ifndef EASYATOM_CST_VERBS_HPP
#define EASYATOM_CST_VERBS_HPP

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

namespace easyatom::cst {

/// Familia de relación. Determina la familia de operador HDC en L17.
enum class Relation {
    Unknown      = 0,
    Causes       = 1,   // X causa Y          (insulina causa baja-glucosa)
    Inhibits     = 2,   // X inhibe Y         (antibiótico inhibe bacteria)
    Increases    = 3,   // X aumenta Y        (ejercicio aumenta pulso)
    Decreases    = 4,   // X disminuye Y      (insulina disminuye glucosa)
    IsA          = 5,   // X es-un Y          (perro es-un mamífero)
    PartOf       = 6,   // X parte-de Y       (corazón parte-de cuerpo)
    HasProperty  = 7,   // X tiene Y          (limón tiene ácido)
    LocatedIn    = 8,   // X está-en Y        (riñón está-en abdomen)
    Treats       = 9,   // X trata Y          (paracetamol trata fiebre)
    PrecedesTime = 10,  // X precede Y        (síntoma precede diagnóstico)
    Equivalent   = 11,  // X equivale Y       (H2O equivale agua)
    OpposesTo    = 12   // X opuesto Y        (frío opuesto calor)
};

[[nodiscard]] inline std::string_view relation_name(Relation r) noexcept {
    switch (r) {
        case Relation::Causes:       return "causes";
        case Relation::Inhibits:     return "inhibits";
        case Relation::Increases:    return "increases";
        case Relation::Decreases:    return "decreases";
        case Relation::IsA:          return "is_a";
        case Relation::PartOf:       return "part_of";
        case Relation::HasProperty:  return "has_property";
        case Relation::LocatedIn:    return "located_in";
        case Relation::Treats:       return "treats";
        case Relation::PrecedesTime: return "precedes";
        case Relation::Equivalent:   return "equivalent";
        case Relation::OpposesTo:    return "opposes";
        default:                     return "unknown";
    }
}

namespace detail {

// Lista plana lema → relación. Se usa para construir un map al primer uso.
// Lemas en minúsculas, sin acentos (el normalizador del L16 los retira).
struct VerbEntry { const char* lemma; Relation rel; };

inline constexpr VerbEntry kVerbCatalog[] = {
    // ES — Causes
    {"causa", Relation::Causes}, {"provoca", Relation::Causes},
    {"genera", Relation::Causes}, {"produce", Relation::Causes},
    // EN — Causes
    {"causes", Relation::Causes}, {"triggers", Relation::Causes},
    {"produces", Relation::Causes},

    // ES — Inhibits
    {"inhibe", Relation::Inhibits}, {"bloquea", Relation::Inhibits},
    {"frena", Relation::Inhibits},
    // EN — Inhibits
    {"inhibits", Relation::Inhibits}, {"blocks", Relation::Inhibits},

    // ES — Increases
    {"aumenta", Relation::Increases}, {"sube", Relation::Increases},
    {"eleva", Relation::Increases}, {"incrementa", Relation::Increases},
    // EN — Increases
    {"increases", Relation::Increases}, {"raises", Relation::Increases},
    {"elevates", Relation::Increases},

    // ES — Decreases
    {"disminuye", Relation::Decreases}, {"baja", Relation::Decreases},
    {"reduce", Relation::Decreases}, {"decrece", Relation::Decreases},
    // EN — Decreases
    {"decreases", Relation::Decreases}, {"lowers", Relation::Decreases},
    {"reduces", Relation::Decreases},

    // ES — IsA
    {"es", Relation::IsA}, {"son", Relation::IsA},
    // EN — IsA
    {"is", Relation::IsA}, {"are", Relation::IsA},

    // ES — PartOf
    {"contiene", Relation::PartOf}, {"compone", Relation::PartOf},
    // EN — PartOf
    {"contains", Relation::PartOf}, {"comprises", Relation::PartOf},

    // ES — HasProperty
    {"tiene", Relation::HasProperty}, {"posee", Relation::HasProperty},
    // EN — HasProperty
    {"has", Relation::HasProperty}, {"have", Relation::HasProperty},

    // ES — LocatedIn
    {"esta-en", Relation::LocatedIn}, {"reside", Relation::LocatedIn},
    // EN — LocatedIn
    {"in", Relation::LocatedIn}, {"located", Relation::LocatedIn},

    // ES — Treats
    {"trata", Relation::Treats}, {"cura", Relation::Treats},
    {"alivia", Relation::Treats},
    // EN — Treats
    {"treats", Relation::Treats}, {"cures", Relation::Treats},

    // ES — PrecedesTime
    {"precede", Relation::PrecedesTime}, {"antes", Relation::PrecedesTime},
    // EN — PrecedesTime
    {"precedes", Relation::PrecedesTime}, {"before", Relation::PrecedesTime},

    // ES — Equivalent
    {"equivale", Relation::Equivalent}, {"igual", Relation::Equivalent},
    // EN — Equivalent
    {"equals", Relation::Equivalent}, {"equivalent", Relation::Equivalent},

    // ES — OpposesTo
    {"opone", Relation::OpposesTo}, {"contradice", Relation::OpposesTo},
    // EN — OpposesTo
    {"opposes", Relation::OpposesTo}, {"contradicts", Relation::OpposesTo}
};

}  // namespace detail

/// Lookup verbo→relación. O(1) tras la primera llamada (mapa estático).
[[nodiscard]] inline Relation classify_verb(std::string_view lemma) {
    static const std::unordered_map<std::string, Relation> kMap = []{
        std::unordered_map<std::string, Relation> m;
        for (const auto& e : detail::kVerbCatalog) {
            m.emplace(std::string(e.lemma), e.rel);
        }
        return m;
    }();
    auto it = kMap.find(std::string(lemma));
    return (it == kMap.end()) ? Relation::Unknown : it->second;
}

/// Total de lemas en el catálogo (útil para tests / introspección).
[[nodiscard]] inline std::size_t catalog_size() noexcept {
    return sizeof(detail::kVerbCatalog) / sizeof(detail::VerbEntry);
}

}  // namespace easyatom::cst

#endif  // EASYATOM_CST_VERBS_HPP
