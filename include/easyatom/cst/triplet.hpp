// =============================================================================
// EasyAtom · Ladrillo 16 — Extractor de tripletas (subject, relation, object).
// =============================================================================
//
// Parser RULE-BASED ES/EN que reduce frases simples a tripletas:
//   "la insulina disminuye la glucosa"
//      → Triplet{ subject="insulina", relation=Decreases, object="glucosa" }
//
// Estrategia (cero ML, cero diccionarios masivos, 100% determinista):
//   1. Normalizar: lower-case, retirar acentos, colapsar espacios, strip
//      puntuación.
//   2. Tokenizar por whitespace.
//   3. Filtrar STOP-WORDS funcionales (artículos, preposiciones cortas).
//   4. Recorrer tokens; el primer token cuyo lema esté en el catálogo de
//      verbos define la relación y particiona la frase en pre/post.
//   5. Sujeto = ÚLTIMO token significativo del lado izquierdo (cabeza
//      sintáctica heurística para frases canónicas SVO).
//      Objeto = PRIMER token significativo del lado derecho.
//   6. Si no se halla verbo, devolver Triplet con Relation::Unknown.
//
// Pensado para órdenes de magnitud LATENCIA O(n) sobre n tokens, sin
// diccionarios externos. Cubre el ~80% de proposiciones clínicas
// declarativas simples — suficiente para que el CST tenga "input" sobre
// el que aplicar el resto del pipeline (L17→L20).

#ifndef EASYATOM_CST_TRIPLET_HPP
#define EASYATOM_CST_TRIPLET_HPP

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "easyatom/cst/verbs.hpp"

namespace easyatom::cst {

struct Triplet {
    std::string subject;
    Relation    relation = Relation::Unknown;
    std::string object;

    [[nodiscard]] bool valid() const noexcept {
        return relation != Relation::Unknown
            && !subject.empty()
            && !object.empty();
    }
};

namespace detail {

// Mapa ASCII de acentos comunes ES → equivalente sin acento. Suficiente
// para texto clínico latino; no pretendemos Unicode completo.
[[nodiscard]] inline char strip_accent_byte(unsigned char c) noexcept {
    switch (c) {
        case 0xE1: return 'a'; // á
        case 0xE9: return 'e'; // é
        case 0xED: return 'i'; // í
        case 0xF3: return 'o'; // ó
        case 0xFA: return 'u'; // ú
        case 0xFC: return 'u'; // ü
        case 0xF1: return 'n'; // ñ
        case 0xC1: return 'a'; // Á
        case 0xC9: return 'e';
        case 0xCD: return 'i';
        case 0xD3: return 'o';
        case 0xDA: return 'u';
        case 0xDC: return 'u';
        case 0xD1: return 'n';
        default:   return static_cast<char>(c);
    }
}

[[nodiscard]] inline std::string normalize(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        // ASCII puntuación → espacio.
        if (c < 0x80) {
            if (std::isalnum(c) || c == '-') {
                out.push_back(static_cast<char>(std::tolower(c)));
            } else {
                if (!out.empty() && out.back() != ' ') out.push_back(' ');
            }
        } else {
            // Latin-1 con acento (el 2º byte UTF-8 traducido naïve).
            // Nuestro mapa es Latin-1; UTF-8 multi-byte: si el byte alto
            // pertenece a U+00C0..U+00FF traducimos por byte único.
            char ch = strip_accent_byte(c);
            if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '-') {
                out.push_back(static_cast<char>(std::tolower(
                    static_cast<unsigned char>(ch))));
            } else {
                if (!out.empty() && out.back() != ' ') out.push_back(' ');
            }
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// Stop-words ES + EN que no aportan a la cabeza nominal.
[[nodiscard]] inline const std::unordered_set<std::string>& stopwords() {
    static const std::unordered_set<std::string> kSW = {
        // ES
        "el","la","los","las","un","una","unos","unas",
        "de","del","al","a","en","y","o","u",
        "que","con","por","para",
        // EN
        "the","a","an","of","to","in","on","at","and","or","by","for","that"
    };
    return kSW;
}

[[nodiscard]] inline std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ' ') {
            if (!cur.empty()) { out.push_back(std::move(cur)); cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(std::move(cur));
    return out;
}

}  // namespace detail

/// Extrae una tripleta de una frase declarativa simple. Si no hay verbo
/// reconocido, devuelve un Triplet cuya `relation == Unknown`.
[[nodiscard]] inline Triplet extract(std::string_view sentence) {
    const std::string norm = detail::normalize(sentence);
    auto toks = detail::tokenize(norm);
    if (toks.empty()) return {};

    const auto& sw = detail::stopwords();
    // Encuentra el primer verbo del catálogo.
    std::size_t verb_idx = toks.size();
    Relation rel = Relation::Unknown;
    for (std::size_t i = 0; i < toks.size(); ++i) {
        Relation r = classify_verb(toks[i]);
        if (r != Relation::Unknown) {
            verb_idx = i;
            rel = r;
            break;
        }
    }
    if (rel == Relation::Unknown) return {};

    // Sujeto: último no-stopword a la izquierda del verbo.
    std::string subject;
    for (std::size_t i = verb_idx; i-- > 0;) {
        if (sw.find(toks[i]) == sw.end()) { subject = toks[i]; break; }
    }
    // Objeto: primer no-stopword a la derecha del verbo.
    std::string object;
    for (std::size_t i = verb_idx + 1; i < toks.size(); ++i) {
        if (sw.find(toks[i]) == sw.end()) { object = toks[i]; break; }
    }
    return Triplet{ std::move(subject), rel, std::move(object) };
}

/// Extrae múltiples tripletas separando por puntuación lógica ('.', ';').
/// Las frases sin verbo válido se descartan silenciosamente.
[[nodiscard]] inline std::vector<Triplet> extract_all(std::string_view text) {
    std::vector<Triplet> out;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) {
            Triplet t = extract(cur);
            if (t.valid()) out.push_back(std::move(t));
            cur.clear();
        }
    };
    for (char c : text) {
        if (c == '.' || c == ';' || c == '\n') flush();
        else cur.push_back(c);
    }
    flush();
    return out;
}

}  // namespace easyatom::cst

#endif  // EASYATOM_CST_TRIPLET_HPP
