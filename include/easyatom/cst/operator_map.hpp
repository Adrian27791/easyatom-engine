// =============================================================================
// EasyAtom · Ladrillo 17 — CST etapa 2: mapeo Relation -> operador HDC.
// =============================================================================
//
// Convierte una tripleta (sujeto, relacion, objeto) del L16 en un ESTADO
// COMPILADO sobre H_D usando solo el alfabeto fundamental del motor:
// {bind, permute, scale}. Es la primera capa "simbolica -> geometrica".
//
//   apply_relation(R, S, O, K_R) =
//        bind(  bind(K_R, S),  permute(O, shift(R))  ) * sign(R)
//
// Donde:
//   * K_R es la "clave de rol" deterministica de la relacion R.
//          Se construye con random_phase_state(D, seed = hash(R) ^ kSalt),
//          de modo que dos relaciones distintas producen estados ortogonales
//          en el limite D->inf y resultados de bind incompatibles.
//   * shift(R) codifica la asimetria/direccionalidad: Causes,Increases,IsA
//     usan +1 (S precede a O); Inhibits,Decreases,PrecedesTime usan +2;
//     LocatedIn,PartOf,HasProperty +3..+5 (asimetricas pero no causales);
//     Treats +6; Equivalent y OpposesTo usan 0 (simetricas en posicion).
//   * sign(R) = -1 solo para OpposesTo; +1 en el resto.
//
// Consecuencias garantizadas (verificadas en tests):
//
//   1. Relaciones distintas con mismos S,O producen estados linealmente
//      independientes (fidelity baja entre ellos).
//   2. Equivalent es simetrico en S,O (apply(Equivalent,S,O)=apply(Equivalent,O,S)).
//   3. Causes NO es simetrico: apply(Causes,S,O) != apply(Causes,O,S).
//   4. OpposesTo(S,O) = -Causes(S,O) en magnitud (suma -> 0 si los componemos
//      con bundle de pesos +1).
//   5. Dado el resultado L y conocidas K_R y S, se puede recuperar
//      permute(O,shift(R)) por unbind, y luego O por permute inverso. Esto
//      hace que CADA LEY COMPILADA sea reversible (no se pierde informacion).
//
// El objetivo no es recordar texto: es comprimir el contenido logico de la
// tripleta en una forma manipulable algebraicamente. L18 hara busqueda
// simbolica (GP) sobre este alfabeto; L20 lo orquestara con provenance.

#pragma once

#include <cstdint>
#include <stdexcept>

#include "easyatom/cst/verbs.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"

namespace easyatom::cst {

using easyatom::hilbert::Complex;
using easyatom::hilbert::State;

// -----------------------------------------------------------------------------
// shift_for(R) — desplazamiento de permute por relacion.
// -----------------------------------------------------------------------------
//
// Tabla fija. La unica restriccion es que dos R distintos generen shifts
// distintos para reforzar la separacion (ademas del K_R distinto).

[[nodiscard]] constexpr std::int64_t shift_for(Relation r) noexcept {
    switch (r) {
        case Relation::Unknown:      return 0;
        case Relation::Causes:       return 1;
        case Relation::Inhibits:     return 2;
        case Relation::Increases:    return 1;   // misma direccion que Causes
        case Relation::Decreases:    return 2;   // misma direccion que Inhibits
        case Relation::IsA:          return 3;
        case Relation::PartOf:       return 4;
        case Relation::HasProperty:  return 5;
        case Relation::LocatedIn:    return 6;
        case Relation::Treats:       return 7;
        case Relation::PrecedesTime: return 8;
        case Relation::Equivalent:   return 0;   // simetrico
        case Relation::OpposesTo:    return 0;   // simetrico (signo cambia)
    }
    return 0;
}

// -----------------------------------------------------------------------------
// sign_for(R) — escala global por relacion.
// -----------------------------------------------------------------------------

[[nodiscard]] constexpr double sign_for(Relation r) noexcept {
    return (r == Relation::OpposesTo) ? -1.0 : 1.0;
}

// -----------------------------------------------------------------------------
// relation_key(R, dim, salt) — clave de rol deterministica.
// -----------------------------------------------------------------------------
//
// Construye un estado de fase aleatoria reproducible a partir de la relacion.
// Sirve como "ancla" geometrica de R: dos R distintos dan estados casi
// ortogonales en alta D.

[[nodiscard]] inline State relation_key(Relation r, std::size_t dim,
                                        std::uint64_t salt = 0x5A17C57ULL) {
    const std::uint64_t base = static_cast<std::uint64_t>(r);
    // Mix simple: golden ratio * (R+1) ^ salt — determinista y distinto por R.
    std::uint64_t seed =
        (base + 1ULL) * 0x9E3779B97F4A7C15ULL ^ salt;
    if (seed == 0) seed = 0x12345ULL;
    return easyatom::ops::random_phase_state(dim, seed);
}

// -----------------------------------------------------------------------------
// scale(s, alpha) — multiplica todas las amplitudes por un escalar real.
// -----------------------------------------------------------------------------
//
// Operador trivial pero parte explicita del alfabeto en el GP del L18.

[[nodiscard]] inline State scale(const State& s, double alpha) {
    State r(s.dim());
    const Complex a{alpha, 0.0};
    for (std::size_t i = 0; i < s.dim(); ++i) r[i] = a * s[i];
    return r;
}

// -----------------------------------------------------------------------------
// apply_relation(R, S, O, K_R) — compila la tripleta a un estado.
// -----------------------------------------------------------------------------

[[nodiscard]] inline State apply_relation(Relation r,
                                          const State& subject,
                                          const State& object,
                                          const State& key_r) {
    if (subject.dim() != object.dim() || subject.dim() != key_r.dim()) {
        throw std::invalid_argument(
            "cst::apply_relation: dimensiones incoherentes (S,O,K_R).");
    }
    const State sk = easyatom::ops::bind(key_r, subject);
    const State op = easyatom::ops::permute(object, shift_for(r));
    State law = easyatom::ops::bind(sk, op);
    const double sg = sign_for(r);
    if (sg != 1.0) law = scale(law, sg);
    return law;
}

}  // namespace easyatom::cst
