// =============================================================================
// EasyAtom · Ladrillo 2 — Operadores fundamentales del Q-Kernel.
// =============================================================================
//
// Aquí se define el VOCABULARIO COMPOSICIONAL del motor. Cada operador es una
// función bien definida sobre H_D (el espacio del Ladrillo 1) con propiedades
// algebraicas verificables.
//
//   bind(a, b)     ─ compone dos conceptos en una unidad inseparable.
//                    Implementado como producto Hadamard-circular (convolución
//                    circular en frecuencia → multiplicación punto a punto en
//                    el "anillo de fase"). Conmutativo.
//                    Tiene inverso exacto: unbind(bind(a,b), a) = b.
//
//   bundle(...)    ─ superposición ponderada normalizada de varios estados.
//                    Es la "OR semántica": el resultado es similar a TODOS los
//                    operandos a la vez.
//
//   permute(a, k)  ─ rotación cíclica (shift) por k posiciones.
//                    Codifica orden / posición / tiempo. Es un operador
//                    UNITARIO: preserva la norma y la inversa es permute(a, -k).
//
//   unbind(c, a)   ─ recupera b dado c = bind(a, b). Inverso exacto de bind.
//
//   project(s, e)  ─ proyección ortogonal de s sobre la dirección e
//                    (re-exportada del Ladrillo 1).
//
// PROPIEDADES MATEMÁTICAS GARANTIZADAS (verificadas en tests):
//
//   1. bind es conmutativo:               bind(a,b) = bind(b,a)
//   2. bind es asociativo:                bind(bind(a,b),c) = bind(a,bind(b,c))
//   3. bind tiene inverso exacto:         unbind(bind(a,b), a) ≈ b
//   4. permute es unitario:               ||permute(a,k)|| = ||a||
//   5. permute es invertible:             permute(permute(a,k),-k) = a
//   6. permute(N, k=N) es identidad
//   7. bundle es la suma de los inputs (sin normalización forzada),
//      la similitud cosenoidal con cada input es alta.
//
// La construcción "Hadamard de fase" es la versión canónica de bind en HDC
// continuo (Plate, "Holographic Reduced Representations", 1995):
//   - Codificamos cada estado en su forma exponencial e^{iφ_k}.
//   - bind(a,b) = a ⊙ b (producto coordenada a coordenada).
//   - Esto es matemáticamente equivalente a la convolución circular en el
//     dominio del tiempo, vía DFT.
//
// La diferencia con HDC clásico binario:
//   - Allí: vectores binarios + XOR → motor combinatorio rígido.
//   - Aquí: vectores complejos + Hadamard → motor diferenciable, unitario,
//     compatible con rotores y operadores cuánticos del Ladrillo 5.
//
// Esto es lo que hace al motor compatible con Clifford (Ladrillo 0) y con
// Koopman/dinámica (Ladrillo 5 futuro). El HDC binario clásico no lo era.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "easyatom/hilbert/state.hpp"

namespace easyatom::ops {

using easyatom::hilbert::Complex;
using easyatom::hilbert::State;

// -----------------------------------------------------------------------------
// bind — producto Hadamard (coordenada a coordenada).
// -----------------------------------------------------------------------------
//
// Para a = (a_0, ..., a_{D-1}) y b = (b_0, ..., b_{D-1}):
//   bind(a, b)_k = a_k * b_k
//
// Propiedades inmediatas:
//   * conmutativo y asociativo (multiplicación de complejos lo es).
//   * tiene inverso punto a punto (unbind), siempre que ningún a_k = 0.
//
// Por eso, los estados que se quieran usar como CLAVES de bind se generan
// con amplitudes en el círculo unidad (e^{iφ}, |a_k|=1). En esa banda, el
// inverso es trivial:  a_k^{-1} = conj(a_k).

[[nodiscard]] inline State bind(const State& a, const State& b) {
    if (a.dim() != b.dim()) {
        throw std::invalid_argument("ops::bind: dimensiones distintas.");
    }
    State r(a.dim());
    for (std::size_t i = 0; i < a.dim(); ++i) {
        r[i] = a[i] * b[i];
    }
    return r;
}

// -----------------------------------------------------------------------------
// unbind — inverso aproximado de bind cuando |a_k| > 0.
// -----------------------------------------------------------------------------
//
//   c = bind(a, b)       =>  c_k = a_k * b_k
//   unbind(c, a)_k       =   c_k * conj(a_k) / |a_k|^2
//
// Si a_k está en el círculo unidad, |a_k|^2 = 1 y la operación se reduce a
// c_k * conj(a_k). El resultado es exactamente b.
//
// Si algún a_k es 0, no hay inverso: lanzamos. (No alucinar por construcción.)

[[nodiscard]] inline State unbind(const State& c, const State& a,
                                  double eps = 1e-30) {
    if (c.dim() != a.dim()) {
        throw std::invalid_argument("ops::unbind: dimensiones distintas.");
    }
    State r(a.dim());
    for (std::size_t i = 0; i < a.dim(); ++i) {
        const double mag2 = std::norm(a[i]);
        if (mag2 < eps) {
            throw std::domain_error(
                "ops::unbind: amplitud nula en a; clave no invertible.");
        }
        r[i] = c[i] * std::conj(a[i]) / Complex{mag2, 0.0};
    }
    return r;
}

// -----------------------------------------------------------------------------
// bundle — superposición ponderada (sin normalización forzada).
// -----------------------------------------------------------------------------
//
//   bundle({s_k}, {w_k})_i = sum_k w_k * s_k[i]
//
// La similitud (vía fidelity) entre el bundle y cualquier s_k es típicamente
// alta. Este operador es la "OR semántica" del motor: representa "esto y
// aquello al mismo tiempo".

[[nodiscard]] inline State bundle(const std::vector<State>& states,
                                  const std::vector<Complex>& weights) {
    if (states.empty()) {
        throw std::invalid_argument("ops::bundle: lista vacía.");
    }
    if (states.size() != weights.size()) {
        throw std::invalid_argument("ops::bundle: tamaños incoherentes.");
    }
    const std::size_t D = states.front().dim();
    State r(D);
    for (std::size_t k = 0; k < states.size(); ++k) {
        if (states[k].dim() != D) {
            throw std::invalid_argument(
                "ops::bundle: estados con dimensiones distintas.");
        }
        for (std::size_t i = 0; i < D; ++i) {
            r[i] += weights[k] * states[k][i];
        }
    }
    return r;
}

/// Conveniencia: bundle con pesos uniformes 1.
[[nodiscard]] inline State bundle(const std::vector<State>& states) {
    if (states.empty()) {
        throw std::invalid_argument("ops::bundle: lista vacía.");
    }
    std::vector<Complex> w(states.size(), Complex{1.0, 0.0});
    return bundle(states, w);
}

// -----------------------------------------------------------------------------
// permute — rotación cíclica de las amplitudes.
// -----------------------------------------------------------------------------
//
//   permute(a, k)_i = a_{(i - k) mod D}
//
// Equivale a un shift circular: el contenido se mueve k posiciones a la
// derecha. Es un operador UNITARIO (preserva la norma) e INVERTIBLE
// (permute(., -k) lo deshace).
//
// Sirve para codificar ORDEN, POSICIÓN, TIEMPO. Por ejemplo, una secuencia
// (a, b, c) puede codificarse como bundle({permute(a,0), permute(b,1),
// permute(c,2)}) y el orden queda preservado mediante la posición de cada
// elemento.

[[nodiscard]] inline State permute(const State& a, std::int64_t k) {
    const std::size_t D = a.dim();
    if (D == 0) throw std::invalid_argument("ops::permute: dim 0.");
    State r(D);
    // Normalizar k al rango [0, D).
    std::int64_t kk = k % static_cast<std::int64_t>(D);
    if (kk < 0) kk += static_cast<std::int64_t>(D);
    const std::size_t shift = static_cast<std::size_t>(kk);
    for (std::size_t i = 0; i < D; ++i) {
        const std::size_t src = (i + D - shift) % D;
        r[i] = a[src];
    }
    return r;
}

// -----------------------------------------------------------------------------
// random_phase_state — clave aleatoria con |a_k|=1.
// -----------------------------------------------------------------------------
//
// Estados con amplitud en el círculo unidad son las "claves" canónicas del
// motor: tienen inverso exacto bajo bind y son la base de una memoria
// asociativa libre de colisiones (en el límite D→∞).
//
// Generador determinista por semilla (xorshift64) — sin RNG externos.

[[nodiscard]] inline State random_phase_state(std::size_t dim, std::uint64_t seed) {
    if (dim == 0) throw std::invalid_argument("random_phase_state: dim 0.");
    State r(dim);
    std::uint64_t s = (seed == 0) ? 0x9E3779B97F4A7C15ULL : seed;
    constexpr double kTwoPi = 6.28318530717958647692;
    for (std::size_t i = 0; i < dim; ++i) {
        // xorshift64
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        const double u = static_cast<double>(s) / static_cast<double>(UINT64_MAX);
        const double phi = u * kTwoPi;
        r[i] = Complex{std::cos(phi), std::sin(phi)};
    }
    return r;
}

}  // namespace easyatom::ops
