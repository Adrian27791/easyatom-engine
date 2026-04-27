// =============================================================================
// EasyAtom · Ladrillo 1 — Espacio de Hilbert simulado H_D sobre los complejos.
// =============================================================================
//
// Un estado |ψ⟩ ∈ H_D es un vector complejo de dimensión D. Sobre él se
// definen:
//   * producto interno hermítico       ⟨φ|ψ⟩
//   * norma                            ||ψ|| = sqrt(⟨ψ|ψ⟩)
//   * normalización                    |ψ⟩ / ||ψ||
//   * superposición ponderada          α|φ⟩ + β|ψ⟩
//   * proyector                        P_ψ = |ψ⟩⟨ψ| / ⟨ψ|ψ⟩
//   * proyección de |φ⟩ sobre |ψ⟩      P_ψ |φ⟩
//   * fidelidad cuántica               F(φ,ψ) = |⟨φ|ψ⟩|² / (⟨φ|φ⟩⟨ψ|ψ⟩)
//
// Reglas de diseño del Ladrillo 1:
//   - Los estados son vectores en C^D, no bits ni símbolos.
//   - El producto interno es hermítico (antilineal en el primer argumento).
//   - Cualquier operación sobre un estado nulo ||ψ||=0 lanza, no aproxima.
//     (Coherente con "no alucinar por construcción".)
//   - Las dimensiones de dos estados deben coincidir o se lanza.
//
// Este ladrillo es el sustrato sobre el que el Ladrillo 2 montará los
// operadores fundamentales (bind / bundle / permute / unbind) como rotores
// reales y proyectores, no como XOR de bits.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace easyatom::hilbert {

using Complex = std::complex<double>;

// -----------------------------------------------------------------------------
// State — ket |ψ⟩ ∈ H_D.
// -----------------------------------------------------------------------------

class State {
public:
    // ---------- Construcción ---------------------------------------------------

    State() = default;

    /// Estado de dimensión D inicializado a cero.
    explicit State(std::size_t dim) : c_(dim, Complex{0.0, 0.0}) {
        if (dim == 0) {
            throw std::invalid_argument("State: dimensión 0 no admitida.");
        }
    }

    /// Construye desde una lista explícita de amplitudes.
    explicit State(std::vector<Complex> amplitudes) : c_(std::move(amplitudes)) {
        if (c_.empty()) {
            throw std::invalid_argument("State: amplitudes vacías.");
        }
    }

    /// Estado base canónico |i⟩: 1 en la posición i, 0 en el resto.
    [[nodiscard]] static State basis(std::size_t dim, std::size_t i) {
        if (dim == 0) throw std::invalid_argument("basis: dim=0.");
        if (i >= dim) throw std::out_of_range("basis: i fuera de rango.");
        State s(dim);
        s.c_[i] = Complex{1.0, 0.0};
        return s;
    }

    // ---------- Acceso ---------------------------------------------------------

    [[nodiscard]] std::size_t dim() const noexcept { return c_.size(); }

    [[nodiscard]] const Complex& operator[](std::size_t i) const { return c_.at(i); }
    Complex& operator[](std::size_t i) { return c_.at(i); }

    [[nodiscard]] const std::vector<Complex>& amplitudes() const noexcept { return c_; }

    // ---------- Aritmética -----------------------------------------------------

    /// |ψ⟩ + |φ⟩  (ambos deben tener la misma dimensión).
    [[nodiscard]] friend State operator+(const State& a, const State& b) {
        check_same_dim(a, b, "operator+");
        State r(a.dim());
        for (std::size_t i = 0; i < a.dim(); ++i) r.c_[i] = a.c_[i] + b.c_[i];
        return r;
    }

    [[nodiscard]] friend State operator-(const State& a, const State& b) {
        check_same_dim(a, b, "operator-");
        State r(a.dim());
        for (std::size_t i = 0; i < a.dim(); ++i) r.c_[i] = a.c_[i] - b.c_[i];
        return r;
    }

    /// Multiplicación escalar (real o complejo).
    [[nodiscard]] friend State operator*(Complex s, const State& v) {
        State r(v.dim());
        for (std::size_t i = 0; i < v.dim(); ++i) r.c_[i] = s * v.c_[i];
        return r;
    }
    [[nodiscard]] friend State operator*(const State& v, Complex s) { return s * v; }
    [[nodiscard]] friend State operator*(double s, const State& v) {
        return Complex{s, 0.0} * v;
    }
    [[nodiscard]] friend State operator*(const State& v, double s) { return s * v; }

    // ---------- Norma y producto interno --------------------------------------

    /// ⟨ψ|ψ⟩ — siempre real ≥ 0.
    [[nodiscard]] double norm_squared() const noexcept {
        double acc = 0.0;
        for (const auto& z : c_) acc += std::norm(z);
        return acc;
    }

    [[nodiscard]] double norm() const noexcept { return std::sqrt(norm_squared()); }

    /// Devuelve una copia normalizada. Lanza si el estado es nulo.
    [[nodiscard]] State normalized() const {
        const double n = norm();
        if (n == 0.0) {
            throw std::domain_error("normalized: estado nulo, sin dirección definida.");
        }
        State r(dim());
        const double inv = 1.0 / n;
        for (std::size_t i = 0; i < dim(); ++i) r.c_[i] = c_[i] * inv;
        return r;
    }

    /// Comparación aproximada coordenada a coordenada.
    [[nodiscard]] bool approx_equal(const State& o, double tol = 1e-12) const {
        if (dim() != o.dim()) return false;
        for (std::size_t i = 0; i < dim(); ++i) {
            if (std::abs(c_[i] - o.c_[i]) > tol) return false;
        }
        return true;
    }

    static void check_same_dim(const State& a, const State& b, const char* op) {
        if (a.dim() != b.dim()) {
            throw std::invalid_argument(
                std::string("Hilbert::") + op + ": dimensiones distintas (" +
                std::to_string(a.dim()) + " vs " + std::to_string(b.dim()) + ").");
        }
    }

private:
    std::vector<Complex> c_;

    friend Complex inner(const State& a, const State& b);
};

// -----------------------------------------------------------------------------
// Producto interno hermítico ⟨a|b⟩ — antilineal en `a`, lineal en `b`.
// -----------------------------------------------------------------------------

[[nodiscard]] inline Complex inner(const State& a, const State& b) {
    State::check_same_dim(a, b, "inner");
    Complex acc{0.0, 0.0};
    for (std::size_t i = 0; i < a.dim(); ++i) {
        acc += std::conj(a.c_[i]) * b.c_[i];
    }
    return acc;
}

// -----------------------------------------------------------------------------
// Operaciones derivadas.
// -----------------------------------------------------------------------------

/// Proyección de `phi` sobre la dirección de `psi` (no nula).
///   P_ψ φ = |ψ⟩ ⟨ψ|φ⟩ / ⟨ψ|ψ⟩
[[nodiscard]] inline State project(const State& phi, const State& psi) {
    const double nps = psi.norm_squared();
    if (nps == 0.0) {
        throw std::domain_error("project: dirección |ψ⟩ nula.");
    }
    const Complex coeff = inner(psi, phi) / Complex{nps, 0.0};
    return coeff * psi;
}

/// Fidelidad cuántica F(φ,ψ) = |⟨φ|ψ⟩|² / (⟨φ|φ⟩⟨ψ|ψ⟩).
/// Devuelve un valor en [0,1]; 1 = misma dirección, 0 = ortogonales.
[[nodiscard]] inline double fidelity(const State& phi, const State& psi) {
    const double nphi = phi.norm_squared();
    const double npsi = psi.norm_squared();
    if (nphi == 0.0 || npsi == 0.0) {
        throw std::domain_error("fidelity: estado nulo.");
    }
    const Complex ip = inner(phi, psi);
    return std::norm(ip) / (nphi * npsi);
}

/// Superposición convexa ponderada de un conjunto de estados (todos con la
/// misma dimensión). `weights[i]` es el coeficiente complejo de `states[i]`.
/// El resultado NO se normaliza automáticamente.
[[nodiscard]] inline State superpose(const std::vector<Complex>& weights,
                                     const std::vector<State>& states) {
    if (weights.size() != states.size() || states.empty()) {
        throw std::invalid_argument("superpose: dimensiones de entrada inválidas.");
    }
    const std::size_t D = states.front().dim();
    State r(D);
    for (std::size_t k = 0; k < states.size(); ++k) {
        if (states[k].dim() != D) {
            throw std::invalid_argument("superpose: estados con dimensiones distintas.");
        }
        for (std::size_t i = 0; i < D; ++i) {
            r[i] += weights[k] * states[k][i];
        }
    }
    return r;
}

}  // namespace easyatom::hilbert
