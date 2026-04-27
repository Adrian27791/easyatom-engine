// Ladrillo C — Estado cuántico sintético "real":
//   amplitudes complejas + operadores unitarios + medida Born.
//
// Esto NO es física cuántica con hardware cuántico. Es la simulación
// matemática exacta de un registro de n qubits sobre amplitudes
// complejas, suficiente para hablar honestamente de:
//   - superposición (amplitudes complejas, no solo reales),
//   - interferencia (suma de amplitudes con fase),
//   - entrelazamiento (estados no factorizables, p.ej. Bell),
//   - colapso por regla de Born (p_i = |<i|psi>|^2).
//
// Convenciones:
//   - n qubits => dim = 2^n. Usamos little-endian: el qubit q es el bit
//     menos significativo cuando q=0, etc.
//   - Las puertas multi-qubit se implementan in-place sobre el vector
//     de amplitudes, recorriendo pares de índices que solo difieren en
//     los bits afectados. Costo O(2^n) por puerta.
//
// Sin dependencias externas. Header-only. C++20.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace easyatom::qstate {

using Complex = std::complex<double>;

inline constexpr double kPi_C = 3.14159265358979323846;

/// Registro de n qubits sobre amplitudes complejas.
/// Estado inicial: |0...0> (amplitud 1.0 en índice 0).
class CState {
public:
    explicit CState(std::size_t n_qubits)
        : n_(n_qubits), amps_(std::size_t{1} << n_qubits, Complex{0.0, 0.0}) {
        if (n_qubits == 0 || n_qubits > 24) {
            throw std::invalid_argument(
                "CState: n_qubits debe estar en [1, 24].");
        }
        amps_[0] = Complex{1.0, 0.0};
    }

    [[nodiscard]] std::size_t n_qubits() const noexcept { return n_; }
    [[nodiscard]] std::size_t dim()      const noexcept { return amps_.size(); }
    [[nodiscard]] const std::vector<Complex>& amplitudes() const noexcept {
        return amps_;
    }
    [[nodiscard]] std::vector<Complex>& amplitudes_mut() noexcept {
        return amps_;
    }

    /// Norma al cuadrado (debe ser 1 si el estado es válido).
    [[nodiscard]] double norm_squared() const noexcept {
        double s = 0.0;
        for (const auto& a : amps_) s += std::norm(a);
        return s;
    }

    /// Renormaliza in-place. Lanza si la norma es 0.
    void renormalize() {
        const double n2 = norm_squared();
        if (n2 <= 0.0) {
            throw std::runtime_error("CState::renormalize: norma 0.");
        }
        const double inv = 1.0 / std::sqrt(n2);
        for (auto& a : amps_) a *= inv;
    }

    /// Distribución de probabilidades de medir cada índice computacional.
    /// p_i = |amps_[i]|^2.
    [[nodiscard]] std::vector<double> probabilities() const {
        std::vector<double> p(amps_.size());
        for (std::size_t i = 0; i < amps_.size(); ++i) p[i] = std::norm(amps_[i]);
        return p;
    }

    // ---------- Puertas de un qubit (in-place) ----------

    void apply_X(std::size_t q) {
        check_q(q);
        const std::size_t mask = std::size_t{1} << q;
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if ((i & mask) == 0) {
                std::swap(amps_[i], amps_[i | mask]);
            }
        }
    }

    void apply_Z(std::size_t q) {
        check_q(q);
        const std::size_t mask = std::size_t{1} << q;
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if (i & mask) amps_[i] = -amps_[i];
        }
    }

    void apply_Y(std::size_t q) {
        check_q(q);
        const std::size_t mask = std::size_t{1} << q;
        const Complex I{0.0, 1.0};
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if ((i & mask) == 0) {
                const std::size_t j = i | mask;
                const Complex a = amps_[i];
                const Complex b = amps_[j];
                // Y = [[0,-i],[i,0]]
                amps_[i] = -I * b;
                amps_[j] =  I * a;
            }
        }
    }

    void apply_H(std::size_t q) {
        check_q(q);
        const std::size_t mask = std::size_t{1} << q;
        const double s = 1.0 / std::sqrt(2.0);
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if ((i & mask) == 0) {
                const std::size_t j = i | mask;
                const Complex a = amps_[i];
                const Complex b = amps_[j];
                amps_[i] = s * (a + b);
                amps_[j] = s * (a - b);
            }
        }
    }

    /// Rotación Rx(theta) = [[cos(t/2), -i sin(t/2)],[-i sin(t/2), cos(t/2)]].
    void apply_Rx(std::size_t q, double theta) {
        check_q(q);
        const std::size_t mask = std::size_t{1} << q;
        const double c = std::cos(theta * 0.5);
        const double s = std::sin(theta * 0.5);
        const Complex mIs{0.0, -s};
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if ((i & mask) == 0) {
                const std::size_t j = i | mask;
                const Complex a = amps_[i];
                const Complex b = amps_[j];
                amps_[i] = c * a + mIs * b;
                amps_[j] = mIs * a + c * b;
            }
        }
    }

    /// Rotación Ry(theta) = [[cos(t/2), -sin(t/2)],[sin(t/2), cos(t/2)]].
    void apply_Ry(std::size_t q, double theta) {
        check_q(q);
        const std::size_t mask = std::size_t{1} << q;
        const double c = std::cos(theta * 0.5);
        const double s = std::sin(theta * 0.5);
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if ((i & mask) == 0) {
                const std::size_t j = i | mask;
                const Complex a = amps_[i];
                const Complex b = amps_[j];
                amps_[i] = c * a - s * b;
                amps_[j] = s * a + c * b;
            }
        }
    }

    /// Rotación Rz(theta): diag(exp(-i t/2), exp(i t/2)).
    void apply_Rz(std::size_t q, double theta) {
        check_q(q);
        const std::size_t mask = std::size_t{1} << q;
        const Complex e0 = std::polar(1.0, -theta * 0.5);
        const Complex e1 = std::polar(1.0,  theta * 0.5);
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            amps_[i] *= ((i & mask) ? e1 : e0);
        }
    }

    /// Fase global e^{i phi} sobre |1> del qubit q.
    void apply_Phase(std::size_t q, double phi) {
        check_q(q);
        const std::size_t mask = std::size_t{1} << q;
        const Complex e = std::polar(1.0, phi);
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if (i & mask) amps_[i] *= e;
        }
    }

    // ---------- Puertas de dos qubits ----------

    /// CNOT(control, target): si control=1, voltea target.
    void apply_CNOT(std::size_t control, std::size_t target) {
        check_q(control); check_q(target);
        if (control == target) {
            throw std::invalid_argument("CNOT: control y target deben diferir.");
        }
        const std::size_t cm = std::size_t{1} << control;
        const std::size_t tm = std::size_t{1} << target;
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if ((i & cm) && ((i & tm) == 0)) {
                std::swap(amps_[i], amps_[i | tm]);
            }
        }
    }

    /// CZ(control, target): aplica fase -1 cuando ambos son |1>.
    void apply_CZ(std::size_t control, std::size_t target) {
        check_q(control); check_q(target);
        if (control == target) {
            throw std::invalid_argument("CZ: control y target deben diferir.");
        }
        const std::size_t cm = std::size_t{1} << control;
        const std::size_t tm = std::size_t{1} << target;
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if ((i & cm) && (i & tm)) amps_[i] = -amps_[i];
        }
    }

    /// SWAP(a, b): intercambia los qubits a y b.
    void apply_SWAP(std::size_t a, std::size_t b) {
        check_q(a); check_q(b);
        if (a == b) return;
        const std::size_t am = std::size_t{1} << a;
        const std::size_t bm = std::size_t{1} << b;
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            const bool ba = (i & am) != 0;
            const bool bb = (i & bm) != 0;
            if (ba && !bb) {
                const std::size_t j = (i & ~am) | bm;
                if (j > i) std::swap(amps_[i], amps_[j]);
            }
        }
    }

    // ---------- Medida (regla de Born) ----------

    /// Marginal de un qubit: P(q=1) = sum |amps_[i]|^2 sobre i con bit q=1.
    [[nodiscard]] double prob_of_one(std::size_t q) const {
        check_q(q);
        const std::size_t mask = std::size_t{1} << q;
        double p = 0.0;
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            if (i & mask) p += std::norm(amps_[i]);
        }
        return p;
    }

    /// Mide un qubit y colapsa el estado. Devuelve 0 o 1.
    int measure_qubit(std::size_t q, std::uint64_t seed) {
        check_q(q);
        const double p1 = prob_of_one(q);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const int outcome = (u(rng) < p1) ? 1 : 0;
        collapse_qubit(q, outcome);
        return outcome;
    }

    /// Mide el registro entero según Born y colapsa al índice resultante.
    /// Devuelve el índice computacional medido en [0, dim).
    std::size_t measure_all(std::uint64_t seed) {
        const auto p = probabilities();
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const double r = u(rng);
        double acc = 0.0;
        std::size_t out = p.size() - 1;
        for (std::size_t i = 0; i < p.size(); ++i) {
            acc += p[i];
            if (r <= acc) { out = i; break; }
        }
        // Colapso a |out>.
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            amps_[i] = (i == out) ? Complex{1.0, 0.0} : Complex{0.0, 0.0};
        }
        return out;
    }

    /// Colapsa un qubit al valor dado (proyectivo + renormalización).
    void collapse_qubit(std::size_t q, int value) {
        check_q(q);
        if (value != 0 && value != 1) {
            throw std::invalid_argument("collapse_qubit: value debe ser 0 o 1.");
        }
        const std::size_t mask = std::size_t{1} << q;
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            const int bit = (i & mask) ? 1 : 0;
            if (bit != value) amps_[i] = Complex{0.0, 0.0};
        }
        renormalize();
    }

    // ---------- Utilidades ----------

    /// Set determinista a un estado base |idx>.
    void set_basis(std::size_t idx) {
        if (idx >= amps_.size()) {
            throw std::out_of_range("set_basis: idx fuera de rango.");
        }
        for (auto& a : amps_) a = Complex{0.0, 0.0};
        amps_[idx] = Complex{1.0, 0.0};
    }

    /// Fidelidad con otro estado: |<phi|psi>|^2.
    [[nodiscard]] double fidelity(const CState& other) const {
        if (other.n_ != n_) {
            throw std::invalid_argument("fidelity: dimensiones distintas.");
        }
        Complex inner{0.0, 0.0};
        for (std::size_t i = 0; i < amps_.size(); ++i) {
            inner += std::conj(other.amps_[i]) * amps_[i];
        }
        return std::norm(inner);
    }

private:
    void check_q(std::size_t q) const {
        if (q >= n_) {
            throw std::out_of_range("CState: qubit fuera de rango.");
        }
    }

    std::size_t n_;
    std::vector<Complex> amps_;
};

// ---------- Helpers de construcción ----------

/// Estado de Bell |Phi+> = (|00> + |11>) / sqrt(2).
inline CState make_bell_phi_plus() {
    CState s(2);
    s.apply_H(0);
    s.apply_CNOT(0, 1);
    return s;
}

/// Estado GHZ de n qubits: (|0...0> + |1...1>) / sqrt(2).
inline CState make_ghz(std::size_t n_qubits) {
    CState s(n_qubits);
    s.apply_H(0);
    for (std::size_t q = 1; q < n_qubits; ++q) {
        s.apply_CNOT(0, q);
    }
    return s;
}

}  // namespace easyatom::qstate
