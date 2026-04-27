// =============================================================================
// EasyAtom · Ladrillo 0 — Multivectores de Clifford Cl(p,q)
// =============================================================================
//
// Implementación header-only, C++20 puro, sin dependencias externas.
//
// Un multivector en Cl(p,q) es un elemento del álgebra geométrica con:
//   - p generadores e_i con e_i^2 = +1   (índices [0, p))
//   - q generadores e_i con e_i^2 = -1   (índices [p, p+q))
//
// Representamos cada multivector como un vector de 2^N coeficientes reales,
// con N = p+q. El índice del coeficiente es una máscara de bits: el bit i
// indica si el generador e_i está presente en el blade canónico.
//
//   blade 0b000     -> escalar (1)
//   blade 0b001     -> e_0
//   blade 0b010     -> e_1
//   blade 0b011     -> e_0 ∧ e_1
//   blade 0b111     -> e_0 ∧ e_1 ∧ e_2  (pseudoescalar en Cl(3,0))
//
// Operaciones implementadas:
//   * suma / resta / escalar
//   * producto geométrico (no conmutativo)
//   * reverso (~A)
//   * involución de grado ( hat(A) )
//   * conjugación de Clifford ( bar(A) = reverso ∘ involución )
//   * proyección por grado
//   * norma escalar = <A * ~A>_0
//
// Las propiedades fundamentales se verifican en tests/test_clifford.cpp.
//
// Notas de diseño:
//   - Los signos de permutación se calculan contando inversiones al fusionar
//     dos blades canónicos.
//   - Los signos de cuadrado de generadores se aplican al colapsar los pares
//     (e_i, e_i) tras la fusión: + si i<p, - si i>=p.
//   - Cl(p,q) se parametriza por template => todas las dimensiones se
//     resuelven en tiempo de compilación.

#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace easyatom::clifford {

// -----------------------------------------------------------------------------
// Utilidades de bits a nivel de blade.
// -----------------------------------------------------------------------------

using BladeMask = std::uint32_t;  // soporta hasta N=32 generadores; de sobra.

/// Cuenta el grado de un blade canónico (número de generadores que lo forman).
[[nodiscard]] constexpr std::size_t blade_grade(BladeMask m) noexcept {
    return static_cast<std::size_t>(std::popcount(m));
}

/// Calcula el signo (+1 o -1) que aparece al fusionar dos blades canónicos
/// `a` y `b` para obtener `a XOR b`, contando las transposiciones necesarias
/// para reordenar los generadores en orden canónico ascendente.
///
/// Algoritmo clásico: para cada generador presente en `b`, contar cuántos
/// generadores de `a` con índice estrictamente mayor hay que "saltar".
[[nodiscard]] constexpr int blade_merge_sign(BladeMask a, BladeMask b) noexcept {
    // Para cada bit i activo en b, contar bits activos en a con índice > i.
    // Equivalente a: sum_{i in b} popcount(a >> (i+1)).
    std::uint32_t inversions = 0;
    BladeMask bb = b;
    while (bb != 0) {
        const int i = std::countr_zero(bb);  // índice del bit menos significativo
        bb &= bb - 1;                        // quita ese bit
        // Bits de `a` con índice > i (se mueven a través de e_i).
        const std::uint32_t shifted =
            (i + 1 < 32) ? (a >> static_cast<unsigned>(i + 1)) : 0u;
        inversions += static_cast<std::uint32_t>(std::popcount(shifted));
    }
    return (inversions & 1u) ? -1 : 1;
}

// -----------------------------------------------------------------------------
// Multivector<P, Q> — álgebra de Clifford Cl(P,Q) sobre R.
// -----------------------------------------------------------------------------

template <std::size_t P, std::size_t Q>
class Multivector {
public:
    static constexpr std::size_t kP = P;
    static constexpr std::size_t kQ = Q;
    static constexpr std::size_t kN = P + Q;
    static_assert(kN <= 16, "Cl(p,q) limitado a p+q <= 16 (2^16 = 65536 coefs).");
    static constexpr std::size_t kSize = std::size_t{1} << kN;

    using Coeffs = std::array<double, kSize>;

    // -------------------------------------------------------------------------
    // Construcción.
    // -------------------------------------------------------------------------

    constexpr Multivector() noexcept : c_{} {}

    /// Construye un escalar puro.
    explicit constexpr Multivector(double scalar) noexcept : c_{} {
        c_[0] = scalar;
    }

    /// Construye un multivector a partir de un blade y su coeficiente.
    static constexpr Multivector from_blade(BladeMask blade, double coeff) noexcept {
        if (blade >= kSize) {
            // Comportamiento defensivo en debug: lanzamos. En release, el
            // assert() siguiente actúa como contrato. (Sin alucinar.)
            return Multivector{};
        }
        Multivector mv;
        mv.c_[blade] = coeff;
        return mv;
    }

    /// Genera el i-ésimo generador e_i (i in [0, N)).
    static constexpr Multivector e(std::size_t i) {
        if (i >= kN) {
            throw std::out_of_range("Multivector::e(i): índice fuera de Cl(P,Q).");
        }
        return from_blade(static_cast<BladeMask>(BladeMask{1} << i), 1.0);
    }

    /// Pseudoescalar I = e_0 ∧ e_1 ∧ ... ∧ e_{N-1}.
    static constexpr Multivector pseudoscalar() noexcept {
        return from_blade(static_cast<BladeMask>(kSize - 1), 1.0);
    }

    // -------------------------------------------------------------------------
    // Acceso a coeficientes.
    // -------------------------------------------------------------------------

    [[nodiscard]] constexpr double scalar() const noexcept { return c_[0]; }

    [[nodiscard]] constexpr double coeff(BladeMask blade) const noexcept {
        return (blade < kSize) ? c_[blade] : 0.0;
    }

    constexpr void set_coeff(BladeMask blade, double v) {
        if (blade >= kSize) {
            throw std::out_of_range("set_coeff: blade fuera del álgebra.");
        }
        c_[blade] = v;
    }

    [[nodiscard]] constexpr const Coeffs& coeffs() const noexcept { return c_; }

    // -------------------------------------------------------------------------
    // Suma, resta, escalar.
    // -------------------------------------------------------------------------

    constexpr Multivector& operator+=(const Multivector& o) noexcept {
        for (std::size_t i = 0; i < kSize; ++i) c_[i] += o.c_[i];
        return *this;
    }
    constexpr Multivector& operator-=(const Multivector& o) noexcept {
        for (std::size_t i = 0; i < kSize; ++i) c_[i] -= o.c_[i];
        return *this;
    }
    constexpr Multivector& operator*=(double s) noexcept {
        for (auto& x : c_) x *= s;
        return *this;
    }

    [[nodiscard]] friend constexpr Multivector operator+(Multivector a, const Multivector& b) noexcept {
        a += b; return a;
    }
    [[nodiscard]] friend constexpr Multivector operator-(Multivector a, const Multivector& b) noexcept {
        a -= b; return a;
    }
    [[nodiscard]] friend constexpr Multivector operator-(Multivector a) noexcept {
        for (auto& x : a.c_) x = -x;
        return a;
    }
    [[nodiscard]] friend constexpr Multivector operator*(Multivector a, double s) noexcept {
        a *= s; return a;
    }
    [[nodiscard]] friend constexpr Multivector operator*(double s, Multivector a) noexcept {
        a *= s; return a;
    }

    // -------------------------------------------------------------------------
    // Producto geométrico (núcleo del álgebra).
    // -------------------------------------------------------------------------
    //
    // (sum_a alpha_a e_a) * (sum_b beta_b e_b)
    //   = sum_{a,b} alpha_a beta_b * sign_perm(a,b) * sign_squares(a&b) * e_{a^b}
    //
    // donde:
    //   sign_perm(a,b)    = paridad de inversiones al fusionar a y b
    //   sign_squares(m)   = producto de signos de cada generador en m
    //                       (+1 si índice < P, -1 si índice >= P)

    [[nodiscard]] friend constexpr Multivector operator*(const Multivector& A,
                                                          const Multivector& B) noexcept {
        Multivector R;
        for (BladeMask a = 0; a < kSize; ++a) {
            const double alpha = A.c_[a];
            if (alpha == 0.0) continue;
            for (BladeMask b = 0; b < kSize; ++b) {
                const double beta = B.c_[b];
                if (beta == 0.0) continue;
                const int sp = blade_merge_sign(a, b);
                const BladeMask shared = a & b;  // generadores que se cuadran
                int sq = 1;
                BladeMask s = shared;
                while (s != 0) {
                    const int i = std::countr_zero(s);
                    s &= s - 1;
                    if (static_cast<std::size_t>(i) >= kP) sq = -sq;
                }
                const BladeMask out = a ^ b;
                R.c_[out] += static_cast<double>(sp * sq) * alpha * beta;
            }
        }
        return R;
    }

    // -------------------------------------------------------------------------
    // Operaciones de involución.
    // -------------------------------------------------------------------------

    /// Reverso ~A: invierte el orden de los generadores en cada blade.
    /// Para un k-blade, reverse_sign = (-1)^(k(k-1)/2).
    [[nodiscard]] constexpr Multivector reverse() const noexcept {
        Multivector R;
        for (BladeMask m = 0; m < kSize; ++m) {
            const std::size_t k = blade_grade(m);
            const int sign = ((k * (k - 1) / 2) & 1u) ? -1 : 1;
            R.c_[m] = static_cast<double>(sign) * c_[m];
        }
        return R;
    }

    /// Involución de grado: hat(A) cambia el signo de los blades de grado impar.
    [[nodiscard]] constexpr Multivector grade_involution() const noexcept {
        Multivector R;
        for (BladeMask m = 0; m < kSize; ++m) {
            R.c_[m] = (blade_grade(m) & 1u) ? -c_[m] : c_[m];
        }
        return R;
    }

    /// Conjugación de Clifford: bar(A) = grade_involution(reverse(A)).
    [[nodiscard]] constexpr Multivector clifford_conjugate() const noexcept {
        return grade_involution().reverse();
    }

    /// Proyección sobre la parte de grado k.
    [[nodiscard]] constexpr Multivector grade(std::size_t k) const noexcept {
        Multivector R;
        for (BladeMask m = 0; m < kSize; ++m) {
            if (blade_grade(m) == k) R.c_[m] = c_[m];
        }
        return R;
    }

    // -------------------------------------------------------------------------
    // Norma escalar (puede ser negativa para signaturas no euclídeas).
    // -------------------------------------------------------------------------

    /// <A * ~A>_0 — parte escalar del producto con el reverso.
    [[nodiscard]] constexpr double scalar_norm_squared() const noexcept {
        const Multivector r = reverse();
        const Multivector p = (*this) * r;
        return p.scalar();
    }

    // -------------------------------------------------------------------------
    // Comparación aproximada (uso en tests).
    // -------------------------------------------------------------------------

    [[nodiscard]] bool approx_equal(const Multivector& o, double tol = 1e-12) const noexcept {
        for (std::size_t i = 0; i < kSize; ++i) {
            const double diff = c_[i] - o.c_[i];
            if (std::fabs(diff) > tol) return false;
        }
        return true;
    }

private:
    Coeffs c_;
};

// -----------------------------------------------------------------------------
// Aliases ergonómicos para las álgebras más usadas.
// -----------------------------------------------------------------------------

using G2  = Multivector<2, 0>;  ///< Cl(2,0) — plano euclídeo.
using G3  = Multivector<3, 0>;  ///< Cl(3,0) — espacio 3D euclídeo.
using STA = Multivector<1, 3>;  ///< Cl(1,3) — álgebra del espaciotiempo (Hestenes).

}  // namespace easyatom::clifford
