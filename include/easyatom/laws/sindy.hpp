// =============================================================================
// EasyAtom · Ladrillo 6 — Identificación esparsa de leyes (SINDy).
// =============================================================================
//
// SINDy (Brunton-Proctor-Kutz 2016) descubre la LEY que rige unos datos
// asumiendo que se expresa como combinación lineal esparsa de funciones
// candidatas dentro de un diccionario:
//      y(x) = Σ_k w_k φ_k(x)        con la mayoría de w_k = 0.
//
// ALGORITMO STLSQ (Sequentially Thresholded Least Squares):
//   1. Resolver mínimos cuadrados ordinarios w = (Φ^T Φ)^{-1} Φ^T y.
//   2. Anular |w_k| < λ.
//   3. Refittear sobre las columnas restantes.
//   4. Repetir 2-3 hasta que no cambie el soporte.
//
// La esparsidad es la garantía de "ley física" frente al sobre-ajuste:
// la dinámica de Lorenz tiene 7 términos de 130, no 130 con coeficientes
// pequeños.
//
// FEATURE LIBRARIES INCLUIDAS:
//   * polynomial(d, max_degree): {1, x_1, ..., x_d, x_1², x_1 x_2, ...}
//   * fourier(d, k_max):         {1, sin(k x_i), cos(k x_i)} para k=1..k_max
//
// REGLAS:
//   * Reusamos easyatom::dynamics::Matrix (ya tiene inverse, transpose, *).
//   * Si la matriz Φ^T Φ es singular en el subconjunto activo, la
//     iteración pasa a refittear sin esos términos (los descarta) en
//     vez de inventar pseudo-inversa silenciosa.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "easyatom/dynamics/koopman.hpp"  // Matrix

namespace easyatom::laws {

using easyatom::dynamics::Matrix;

// Tipo de una función candidata: toma un vector x y devuelve un escalar.
using Feature = std::function<double(const std::vector<double>&)>;

struct LabeledFeature {
    Feature fn;
    std::string name;
};

// -----------------------------------------------------------------------------
// Bibliotecas de candidatos.
// -----------------------------------------------------------------------------
namespace library {

inline std::string monomial_name(const std::vector<int>& exps) {
    std::string s;
    bool any = false;
    for (std::size_t i = 0; i < exps.size(); ++i) {
        if (exps[i] == 0) continue;
        if (any) s += "*";
        s += "x" + std::to_string(i);
        if (exps[i] > 1) s += "^" + std::to_string(exps[i]);
        any = true;
    }
    if (!any) s = "1";
    return s;
}

/// Polinomios totales de grado ≤ max_degree en d variables.
inline std::vector<LabeledFeature> polynomial(std::size_t d,
                                              int max_degree) {
    if (max_degree < 0) {
        throw std::invalid_argument("polynomial: grado negativo.");
    }
    std::vector<LabeledFeature> out;
    std::vector<int> exps(d, 0);
    std::function<void(std::size_t, int)> rec =
        [&](std::size_t pos, int rem) {
            if (pos == d) {
                std::vector<int> e = exps;  // copia capturada por valor
                out.push_back({
                    [e](const std::vector<double>& x) {
                        double p = 1.0;
                        for (std::size_t i = 0; i < e.size(); ++i) {
                            for (int k = 0; k < e[i]; ++k) p *= x[i];
                        }
                        return p;
                    },
                    monomial_name(e)});
                return;
            }
            for (int k = 0; k <= rem; ++k) {
                exps[pos] = k;
                rec(pos + 1, rem - k);
            }
            exps[pos] = 0;
        };
    rec(0, max_degree);
    return out;
}

/// Diccionario de Fourier: {1} ∪ {sin(k·x_i), cos(k·x_i)} k=1..kmax.
inline std::vector<LabeledFeature> fourier(std::size_t d, int k_max) {
    if (k_max < 1) throw std::invalid_argument("fourier: k_max < 1.");
    std::vector<LabeledFeature> out;
    out.push_back({[](const std::vector<double>&) { return 1.0; }, "1"});
    for (std::size_t i = 0; i < d; ++i) {
        for (int k = 1; k <= k_max; ++k) {
            out.push_back({[i, k](const std::vector<double>& x) {
                               return std::sin(k * x[i]);
                           },
                           "sin(" + std::to_string(k) + "*x" +
                               std::to_string(i) + ")"});
            out.push_back({[i, k](const std::vector<double>& x) {
                               return std::cos(k * x[i]);
                           },
                           "cos(" + std::to_string(k) + "*x" +
                               std::to_string(i) + ")"});
        }
    }
    return out;
}

}  // namespace library

// -----------------------------------------------------------------------------
// Resultado del ajuste.
// -----------------------------------------------------------------------------
struct SparseLaw {
    std::vector<double> coefficients;     // tamaño = #features
    std::vector<std::string> names;       // mismos índices

    [[nodiscard]] double evaluate(const std::vector<LabeledFeature>& lib,
                                  const std::vector<double>& x) const {
        if (lib.size() != coefficients.size()) {
            throw std::invalid_argument("SparseLaw::evaluate: dim != lib.");
        }
        double y = 0.0;
        for (std::size_t k = 0; k < lib.size(); ++k) {
            if (coefficients[k] == 0.0) continue;
            y += coefficients[k] * lib[k].fn(x);
        }
        return y;
    }

    [[nodiscard]] std::size_t support_size() const noexcept {
        std::size_t s = 0;
        for (double c : coefficients) if (c != 0.0) ++s;
        return s;
    }

    [[nodiscard]] std::string pretty(double tol = 0.0) const {
        std::string out;
        bool first = true;
        for (std::size_t k = 0; k < coefficients.size(); ++k) {
            const double c = coefficients[k];
            if (std::abs(c) <= tol) continue;
            if (!first) out += (c >= 0 ? " + " : " - ");
            else if (c < 0) out += "-";
            first = false;
            out += std::to_string(std::abs(c)) + "*" + names[k];
        }
        if (first) out = "0";
        return out;
    }
};

// -----------------------------------------------------------------------------
// Detalle: OLS sobre subconjunto activo. data_X: M filas × full_features.
// -----------------------------------------------------------------------------
namespace detail {

inline std::vector<double> ols_subset(
    const Matrix& Phi_full,                // M × N
    const std::vector<double>& y,          // M
    const std::vector<std::size_t>& active // índices de columnas activas
) {
    const std::size_t M = Phi_full.rows();
    const std::size_t N = Phi_full.cols();
    const std::size_t K = active.size();
    std::vector<double> w(N, 0.0);
    if (K == 0) return w;
    if (y.size() != M) {
        throw std::invalid_argument("ols_subset: y de tamaño incompatible.");
    }

    // Construye Phi_act: M×K
    Matrix Phi_act(M, K);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < K; ++j)
            Phi_act.at(i, j) = Phi_full.at(i, active[j]);

    Matrix PhiT = Phi_act.transpose();      // K×M
    Matrix PtP = PhiT * Phi_act;            // K×K
    // Vector y como Matrix M×1
    Matrix yM(M, 1);
    for (std::size_t i = 0; i < M; ++i) yM.at(i, 0) = y[i];
    Matrix Pty = PhiT * yM;                 // K×1

    Matrix PtP_inv = PtP.inverse();         // lanza si singular
    Matrix wM = PtP_inv * Pty;              // K×1
    for (std::size_t j = 0; j < K; ++j) w[active[j]] = wM.at(j, 0);
    return w;
}

}  // namespace detail

// -----------------------------------------------------------------------------
// Sequentially Thresholded Least Squares.
// -----------------------------------------------------------------------------
//
// data_x: M muestras de entrada (cada una vector de dim d).
// data_y: M valores escalares y_i = ley(x_i) + ruido.
// lambda: umbral de esparsidad (coeficientes con |w| < λ se eliminan).
// max_iter: tope de iteraciones del refit.
//
[[nodiscard]] inline SparseLaw stlsq(
    const std::vector<LabeledFeature>& lib,
    const std::vector<std::vector<double>>& data_x,
    const std::vector<double>& data_y,
    double lambda,
    int max_iter = 20)
{
    if (lib.empty()) throw std::invalid_argument("stlsq: lib vacía.");
    if (data_x.empty()) throw std::invalid_argument("stlsq: sin muestras.");
    if (data_x.size() != data_y.size()) {
        throw std::invalid_argument("stlsq: |x| != |y|.");
    }
    if (lambda < 0.0) throw std::invalid_argument("stlsq: lambda<0.");

    const std::size_t M = data_x.size();
    const std::size_t N = lib.size();

    // Construye Φ (M × N).
    Matrix Phi(M, N);
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            Phi.at(i, j) = lib[j].fn(data_x[i]);
        }
    }

    // Inicialmente, todas las columnas activas.
    std::vector<std::size_t> active;
    active.reserve(N);
    for (std::size_t j = 0; j < N; ++j) active.push_back(j);

    std::vector<double> w(N, 0.0);
    for (int it = 0; it < max_iter; ++it) {
        // Refit sobre el soporte activo. Si singular, eliminamos la
        // columna problemática y reintentamos.
        bool fitted = false;
        while (!fitted && !active.empty()) {
            try {
                w = detail::ols_subset(Phi, data_y, active);
                fitted = true;
            } catch (const std::domain_error&) {
                active.pop_back();  // descarta el último; vuelve a probar
                std::fill(w.begin(), w.end(), 0.0);
            }
        }
        if (active.empty()) break;

        // Aplica umbral.
        std::vector<std::size_t> next_active;
        next_active.reserve(active.size());
        for (std::size_t j : active) {
            if (std::abs(w[j]) >= lambda) next_active.push_back(j);
            else                           w[j] = 0.0;
        }
        if (next_active.size() == active.size()) break;  // converged
        active.swap(next_active);
    }

    SparseLaw out;
    out.coefficients = std::move(w);
    out.names.reserve(N);
    for (const auto& f : lib) out.names.push_back(f.name);
    return out;
}

}  // namespace easyatom::laws
