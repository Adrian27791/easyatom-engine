// =============================================================================
// EasyAtom · Ladrillo 5 — Operador de Koopman (EDMD).
// =============================================================================
//
// Para un sistema dinámico discreto x_{t+1} = f(x_t) sobre un espacio de
// estados X, el OPERADOR DE KOOPMAN K actúa sobre observables (funciones
// g: X → R) mediante composición:
//      (K g)(x) = g(f(x))
//
// Aunque f sea no lineal, K es LINEAL sobre el espacio de funciones. Esto
// es la idea clave: a costa de subir a un espacio de funciones, podemos
// estudiar dinámica no lineal con álgebra lineal.
//
// EDMD (Extended Dynamic Mode Decomposition, Williams-Kevrekidis-Rowley
// 2015): dado un diccionario finito de observables Ψ(x) = (ψ_1(x),...,ψ_N(x))
// y M pares (x_t, x_{t+1}), buscamos K ∈ R^{N×N} que mejor cumpla
//      Ψ(x_{t+1}) ≈ K · Ψ(x_t)            ∀t
// en el sentido de mínimos cuadrados:
//      K = Y X^T (X X^T)^{-1}        con X = [Ψ(x_t)]_t,  Y = [Ψ(x_{t+1})]_t
//
// Si las observables son afines y(x) = [1, x_1, x_2, ..., x_d], y la
// dinámica subyacente es x_{t+1} = A x_t + b, entonces K recupera A
// y b EXACTAMENTE (con datos suficientes y no degenerados).
//
// LO QUE IMPLEMENTAMOS:
//   * Matrix: tipo denso pequeño con ops lineales, transpose, inverse
//     (Gauss-Jordan con pivoteo parcial), multiplicación, identidad.
//   * Koopman::fit(snapshots): EDMD por mínimos cuadrados.
//   * Koopman::advance(state): aplica K una vez.
//   * Koopman::advance_n(state, n): aplica K^n.
//
// REGLAS:
//   * No usamos BLAS/LAPACK; implementación didáctica O(N^3) suficiente
//     para diccionarios pequeños (N ≤ ~64) y pruebas numéricas exactas.
//   * Si X X^T es singular lanzamos: no inventamos pseudo-inversa silenciosa.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace easyatom::dynamics {

// -----------------------------------------------------------------------------
// Matrix densa pequeña, almacenamiento row-major.
// -----------------------------------------------------------------------------

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols)
        : r_(rows), c_(cols), a_(rows * cols, 0.0) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix: filas o columnas = 0.");
        }
    }

    [[nodiscard]] std::size_t rows() const noexcept { return r_; }
    [[nodiscard]] std::size_t cols() const noexcept { return c_; }

    [[nodiscard]] double& at(std::size_t i, std::size_t j) {
        return a_[i * c_ + j];
    }
    [[nodiscard]] double at(std::size_t i, std::size_t j) const {
        return a_[i * c_ + j];
    }

    [[nodiscard]] static Matrix identity(std::size_t n) {
        Matrix M(n, n);
        for (std::size_t i = 0; i < n; ++i) M.at(i, i) = 1.0;
        return M;
    }

    [[nodiscard]] Matrix transpose() const {
        Matrix T(c_, r_);
        for (std::size_t i = 0; i < r_; ++i)
            for (std::size_t j = 0; j < c_; ++j) T.at(j, i) = at(i, j);
        return T;
    }

    [[nodiscard]] Matrix operator*(const Matrix& B) const {
        if (c_ != B.r_) {
            throw std::invalid_argument(
                "Matrix*: dimensiones incompatibles (" + std::to_string(c_) +
                " vs " + std::to_string(B.r_) + ").");
        }
        Matrix R(r_, B.c_);
        for (std::size_t i = 0; i < r_; ++i) {
            for (std::size_t k = 0; k < c_; ++k) {
                const double aik = at(i, k);
                if (aik == 0.0) continue;
                for (std::size_t j = 0; j < B.c_; ++j) {
                    R.at(i, j) += aik * B.at(k, j);
                }
            }
        }
        return R;
    }

    [[nodiscard]] std::vector<double> apply(const std::vector<double>& v) const {
        if (v.size() != c_) {
            throw std::invalid_argument("Matrix::apply: dim incompatible.");
        }
        std::vector<double> r(r_, 0.0);
        for (std::size_t i = 0; i < r_; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < c_; ++j) s += at(i, j) * v[j];
            r[i] = s;
        }
        return r;
    }

    /// Inversa (Gauss-Jordan con pivoteo parcial). Lanza si es singular.
    [[nodiscard]] Matrix inverse(double pivot_tol = 1e-12) const {
        if (r_ != c_) {
            throw std::invalid_argument("Matrix::inverse: no cuadrada.");
        }
        const std::size_t n = r_;
        Matrix A = *this;
        Matrix I = identity(n);
        for (std::size_t k = 0; k < n; ++k) {
            // Pivote por valor absoluto máximo en columna k, filas k..n-1.
            std::size_t piv = k;
            double best = std::abs(A.at(k, k));
            for (std::size_t i = k + 1; i < n; ++i) {
                const double v = std::abs(A.at(i, k));
                if (v > best) { best = v; piv = i; }
            }
            if (best < pivot_tol) {
                throw std::domain_error("Matrix::inverse: matriz singular.");
            }
            if (piv != k) {
                for (std::size_t j = 0; j < n; ++j) {
                    std::swap(A.at(k, j), A.at(piv, j));
                    std::swap(I.at(k, j), I.at(piv, j));
                }
            }
            const double inv_p = 1.0 / A.at(k, k);
            for (std::size_t j = 0; j < n; ++j) {
                A.at(k, j) *= inv_p;
                I.at(k, j) *= inv_p;
            }
            for (std::size_t i = 0; i < n; ++i) {
                if (i == k) continue;
                const double f = A.at(i, k);
                if (f == 0.0) continue;
                for (std::size_t j = 0; j < n; ++j) {
                    A.at(i, j) -= f * A.at(k, j);
                    I.at(i, j) -= f * I.at(k, j);
                }
            }
        }
        return I;
    }

    [[nodiscard]] bool approx_equal(const Matrix& B, double tol = 1e-9) const {
        if (r_ != B.r_ || c_ != B.c_) return false;
        for (std::size_t i = 0; i < r_; ++i)
            for (std::size_t j = 0; j < c_; ++j)
                if (std::abs(at(i, j) - B.at(i, j)) > tol) return false;
        return true;
    }

private:
    std::size_t r_ = 0;
    std::size_t c_ = 0;
    std::vector<double> a_;
};

// -----------------------------------------------------------------------------
// Operador de Koopman aproximado (EDMD).
// -----------------------------------------------------------------------------
//
// snapshots = lista de pares (Ψ(x_t), Ψ(x_{t+1})), ya evaluados sobre el
// diccionario de observables. Todos del mismo tamaño N.
//
// Construye X (N×M) y Y (N×M), luego K = Y X^T (X X^T)^{-1}.

class Koopman {
public:
    using Snapshot = std::pair<std::vector<double>, std::vector<double>>;

    [[nodiscard]] static Koopman fit(const std::vector<Snapshot>& snapshots) {
        if (snapshots.empty()) {
            throw std::invalid_argument("Koopman::fit: sin datos.");
        }
        const std::size_t N = snapshots.front().first.size();
        const std::size_t M = snapshots.size();
        if (N == 0) throw std::invalid_argument("Koopman::fit: dim 0.");

        Matrix X(N, M), Y(N, M);
        for (std::size_t t = 0; t < M; ++t) {
            const auto& [psi_t, psi_t1] = snapshots[t];
            if (psi_t.size() != N || psi_t1.size() != N) {
                throw std::invalid_argument(
                    "Koopman::fit: snapshot con dimensión inconsistente.");
            }
            for (std::size_t i = 0; i < N; ++i) {
                X.at(i, t) = psi_t[i];
                Y.at(i, t) = psi_t1[i];
            }
        }
        // K = Y X^T (X X^T)^{-1}
        Matrix XT = X.transpose();
        Matrix XXT = X * XT;          // N×N
        Matrix YXT = Y * XT;          // N×N
        Matrix XXT_inv = XXT.inverse();
        Matrix K = YXT * XXT_inv;     // N×N
        return Koopman(std::move(K));
    }

    [[nodiscard]] const Matrix& matrix() const noexcept { return K_; }

    /// Avanza un estado (en el espacio de observables) un paso.
    [[nodiscard]] std::vector<double> advance(const std::vector<double>& psi) const {
        return K_.apply(psi);
    }

    /// Avanza n pasos.
    [[nodiscard]] std::vector<double> advance_n(std::vector<double> psi,
                                                std::size_t n) const {
        for (std::size_t k = 0; k < n; ++k) psi = K_.apply(psi);
        return psi;
    }

private:
    explicit Koopman(Matrix K) : K_(std::move(K)) {}
    Matrix K_;
};

}  // namespace easyatom::dynamics
