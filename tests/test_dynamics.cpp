// Tests del Ladrillo 5 — Koopman / EDMD.

#include "test_framework.hpp"
#include "easyatom/dynamics/koopman.hpp"

#include <cmath>
#include <vector>

using easyatom::dynamics::Matrix;
using easyatom::dynamics::Koopman;

constexpr double kKTol = 1e-9;

// -----------------------------------------------------------------------------
// Matrix.
// -----------------------------------------------------------------------------

EATEST_CASE(matrix_identity_es_neutra) {
    Matrix I = Matrix::identity(3);
    Matrix A(3, 3);
    A.at(0,0)=1; A.at(0,1)=2; A.at(0,2)=3;
    A.at(1,0)=4; A.at(1,1)=5; A.at(1,2)=6;
    A.at(2,0)=7; A.at(2,1)=8; A.at(2,2)=10;
    Matrix R = A * I;
    EATEST_REQUIRE(R.approx_equal(A, kKTol));
}

EATEST_CASE(matrix_inversa_por_inversa_es_identidad) {
    Matrix A(2, 2);
    A.at(0,0)=4; A.at(0,1)=7;
    A.at(1,0)=2; A.at(1,1)=6;
    Matrix Ainv = A.inverse();
    Matrix prod = A * Ainv;
    EATEST_REQUIRE(prod.approx_equal(Matrix::identity(2), 1e-9));
}

EATEST_CASE(matrix_singular_lanza) {
    Matrix S(2, 2);
    S.at(0,0)=1; S.at(0,1)=2;
    S.at(1,0)=2; S.at(1,1)=4;  // filas linealmente dependientes
    bool threw = false;
    try { (void)S.inverse(); } catch (const std::domain_error&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(matrix_transpose_double_es_identidad) {
    Matrix A(2, 3);
    A.at(0,0)=1; A.at(0,1)=2; A.at(0,2)=3;
    A.at(1,0)=4; A.at(1,1)=5; A.at(1,2)=6;
    Matrix B = A.transpose().transpose();
    EATEST_REQUIRE(B.approx_equal(A, kKTol));
}

// -----------------------------------------------------------------------------
// Koopman: recupera dinámicas lineales exactamente.
// -----------------------------------------------------------------------------

EATEST_CASE(koopman_lineal_recupera_dinamica_exacta) {
    // Sistema lineal en R^2: x_{t+1} = A x_t con
    //   A = [[ 0.9, 0.1],
    //        [-0.1, 0.95]]
    // Observables: Ψ(x) = (1, x_0, x_1).  K es 3x3 con bloque A en
    // las dos últimas filas/columnas.
    const double a00 = 0.9, a01 = 0.1, a10 = -0.1, a11 = 0.95;
    auto step = [&](double x, double y) {
        return std::pair<double,double>{a00*x + a01*y, a10*x + a11*y};
    };
    auto psi = [](double x, double y) -> std::vector<double> {
        return {1.0, x, y};
    };

    std::vector<Koopman::Snapshot> snaps;
    // Generamos trayectorias desde varias condiciones iniciales.
    for (double x0 : {-2.0, -1.0, 0.5, 1.5, 3.0}) {
        for (double y0 : {-1.5, 0.0, 2.0}) {
            double x = x0, y = y0;
            for (int t = 0; t < 8; ++t) {
                auto [xn, yn] = step(x, y);
                snaps.push_back({psi(x, y), psi(xn, yn)});
                x = xn; y = yn;
            }
        }
    }

    auto K = Koopman::fit(snaps);
    const auto& M = K.matrix();

    // Esperamos que la primera fila reproduzca la constante (0,0)→1
    // y las otras dos reproduzcan la dinámica.
    EATEST_REQUIRE_NEAR(M.at(0,0), 1.0, 1e-8);
    EATEST_REQUIRE_NEAR(M.at(0,1), 0.0, 1e-8);
    EATEST_REQUIRE_NEAR(M.at(0,2), 0.0, 1e-8);
    EATEST_REQUIRE_NEAR(M.at(1,1), a00, 1e-8);
    EATEST_REQUIRE_NEAR(M.at(1,2), a01, 1e-8);
    EATEST_REQUIRE_NEAR(M.at(2,1), a10, 1e-8);
    EATEST_REQUIRE_NEAR(M.at(2,2), a11, 1e-8);
}

EATEST_CASE(koopman_avanza_estado_correctamente) {
    // Misma dinámica, verificamos que advance reproduce la trayectoria.
    const double a00 = 0.9, a01 = 0.1, a10 = -0.1, a11 = 0.95;
    auto step = [&](double x, double y) {
        return std::pair<double,double>{a00*x + a01*y, a10*x + a11*y};
    };
    auto psi = [](double x, double y) -> std::vector<double> {
        return {1.0, x, y};
    };

    std::vector<Koopman::Snapshot> snaps;
    for (double x0 : {-1.0, 0.0, 1.0, 2.0}) {
        for (double y0 : {-1.0, 0.5, 2.0}) {
            double x = x0, y = y0;
            for (int t = 0; t < 6; ++t) {
                auto [xn, yn] = step(x, y);
                snaps.push_back({psi(x, y), psi(xn, yn)});
                x = xn; y = yn;
            }
        }
    }

    auto K = Koopman::fit(snaps);

    // Predicción a 5 pasos partiendo de (3, -2).
    double x = 3.0, y = -2.0;
    auto state = psi(x, y);
    auto pred = K.advance_n(state, 5);
    for (int t = 0; t < 5; ++t) {
        auto [xn, yn] = step(x, y);
        x = xn; y = yn;
    }
    EATEST_REQUIRE_NEAR(pred[1], x, 1e-7);
    EATEST_REQUIRE_NEAR(pred[2], y, 1e-7);
}

EATEST_CASE(koopman_sin_datos_lanza) {
    bool threw = false;
    try { (void)Koopman::fit({}); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(koopman_dinamica_afin_recupera_termino_constante) {
    // x_{t+1} = 0.5 x_t + 1
    // Punto fijo: x = 2.
    auto step = [](double x) { return 0.5 * x + 1.0; };
    auto psi = [](double x) -> std::vector<double> { return {1.0, x}; };
    std::vector<Koopman::Snapshot> snaps;
    for (double x0 : {-3.0, -1.0, 0.0, 1.0, 5.0, 7.0}) {
        double x = x0;
        for (int t = 0; t < 10; ++t) {
            const double xn = step(x);
            snaps.push_back({psi(x), psi(xn)});
            x = xn;
        }
    }
    auto K = Koopman::fit(snaps);
    const auto& M = K.matrix();
    // K[1,0] = b = 1 (término constante).
    // K[1,1] = a = 0.5.
    EATEST_REQUIRE_NEAR(M.at(1, 0), 1.0, 1e-8);
    EATEST_REQUIRE_NEAR(M.at(1, 1), 0.5, 1e-8);

    // El punto fijo predicho debe ser 2: tras muchos pasos desde 100,
    // converge a 2.
    auto state = psi(100.0);
    auto pred = K.advance_n(state, 100);
    EATEST_REQUIRE_NEAR(pred[1], 2.0, 1e-6);
}
