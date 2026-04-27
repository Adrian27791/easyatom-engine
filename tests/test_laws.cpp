// Tests del Ladrillo 6 — STLSQ / SINDy.

#include "test_framework.hpp"
#include "easyatom/laws/sindy.hpp"

#include <cmath>
#include <vector>

using namespace easyatom::laws;

constexpr double kLawTol = 1e-9;
constexpr double kLawPi  = 3.14159265358979323846;

EATEST_CASE(library_polynomial_d1_grado2_tiene_3_features) {
    auto lib = library::polynomial(1, 2);  // {1, x0, x0^2}
    EATEST_REQUIRE(lib.size() == 3);
    std::vector<double> x = {3.0};
    EATEST_REQUIRE_NEAR(lib[0].fn(x), 1.0, kLawTol);
    EATEST_REQUIRE_NEAR(lib[1].fn(x), 3.0, kLawTol);
    EATEST_REQUIRE_NEAR(lib[2].fn(x), 9.0, kLawTol);
}

EATEST_CASE(library_polynomial_d2_grado2_tiene_6_features) {
    // {1, x0, x1, x0^2, x0*x1, x1^2}
    auto lib = library::polynomial(2, 2);
    EATEST_REQUIRE(lib.size() == 6);
}

EATEST_CASE(stlsq_recupera_lineal_2x_mas_3) {
    auto lib = library::polynomial(1, 3);  // {1, x, x^2, x^3}
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    for (int i = -10; i <= 10; ++i) {
        const double x = static_cast<double>(i) * 0.5;
        X.push_back({x});
        Y.push_back(2.0 * x + 3.0);
    }
    auto law = stlsq(lib, X, Y, /*lambda*/ 1e-4);
    EATEST_REQUIRE(law.support_size() == 2);
    EATEST_REQUIRE_NEAR(law.coefficients[0], 3.0, 1e-8);  // const
    EATEST_REQUIRE_NEAR(law.coefficients[1], 2.0, 1e-8);  // x
    EATEST_REQUIRE(law.coefficients[2] == 0.0);
    EATEST_REQUIRE(law.coefficients[3] == 0.0);
}

EATEST_CASE(stlsq_recupera_x_cuadrado_mas_uno) {
    auto lib = library::polynomial(1, 4);
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    for (int i = -8; i <= 8; ++i) {
        const double x = static_cast<double>(i) * 0.4;
        X.push_back({x});
        Y.push_back(x * x + 1.0);
    }
    auto law = stlsq(lib, X, Y, 1e-4);
    EATEST_REQUIRE(law.support_size() == 2);
    EATEST_REQUIRE_NEAR(law.coefficients[0], 1.0, 1e-8);
    EATEST_REQUIRE_NEAR(law.coefficients[2], 1.0, 1e-8);
    EATEST_REQUIRE(law.coefficients[1] == 0.0);
    EATEST_REQUIRE(law.coefficients[3] == 0.0);
}

EATEST_CASE(stlsq_evaluate_reproduce_la_ley) {
    auto lib = library::polynomial(2, 2);
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    // y = 5 + 2 x0 - 3 x1 + 0.5 x0 x1
    auto truth = [](double a, double b) {
        return 5.0 + 2.0 * a - 3.0 * b + 0.5 * a * b;
    };
    for (int i = -3; i <= 3; ++i) {
        for (int j = -3; j <= 3; ++j) {
            const double a = static_cast<double>(i);
            const double b = static_cast<double>(j);
            X.push_back({a, b});
            Y.push_back(truth(a, b));
        }
    }
    auto law = stlsq(lib, X, Y, 1e-4);
    for (double a = -2.5; a <= 2.5; a += 1.0) {
        for (double b = -2.5; b <= 2.5; b += 1.0) {
            EATEST_REQUIRE_NEAR(law.evaluate(lib, {a, b}), truth(a, b), 1e-7);
        }
    }
}

EATEST_CASE(stlsq_recupera_seno_puro) {
    auto lib = library::fourier(1, 3);  // {1, sin(kx), cos(kx)} k=1..3
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    for (int i = 0; i < 40; ++i) {
        const double x = -kLawPi + (2.0 * kLawPi) * (i / 39.0);
        X.push_back({x});
        Y.push_back(std::sin(x));
    }
    auto law = stlsq(lib, X, Y, 1e-4);
    // Esperamos un único término no nulo: sin(1*x0).
    EATEST_REQUIRE(law.support_size() == 1);
    // Encuentra qué coeficiente es no nulo.
    int idx = -1;
    for (std::size_t k = 0; k < law.coefficients.size(); ++k) {
        if (law.coefficients[k] != 0.0) idx = static_cast<int>(k);
    }
    EATEST_REQUIRE(idx >= 0);
    EATEST_REQUIRE(law.names[idx] == "sin(1*x0)");
    EATEST_REQUIRE_NEAR(law.coefficients[idx], 1.0, 1e-8);
}

EATEST_CASE(stlsq_lambda_cero_no_aplica_umbral) {
    auto lib = library::polynomial(1, 2);
    std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}};
    std::vector<double> Y = {0.0, 1.0, 4.0};  // y = x^2
    auto law = stlsq(lib, X, Y, 0.0);
    // Sin umbral, OLS exacto recupera coef[2] = 1, otros = 0 (sistema
    // exacto), pero todos los coef se quedan en el soporte.
    EATEST_REQUIRE_NEAR(law.coefficients[2], 1.0, 1e-8);
}

EATEST_CASE(stlsq_lib_vacia_lanza) {
    bool threw = false;
    try { (void)stlsq({}, {{0.0}}, {0.0}, 0.1); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}

EATEST_CASE(stlsq_x_y_dim_distinta_lanza) {
    auto lib = library::polynomial(1, 1);
    bool threw = false;
    try { (void)stlsq(lib, {{0.0}, {1.0}}, {0.0}, 0.1); }
    catch (const std::invalid_argument&) { threw = true; }
    EATEST_REQUIRE(threw);
}
