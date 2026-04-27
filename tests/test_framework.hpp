// Framework de tests minimalista.
// Sin dependencias. Cero magia. Una macro de assert con mensaje claro.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace eatest {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) {
        registry().push_back({name, fn});
    }
};

inline int run_all() {
    int passed = 0;
    int failed = 0;
    for (auto& t : registry()) {
        std::printf("[ RUN      ] %s\n", t.name);
        try {
            t.fn();
            std::printf("[       OK ] %s\n", t.name);
            ++passed;
        } catch (const std::exception& e) {
            std::printf("[  FAILED  ] %s\n   what(): %s\n", t.name, e.what());
            ++failed;
        } catch (...) {
            std::printf("[  FAILED  ] %s\n   <excepción desconocida>\n", t.name);
            ++failed;
        }
    }
    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace eatest

#define EATEST_CASE(NAME)                                                     \
    static void NAME();                                                       \
    static ::eatest::Registrar registrar_##NAME(#NAME, &NAME);                \
    static void NAME()

#define EATEST_REQUIRE(COND)                                                  \
    do {                                                                      \
        if (!(COND)) {                                                        \
            throw std::runtime_error(                                         \
                std::string("REQUIRE falló: ") + #COND +                      \
                " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")");      \
        }                                                                     \
    } while (0)

#define EATEST_REQUIRE_NEAR(A, B, TOL)                                        \
    do {                                                                      \
        const double _a = (A);                                                \
        const double _b = (B);                                                \
        const double _d = std::fabs(_a - _b);                                 \
        if (_d > (TOL)) {                                                     \
            throw std::runtime_error(                                         \
                std::string("REQUIRE_NEAR falló: |") + std::to_string(_a) +   \
                " - " + std::to_string(_b) + "| = " + std::to_string(_d) +    \
                " > tol=" + std::to_string(static_cast<double>(TOL)) +        \
                " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")");      \
        }                                                                     \
    } while (0)
