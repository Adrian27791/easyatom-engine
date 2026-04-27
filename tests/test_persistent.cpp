// Tests del Ladrillo 19 — homologia persistente naive.

#include "test_framework.hpp"
#include "easyatom/topo/persistent.hpp"

#include <stdexcept>
#include <vector>

using easyatom::topo::BettiResult;
using easyatom::topo::vietoris_rips_betti;

EATEST_CASE(persistent_nube_vacia) {
    BettiResult r = vietoris_rips_betti({}, 1.0);
    EATEST_REQUIRE(r.beta_0 == 0);
    EATEST_REQUIRE(r.beta_1 == 0);
}

EATEST_CASE(persistent_un_punto_es_una_componente) {
    std::vector<std::vector<double>> pts = {{0.0, 0.0}};
    BettiResult r = vietoris_rips_betti(pts, 0.5);
    EATEST_REQUIRE(r.beta_0 == 1);
    EATEST_REQUIRE(r.beta_1 == 0);
}

EATEST_CASE(persistent_dos_puntos_lejos_son_dos_componentes) {
    std::vector<std::vector<double>> pts = {{0,0}, {10,0}};
    BettiResult r = vietoris_rips_betti(pts, 1.0);
    EATEST_REQUIRE(r.beta_0 == 2);
    EATEST_REQUIRE(r.edges  == 0);
    EATEST_REQUIRE(r.beta_1 == 0);
}

EATEST_CASE(persistent_dos_puntos_cercanos_son_una_componente) {
    std::vector<std::vector<double>> pts = {{0,0}, {0.5,0}};
    BettiResult r = vietoris_rips_betti(pts, 1.0);
    EATEST_REQUIRE(r.beta_0 == 1);
    EATEST_REQUIRE(r.edges  == 1);
}

EATEST_CASE(persistent_triangulo_relleno_no_tiene_hueco) {
    // Triangulo equilatero pequeño: las 3 aristas existen y el 2-simplex tambien.
    std::vector<std::vector<double>> pts = {
        {0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}
    };
    BettiResult r = vietoris_rips_betti(pts, 1.05);
    EATEST_REQUIRE(r.edges  == 3);
    EATEST_REQUIRE(r.faces  == 1);
    EATEST_REQUIRE(r.beta_0 == 1);
    EATEST_REQUIRE(r.beta_1 == 0);
}

EATEST_CASE(persistent_cuadrado_es_un_loop) {
    // 4 puntos en un cuadrado de lado 1; eps=1.05 captura solo las aristas
    // (no las diagonales ~1.414) => no hay 2-simplices => beta_1=1.
    std::vector<std::vector<double>> pts = {
        {0,0}, {1,0}, {1,1}, {0,1}
    };
    BettiResult r = vietoris_rips_betti(pts, 1.05);
    EATEST_REQUIRE(r.edges  == 4);
    EATEST_REQUIRE(r.faces  == 0);
    EATEST_REQUIRE(r.beta_0 == 1);
    EATEST_REQUIRE(r.beta_1 == 1);
}

EATEST_CASE(persistent_cuadrado_con_diagonales_se_rellena) {
    // eps grande captura tambien las diagonales => 2 triangulos => beta_1=0.
    std::vector<std::vector<double>> pts = {
        {0,0}, {1,0}, {1,1}, {0,1}
    };
    BettiResult r = vietoris_rips_betti(pts, 2.0);
    EATEST_REQUIRE(r.edges  == 6);
    EATEST_REQUIRE(r.faces  >= 2);
    EATEST_REQUIRE(r.beta_0 == 1);
    EATEST_REQUIRE(r.beta_1 == 0);
}

EATEST_CASE(persistent_epsilon_negativo_lanza) {
    bool t = false;
    try { vietoris_rips_betti({{0.0}}, -1.0); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}
