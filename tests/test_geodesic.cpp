// =============================================================================
// tests/test_geodesic.cpp  --  L31 planner geodesico (Dijkstra + filtro beta_0)
// =============================================================================

#include <stdexcept>
#include <cmath>

#include "easyatom/plan/geodesic.hpp"
#include "test_framework.hpp"

using easyatom::cst::CompiledLaw;
using easyatom::hilbert::Complex;
using easyatom::hilbert::State;
using easyatom::plan::find_path;

// helper: ley con state e_i en dimension d (un canal puro).
static CompiledLaw mk_law(std::size_t d, std::size_t i) {
    CompiledLaw L;
    State s(d);
    s[i] = Complex{1.0, 0.0};
    L.state = std::move(s);
    return L;
}

// helper: ley con state mezcla normalizada de e_a + e_b.
static CompiledLaw mk_mix(std::size_t d, std::size_t a, std::size_t b,
                          double wa, double wb) {
    CompiledLaw L;
    State s(d);
    s[a] = Complex{wa, 0.0};
    s[b] = Complex{wb, 0.0};
    L.state = std::move(s);
    return L;
}

EATEST_CASE(geo_src_igual_dst_devuelve_singleton) {
    std::vector<CompiledLaw> laws = { mk_law(4, 0), mk_law(4, 1) };
    auto p = find_path(laws, 0, 0, 1.0);
    EATEST_REQUIRE(p.size() == 1);
    EATEST_REQUIRE(p[0] == 0);
}

EATEST_CASE(geo_path_directo_si_max_eps_alto) {
    // dos leyes ortogonales -> w = 1.0; con max_eps=1.0 acepta arista.
    std::vector<CompiledLaw> laws = { mk_law(4, 0), mk_law(4, 1) };
    auto p = find_path(laws, 0, 1, 1.0);
    EATEST_REQUIRE(p.size() == 2);
    EATEST_REQUIRE(p[0] == 0);
    EATEST_REQUIRE(p[1] == 1);
}

EATEST_CASE(geo_componentes_disjuntas_devuelve_vacio) {
    // Filtro estricto: solo aristas con fidelity ~1 -> ortogonales se cortan.
    std::vector<CompiledLaw> laws = { mk_law(4, 0), mk_law(4, 1) };
    auto p = find_path(laws, 0, 1, 0.1);
    EATEST_REQUIRE(p.empty());
}

EATEST_CASE(geo_dijkstra_prefiere_ruta_mas_corta) {
    // b es mezcla equilibrada: fidelity(a,b)=0.5 y fidelity(b,c)=0.5,
    // ambas aristas pasan max_eps=0.5. La directa a-c tiene w=1.0 -> cortada.
    // Camino esperado: 0 -> 1 -> 2.
    const double r = 1.0 / std::sqrt(2.0);
    auto a = mk_law(4, 0);
    auto b = mk_mix(4, 0, 1, r, r);
    auto c = mk_law(4, 1);
    std::vector<CompiledLaw> laws = { a, b, c };
    auto p = find_path(laws, 0, 2, 0.6);
    EATEST_REQUIRE(p.size() == 3);
    EATEST_REQUIRE(p[0] == 0);
    EATEST_REQUIRE(p[1] == 1);
    EATEST_REQUIRE(p[2] == 2);
}

EATEST_CASE(geo_filtro_eps_corta_aristas_lejanas) {
    // Si max_eps < 1 - fidelity(a,c), no hay ruta directa 0->2.
    std::vector<CompiledLaw> laws = {
        mk_law(4, 0),
        mk_law(4, 1),
        mk_law(4, 2),
    };
    // Todas ortogonales entre si: w=1.0 cada arista; max_eps=0.5 corta todas.
    auto p = find_path(laws, 0, 2, 0.5);
    EATEST_REQUIRE(p.empty());
}

EATEST_CASE(geo_lanza_si_laws_vacio) {
    bool t = false;
    try { (void)find_path({}, 0, 0, 0.5); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(geo_lanza_si_indices_fuera_de_rango) {
    std::vector<CompiledLaw> laws = { mk_law(4, 0) };
    bool t = false;
    try { (void)find_path(laws, 0, 5, 0.5); }
    catch (const std::out_of_range&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(geo_lanza_si_max_eps_invalido) {
    std::vector<CompiledLaw> laws = { mk_law(4, 0), mk_law(4, 1) };
    bool t = false;
    try { (void)find_path(laws, 0, 1, -0.1); }
    catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}
