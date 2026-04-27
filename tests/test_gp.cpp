// Tests del Ladrillo 18 — GP simbolico.

#include "test_framework.hpp"
#include "easyatom/cst/gp.hpp"
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"

#include <stdexcept>

using easyatom::cst::gp::GPConfig;
using easyatom::cst::gp::Node;
using easyatom::cst::gp::Op;
using easyatom::cst::gp::ast_size;
using easyatom::cst::gp::clone;
using easyatom::cst::gp::eval;
using easyatom::cst::gp::run_gp;
using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;
using easyatom::ops::bind;
using easyatom::ops::random_phase_state;

EATEST_CASE(gp_eval_hoja_devuelve_terminal) {
    const std::size_t D = 256;
    State A = random_phase_state(D, 1);
    Node n; n.op = Op::Leaf; n.leaf_idx = 0;
    State out = eval(n, {A});
    EATEST_REQUIRE(fidelity(out, A) > 0.999);
}

EATEST_CASE(gp_eval_bind_y_ast_size) {
    const std::size_t D = 256;
    State A = random_phase_state(D, 1);
    State B = random_phase_state(D, 2);
    Node root; root.op = Op::Bind;
    root.left  = std::make_unique<Node>(); root.left->op  = Op::Leaf; root.left->leaf_idx  = 0;
    root.right = std::make_unique<Node>(); root.right->op = Op::Leaf; root.right->leaf_idx = 1;
    State out = eval(root, {A, B});
    State ref = bind(A, B);
    EATEST_REQUIRE(fidelity(out, ref) > 0.999);
    EATEST_REQUIRE(ast_size(root) == 3);
}

EATEST_CASE(gp_clone_produce_arbol_independiente) {
    Node root; root.op = Op::Permute; root.shift = 3;
    root.left = std::make_unique<Node>(); root.left->op = Op::Leaf; root.left->leaf_idx = 0;
    auto cp = clone(root);
    cp->shift = 99;   // mutar la copia
    EATEST_REQUIRE(root.shift == 3);
    EATEST_REQUIRE(cp->shift == 99);
}

EATEST_CASE(gp_descubre_bind_de_dos_terminales) {
    // Objetivo: dado A,B como terminales, encontrar una expresion cuyo eval
    // sea proporcional a bind(A,B). Fitness = fidelity(out, target).
    const std::size_t D = 1024;
    State A = random_phase_state(D, 11);
    State B = random_phase_state(D, 22);
    State target = bind(A, B);
    GPConfig cfg;
    cfg.population  = 80;
    cfg.generations = 25;
    cfg.max_depth   = 3;
    cfg.seed        = 7;
    auto r = run_gp({A, B}, [&](const State& s) {
        try { return fidelity(s, target); }
        catch (...) { return -1.0; }
    }, cfg);
    EATEST_REQUIRE(r.fitness > 0.95);
}

EATEST_CASE(gp_terminales_vacios_lanza) {
    bool t = false;
    try {
        run_gp({}, [](const State&){ return 0.0; }, {});
    } catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(gp_pop_cero_lanza) {
    const std::size_t D = 64;
    State A = random_phase_state(D, 1);
    GPConfig cfg; cfg.population = 0;
    bool t = false;
    try {
        run_gp({A}, [](const State&){ return 0.0; }, cfg);
    } catch (const std::invalid_argument&) { t = true; }
    EATEST_REQUIRE(t);
}

EATEST_CASE(gp_es_determinista_con_misma_semilla) {
    const std::size_t D = 256;
    State A = random_phase_state(D, 5);
    State B = random_phase_state(D, 6);
    State target = bind(A, B);
    auto fit = [&](const State& s) {
        try { return fidelity(s, target); } catch (...) { return -1.0; }
    };
    GPConfig cfg; cfg.population = 30; cfg.generations = 8; cfg.max_depth = 3; cfg.seed = 1234;
    auto r1 = run_gp({A, B}, fit, cfg);
    auto r2 = run_gp({A, B}, fit, cfg);
    EATEST_REQUIRE(r1.fitness == r2.fitness);
    EATEST_REQUIRE(r1.ast_size == r2.ast_size);
}

EATEST_CASE(gp_ignora_individuos_que_lanzan) {
    // Un fitness que siempre lanza debe traducirse en -inf y el GP termina
    // sin crashear; la semilla solo debe poder construir individuos validos.
    const std::size_t D = 64;
    State A = random_phase_state(D, 9);
    GPConfig cfg; cfg.population = 10; cfg.generations = 3; cfg.max_depth = 2; cfg.seed = 42;
    auto r = run_gp({A}, [](const State&) -> double {
        throw std::runtime_error("boom");
    }, cfg);
    // El mejor existira, con fitness muy negativo.
    EATEST_REQUIRE(r.best != nullptr);
    EATEST_REQUIRE(r.fitness < -1e20);
}
