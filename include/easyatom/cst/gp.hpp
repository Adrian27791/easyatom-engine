// =============================================================================
// EasyAtom · Ladrillo 18 — GP simbolico (programacion genetica sobre HDC).
// =============================================================================
//
// Busca, dentro del alfabeto fundamental del motor, una expresion algebraica
// que maximice una funcion de aptitud sobre H_D. Es la primera capa de
// "razonamiento simbolico generativo": en lugar de buscar correlaciones
// estadisticas, el motor compone OPERADORES.
//
// Alfabeto:  bind(a,b) | bundle(a,b) | permute(a,k) | unbind(a,b) | scale(a,r)
// Hojas:     terminales = vector<State> dado por el usuario (los simbolos
//            disponibles, p.ej. {S, O, K_R} de una tripleta).
//
// Aptitud: el usuario provee fitness: const State& -> double. La busqueda
// maximiza fitness(eval(expr)) con penalizacion lineal por |AST| para
// favorecer expresiones simples (Occam algorithmic).
//
// Determinismo: TODO depende de una semilla. Misma semilla + mismos inputs
// + misma fitness => mismo resultado. Sin RNG global. xorshift64 local.
//
// Sin dependencias externas. C++20 header-only. O(pop * gen * D * depth).
//
// Esta es la base sobre la que L20 (compile_law) puede DESCUBRIR formas
// equivalentes a una ley dada (compresion algoritmica) y sobre la que L21
// (teoremas) puede componer leyes. NO es un sistema de busqueda generica:
// es el motor algebraico que "experimenta" con su propio vocabulario.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "easyatom/cst/operator_map.hpp"   // scale (header-only)
#include "easyatom/hilbert/state.hpp"
#include "easyatom/ops/fundamental.hpp"

namespace easyatom::cst::gp {

using easyatom::hilbert::State;

// -----------------------------------------------------------------------------
// AST.
// -----------------------------------------------------------------------------

enum class Op : std::uint8_t {
    Leaf,        // hoja (terminal): toma terminals[leaf_idx]
    Bind,        // bind(L, R)
    Bundle,      // bundle({L, R}) (pesos +1)
    Permute,     // permute(L, shift)
    Unbind,      // unbind(L, R)
    Scale,       // scale(L, alpha)
};

struct Node {
    Op op = Op::Leaf;
    std::uint32_t leaf_idx = 0;       // si Leaf
    std::int32_t  shift    = 0;       // si Permute
    double        alpha    = 1.0;     // si Scale
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

[[nodiscard]] inline std::size_t ast_size(const Node& n) {
    std::size_t s = 1;
    if (n.left)  s += ast_size(*n.left);
    if (n.right) s += ast_size(*n.right);
    return s;
}

[[nodiscard]] inline std::unique_ptr<Node> clone(const Node& n) {
    auto r = std::make_unique<Node>();
    r->op = n.op;
    r->leaf_idx = n.leaf_idx;
    r->shift    = n.shift;
    r->alpha    = n.alpha;
    if (n.left)  r->left  = clone(*n.left);
    if (n.right) r->right = clone(*n.right);
    return r;
}

// -----------------------------------------------------------------------------
// Evaluacion del AST sobre un conjunto de terminales.
// -----------------------------------------------------------------------------
//
// Cualquier excepcion del nucleo HDC (dimensiones, unbind sobre cero) se
// propaga al llamador, pero la busqueda (run_gp) la captura y la usa para
// asignar fitness = -infinito al individuo, descartandolo. Asi el algoritmo
// nunca alucina valores: un programa que falla simplemente no se reproduce.

[[nodiscard]] inline State eval(const Node& n, const std::vector<State>& terms) {
    switch (n.op) {
        case Op::Leaf: {
            if (n.leaf_idx >= terms.size())
                throw std::out_of_range("gp::eval: leaf_idx fuera de rango.");
            return terms[n.leaf_idx];
        }
        case Op::Bind:
            return easyatom::ops::bind(eval(*n.left, terms), eval(*n.right, terms));
        case Op::Bundle:
            return easyatom::ops::bundle({eval(*n.left, terms), eval(*n.right, terms)});
        case Op::Permute:
            return easyatom::ops::permute(eval(*n.left, terms), n.shift);
        case Op::Unbind:
            return easyatom::ops::unbind(eval(*n.left, terms), eval(*n.right, terms));
        case Op::Scale:
            return easyatom::cst::scale(eval(*n.left, terms), n.alpha);
    }
    throw std::logic_error("gp::eval: opcode invalido.");
}

// -----------------------------------------------------------------------------
// Generador deterministico (xorshift64).
// -----------------------------------------------------------------------------

struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed)
        : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    std::uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    std::uint32_t u32(std::uint32_t mod) {
        return static_cast<std::uint32_t>(next() % mod);
    }
    double unit() {
        return static_cast<double>(next()) /
               static_cast<double>(UINT64_MAX);
    }
    std::int32_t int_in(std::int32_t lo, std::int32_t hi) {
        // [lo, hi]
        if (hi <= lo) return lo;
        const std::uint32_t span = static_cast<std::uint32_t>(hi - lo + 1);
        return lo + static_cast<std::int32_t>(u32(span));
    }
};

// -----------------------------------------------------------------------------
// Construccion aleatoria de un AST (metodo "grow").
// -----------------------------------------------------------------------------

[[nodiscard]] inline std::unique_ptr<Node> random_tree(
    Rng& rng, std::uint32_t terminals, int depth_remaining) {
    auto n = std::make_unique<Node>();
    // Si se acabo la profundidad, forzar hoja.
    const bool force_leaf =
        (depth_remaining <= 0) || (rng.unit() < 0.30);
    if (force_leaf) {
        n->op = Op::Leaf;
        n->leaf_idx = rng.u32(terminals);
        return n;
    }
    // 5 operadores internos, distribucion uniforme.
    const std::uint32_t pick = rng.u32(5);
    switch (pick) {
        case 0: n->op = Op::Bind;    break;
        case 1: n->op = Op::Bundle;  break;
        case 2: n->op = Op::Permute; n->shift = rng.int_in(-4, 4); break;
        case 3: n->op = Op::Unbind;  break;
        default: {
            n->op = Op::Scale;
            // alpha discreta en {-1, -0.5, 0.5, 1, 2} (sin 0 para no anular).
            const double opts[5] = {-1.0, -0.5, 0.5, 1.0, 2.0};
            n->alpha = opts[rng.u32(5)];
        } break;
    }
    n->left = random_tree(rng, terminals, depth_remaining - 1);
    if (n->op == Op::Bind || n->op == Op::Bundle || n->op == Op::Unbind) {
        n->right = random_tree(rng, terminals, depth_remaining - 1);
    }
    return n;
}

// -----------------------------------------------------------------------------
// Crossover y mutacion.
// -----------------------------------------------------------------------------

namespace detail {
inline void collect_nodes(Node* root, std::vector<Node*>& out) {
    if (!root) return;
    out.push_back(root);
    collect_nodes(root->left.get(),  out);
    collect_nodes(root->right.get(), out);
}
}  // namespace detail

inline void crossover(Node& a, Node& b, Rng& rng) {
    std::vector<Node*> na, nb;
    detail::collect_nodes(&a, na);
    detail::collect_nodes(&b, nb);
    if (na.empty() || nb.empty()) return;
    Node* x = na[rng.u32(static_cast<std::uint32_t>(na.size()))];
    Node* y = nb[rng.u32(static_cast<std::uint32_t>(nb.size()))];
    // Swap subarboles: clonamos para evitar liberar dos veces.
    auto cx = clone(*x);
    auto cy = clone(*y);
    *x = std::move(*cy);
    *y = std::move(*cx);
}

inline void mutate(Node& root, Rng& rng, std::uint32_t terminals,
                   int max_depth) {
    std::vector<Node*> all;
    detail::collect_nodes(&root, all);
    if (all.empty()) return;
    Node* tgt = all[rng.u32(static_cast<std::uint32_t>(all.size()))];
    // Reemplazar el subarbol por uno nuevo aleatorio de profundidad limitada.
    const int new_depth = rng.int_in(1, std::max(1, max_depth - 1));
    auto n = random_tree(rng, terminals, new_depth);
    *tgt = std::move(*n);
}

// -----------------------------------------------------------------------------
// Configuracion y resultado del GP.
// -----------------------------------------------------------------------------

struct GPConfig {
    std::uint32_t population   = 60;
    std::uint32_t generations  = 30;
    int           max_depth    = 4;
    double        complexity_lambda = 0.001;  // penalizacion por |AST|
    double        crossover_p  = 0.7;
    double        mutate_p     = 0.3;
    std::uint32_t tournament_k = 3;
    std::uint64_t seed         = 42;
};

struct GPResult {
    std::unique_ptr<Node> best;
    double                fitness = 0.0;
    std::size_t           ast_size = 0;
};

using Fitness = std::function<double(const State&)>;

// -----------------------------------------------------------------------------
// Bucle principal.
// -----------------------------------------------------------------------------

[[nodiscard]] inline GPResult run_gp(
    const std::vector<State>& terminals,
    const Fitness&            fit,
    const GPConfig&           cfg = {}) {
    if (terminals.empty())
        throw std::invalid_argument("gp::run_gp: terminales vacios.");
    if (cfg.population == 0 || cfg.generations == 0)
        throw std::invalid_argument("gp::run_gp: pop/gen = 0.");
    Rng rng(cfg.seed);
    const auto T = static_cast<std::uint32_t>(terminals.size());

    auto eval_fitness = [&](const Node& n) -> double {
        try {
            const State out = eval(n, terminals);
            const double base = fit(out);
            return base - cfg.complexity_lambda *
                   static_cast<double>(ast_size(n));
        } catch (...) {
            return -1e30;
        }
    };

    // Poblacion inicial.
    std::vector<std::unique_ptr<Node>> pop;
    pop.reserve(cfg.population);
    std::vector<double> fitn(cfg.population, -1e30);
    for (std::uint32_t i = 0; i < cfg.population; ++i) {
        pop.emplace_back(random_tree(rng, T, cfg.max_depth));
        fitn[i] = eval_fitness(*pop[i]);
    }

    auto tournament = [&]() -> std::uint32_t {
        std::uint32_t best_idx = rng.u32(cfg.population);
        double best_f = fitn[best_idx];
        for (std::uint32_t k = 1; k < cfg.tournament_k; ++k) {
            std::uint32_t c = rng.u32(cfg.population);
            if (fitn[c] > best_f) { best_f = fitn[c]; best_idx = c; }
        }
        return best_idx;
    };

    for (std::uint32_t g = 0; g < cfg.generations; ++g) {
        // Elitismo 1: mejor sobrevive sin tocar.
        std::uint32_t elite_idx = 0;
        for (std::uint32_t i = 1; i < cfg.population; ++i)
            if (fitn[i] > fitn[elite_idx]) elite_idx = i;
        std::vector<std::unique_ptr<Node>> next_pop;
        next_pop.reserve(cfg.population);
        std::vector<double> next_fit(cfg.population, -1e30);
        next_pop.emplace_back(clone(*pop[elite_idx]));
        next_fit[0] = fitn[elite_idx];
        for (std::uint32_t i = 1; i < cfg.population; ++i) {
            auto child = clone(*pop[tournament()]);
            if (rng.unit() < cfg.crossover_p) {
                auto partner = clone(*pop[tournament()]);
                crossover(*child, *partner, rng);
            }
            if (rng.unit() < cfg.mutate_p) {
                mutate(*child, rng, T, cfg.max_depth);
            }
            next_fit[i] = eval_fitness(*child);
            next_pop.emplace_back(std::move(child));
        }
        pop  = std::move(next_pop);
        fitn = std::move(next_fit);
    }

    // Mejor final.
    std::uint32_t best_idx = 0;
    for (std::uint32_t i = 1; i < cfg.population; ++i)
        if (fitn[i] > fitn[best_idx]) best_idx = i;
    GPResult r;
    r.best     = std::move(pop[best_idx]);
    r.fitness  = fitn[best_idx];
    r.ast_size = ast_size(*r.best);
    return r;
}

}  // namespace easyatom::cst::gp
