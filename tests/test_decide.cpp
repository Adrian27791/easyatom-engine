#include "test_framework.hpp"
#include "easyatom/decide/decisor.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using easyatom::decide::Decision;
using easyatom::decide::DecisionKind;
using easyatom::decide::DecisionPolicy;
using easyatom::decide::decide;
using easyatom::decide::shannon_entropy;
using easyatom::decide::effective_support;
using easyatom::decide::fisher_consistency_check;
using easyatom::infogeo::Distribution;

EATEST_CASE(decide_acepta_pico_claro) {
    Distribution p({0.85, 0.10, 0.05});
    auto d = decide(p, {"a","b","c"});
    EATEST_REQUIRE(d.kind == DecisionKind::Accept);
    EATEST_REQUIRE(d.winner_name == "a");
    EATEST_REQUIRE(d.runner_up_name == "b");
    EATEST_REQUIRE(d.is_actionable());
    EATEST_REQUIRE_NEAR(d.confidence, 0.85, 1e-12);
    EATEST_REQUIRE_NEAR(d.margin, 0.75, 1e-12);
}

EATEST_CASE(decide_marca_ambiguous_si_top1_top2_cerca) {
    Distribution p({0.46, 0.44, 0.10});
    auto d = decide(p, {"a","b","c"});
    EATEST_REQUIRE(d.kind == DecisionKind::Ambiguous);
    EATEST_REQUIRE(d.winner_name == "a");
    EATEST_REQUIRE(d.runner_up_name == "b");
    EATEST_REQUIRE(!d.is_actionable());
    bool found = false;
    for (auto& r : d.reason_codes) if (r == "low_margin") found = true;
    EATEST_REQUIRE(found);
}

EATEST_CASE(decide_se_abstiene_si_confianza_baja) {
    Distribution p({0.20, 0.18, 0.16, 0.16, 0.15, 0.15});
    auto d = decide(p, {"a","b","c","d","e","f"});
    // confianza 0.20 < 0.35 default → Abstain (se evalúa antes que entropy)
    EATEST_REQUIRE(d.kind == DecisionKind::Abstain ||
                   d.kind == DecisionKind::Degenerate);
    EATEST_REQUIRE(!d.is_actionable());
}

EATEST_CASE(decide_marca_degenerate_si_uniforme) {
    Distribution p = Distribution::uniform(8);
    DecisionPolicy pol; pol.min_confidence = 0.0; pol.min_margin = 0.0;
    auto d = decide(p, {"a","b","c","d","e","f","g","h"}, pol);
    EATEST_REQUIRE(d.kind == DecisionKind::Degenerate);
    EATEST_REQUIRE_NEAR(d.entropy_ratio, 1.0, 1e-9);
}

EATEST_CASE(decide_invalid_si_nombres_no_coinciden) {
    Distribution p({0.6, 0.4});
    auto d = decide(p, {"a","b","c"});
    EATEST_REQUIRE(d.kind == DecisionKind::Invalid);
}

EATEST_CASE(decide_dim_uno_acepta_trivial) {
    Distribution p({1.0});
    auto d = decide(p, {"unico"});
    EATEST_REQUIRE(d.kind == DecisionKind::Accept);
    EATEST_REQUIRE(d.winner_name == "unico");
    EATEST_REQUIRE_NEAR(d.confidence, 1.0, 1e-12);
}

EATEST_CASE(decide_strict_es_mas_exigente) {
    Distribution p({0.50, 0.40, 0.10});
    auto easy = decide(p, {"a","b","c"});
    auto hard = decide(p, {"a","b","c"}, DecisionPolicy::strict());
    EATEST_REQUIRE(easy.kind == DecisionKind::Accept);
    EATEST_REQUIRE(hard.kind != DecisionKind::Accept);
}

EATEST_CASE(decide_entropia_y_n_efectivo_correctos) {
    Distribution u = Distribution::uniform(4);
    EATEST_REQUIRE_NEAR(shannon_entropy(u), std::log(4.0), 1e-12);
    EATEST_REQUIRE_NEAR(effective_support(u), 4.0, 1e-9);
    Distribution pico({1.0, 0.0, 0.0, 0.0});
    EATEST_REQUIRE_NEAR(shannon_entropy(pico), 0.0, 1e-12);
    EATEST_REQUIRE_NEAR(effective_support(pico), 1.0, 1e-9);
}

EATEST_CASE(decide_fisher_consistency_check_ok) {
    Distribution a({0.5, 0.5});
    Distribution b({0.5, 0.5});
    auto c = fisher_consistency_check(a, b, 0.01);
    EATEST_REQUIRE(c.passed);
    EATEST_REQUIRE_NEAR(c.distance, 0.0, 1e-12);
}

EATEST_CASE(decide_fisher_consistency_check_fail) {
    Distribution a({0.99, 0.01});
    Distribution b({0.01, 0.99});
    auto c = fisher_consistency_check(a, b, 0.1);
    EATEST_REQUIRE(!c.passed);
    EATEST_REQUIRE(c.distance > 0.5);
}
