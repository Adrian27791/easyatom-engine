// =============================================================================
// easyatom/reason/proof.hpp  --  L26
//
// Verificador formal de cadenas de inferencia simbolica.
//
// Cada paso de la prueba referencia DOS triples ya conocidos (premisas o
// teoremas derivados en pasos anteriores) por sus indices en el "pool"
// acumulado, mas el triplet derivado que el paso afirma probar.
//
// La verificacion exige (regla unica de modus-ponens-composicional):
//   - lhs.object  == rhs.subject       (encadenable)
//   - infer(lhs.relation, rhs.relation) == derived.relation
//   - derived.subject == lhs.subject
//   - derived.object  == rhs.object
//   - ninguna relacion es Unknown
//
// API:
//   struct ProofStep   { size_t lhs_index; size_t rhs_index; Triplet derived; };
//   struct ProofCertificate {
//       bool   valid;
//       size_t first_invalid_index;     // == steps.size() si valid
//       std::vector<Triplet> pool;      // premisas + derivados validados
//   };
//   bool             verify_step(pool, step);
//   ProofCertificate check_proof(premises, steps);
//
// Header-only. Reusa easyatom::reason::infer (L21).
// =============================================================================

#ifndef EASYATOM_REASON_PROOF_HPP
#define EASYATOM_REASON_PROOF_HPP

#include <cstddef>
#include <vector>

#include "easyatom/cst/triplet.hpp"
#include "easyatom/cst/verbs.hpp"
#include "easyatom/reason/theorems.hpp"   // infer(Relation, Relation)

namespace easyatom::reason {

using easyatom::cst::Relation;
using easyatom::cst::Triplet;

struct ProofStep {
    std::size_t lhs_index = 0;
    std::size_t rhs_index = 0;
    Triplet     derived;
};

struct ProofCertificate {
    bool                 valid               = false;
    std::size_t          first_invalid_index = 0;
    std::vector<Triplet> pool;
};

[[nodiscard]] inline bool verify_step(
    const std::vector<Triplet>& pool, const ProofStep& step) {
    if (step.lhs_index >= pool.size()) return false;
    if (step.rhs_index >= pool.size()) return false;

    const Triplet& a = pool[step.lhs_index];
    const Triplet& b = pool[step.rhs_index];
    const Triplet& d = step.derived;

    if (a.relation == Relation::Unknown) return false;
    if (b.relation == Relation::Unknown) return false;
    if (d.relation == Relation::Unknown) return false;

    if (a.object != b.subject) return false;
    if (d.subject != a.subject) return false;
    if (d.object  != b.object)  return false;

    const Relation expected = infer(a.relation, b.relation);
    if (expected == Relation::Unknown) return false;
    if (expected != d.relation) return false;

    return true;
}

[[nodiscard]] inline ProofCertificate check_proof(
    const std::vector<Triplet>&   premises,
    const std::vector<ProofStep>& steps) {
    ProofCertificate cert;
    cert.pool = premises;

    for (std::size_t i = 0; i < steps.size(); ++i) {
        if (!verify_step(cert.pool, steps[i])) {
            cert.valid               = false;
            cert.first_invalid_index = i;
            return cert;
        }
        cert.pool.push_back(steps[i].derived);
    }

    cert.valid               = true;
    cert.first_invalid_index = steps.size();
    return cert;
}

}  // namespace easyatom::reason

#endif  // EASYATOM_REASON_PROOF_HPP
