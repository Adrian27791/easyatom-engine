// =============================================================================
// EasyAtom · Ladrillo 9 — Trace: introspección de la cadena de razonamiento.
// =============================================================================
//
// El motor por sí solo es opaco: una secuencia de operaciones produce un
// estado final |ψ⟩ y una decisión (argmax / distribución). Este ladrillo
// abre el motor: registra CADA operación, el estado resultante, y métricas
// geométricas (norma, fidelidad con el estado anterior, distancia
// Fisher-Rao entre las distribuciones inducidas por estados consecutivos
// proyectados sobre la codebook).
//
// El resultado es una TRAYECTORIA estructurada que el caller puede:
//   * inspeccionar evento por evento,
//   * proyectar sobre la codebook en cualquier paso (qué pensaba el motor
//     en ese momento),
//   * exportar como JSON simple para auditoría externa.
//
// FILOSOFÍA:
//   Nada se infiere a posteriori. El trace es el HECHO. Si no está en el
//   trace, no ocurrió. Esto es lo opuesto a un LLM, donde la "explicación"
//   es otro forward pass que puede mentir.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "easyatom/hilbert/state.hpp"
#include "easyatom/infogeo/fisher.hpp"
#include "easyatom/qkernel/qkernel.hpp"

namespace easyatom::introspect {

using easyatom::hilbert::State;
using easyatom::hilbert::fidelity;
using easyatom::infogeo::Distribution;
using easyatom::infogeo::fisher_rao_distance;
using easyatom::qkernel::QKernel;

enum class Op : std::uint8_t {
    Ingest,
    Compose,        // bind(role, filler)
    BundlePairs,    // bundle de varios bind
    Query,          // unbind(state, role)
    Collapse,       // proyección sobre codebook → distribution
    Argmax,         // ganador de un collapse
    Custom
};

[[nodiscard]] inline const char* op_name(Op o) noexcept {
    switch (o) {
        case Op::Ingest:      return "ingest";
        case Op::Compose:     return "compose";
        case Op::BundlePairs: return "bundle_pairs";
        case Op::Query:       return "query";
        case Op::Collapse:    return "collapse";
        case Op::Argmax:      return "argmax";
        case Op::Custom:      return "custom";
    }
    return "?";
}

// Un evento individual de la trayectoria.
struct TraceEvent {
    Op op;
    std::vector<std::string> inputs;       // nombres simbólicos de los inputs
    std::string output_label;              // nombre opcional del output

    // Estado resultante (si la op produce un State); vacío si no aplica.
    bool has_state = false;
    State result_state;

    // Distribución (si la op fue Collapse); vacía si no aplica.
    bool has_dist = false;
    Distribution result_dist;              // sobre `dist_names`
    std::vector<std::string> dist_names;

    // Métricas geométricas:
    //   * state_norm: ||result_state||² (1.0 si normalizado).
    //   * fidelity_prev: F(result_state, estado del evento State previo).
    //   * fisher_step: distancia Fisher-Rao entre la dist colapsada del
    //                  evento previo y la del actual (NaN si no aplica).
    double state_norm     = std::nan("");
    double fidelity_prev  = std::nan("");
    double fisher_step    = std::nan("");
};

// -----------------------------------------------------------------------------
// TracedKernel — wrapper de QKernel que registra todo en un Trace.
// -----------------------------------------------------------------------------
//
// API simétrica al QKernel pero cada método grava un TraceEvent. Si quieres
// el kernel "crudo" sin trazar, usa QKernel directamente.

class TracedKernel {
public:
    TracedKernel(std::size_t dim, std::uint64_t master_seed,
                 std::vector<std::string> probe_codebook = {})
        : kernel_(dim, master_seed),
          probe_codebook_(std::move(probe_codebook)) {
        // probe_codebook = lista opcional de nombres a usar como base para
        // proyectar el estado actual y calcular Fisher-Rao paso a paso.
    }

    [[nodiscard]] const QKernel& kernel() const noexcept { return kernel_; }
    [[nodiscard]]       QKernel& kernel()       noexcept { return kernel_; }
    [[nodiscard]] const std::vector<TraceEvent>& events() const noexcept {
        return events_;
    }
    void clear_trace() {
        events_.clear();
        last_dist_.reset();
    }

    /// Configura una codebook-sonda para introspección continua.
    void set_probe_codebook(std::vector<std::string> names) {
        probe_codebook_ = std::move(names);
    }

    [[nodiscard]] const std::vector<std::string>& probe_codebook() const noexcept {
        return probe_codebook_;
    }

    // --- Operaciones registradas ------------------------------------------

    const State& ingest(const std::string& name) {
        const State& s = kernel_.ingest(name);
        TraceEvent e;
        e.op = Op::Ingest;
        e.inputs = {name};
        e.output_label = name;
        record_state(e, s);
        events_.push_back(std::move(e));
        return s;
    }

    State compose(const std::string& role, const std::string& filler) {
        const State& r = kernel_.ingest(role);
        const State& f = kernel_.ingest(filler);
        State out = kernel_.compose(r, f);
        TraceEvent e;
        e.op = Op::Compose;
        e.inputs = {role, filler};
        e.output_label = role + ":" + filler;
        record_state(e, out);
        events_.push_back(std::move(e));
        return out;
    }

    State bundle_pairs(const std::vector<std::pair<std::string, std::string>>& pairs) {
        if (pairs.empty()) {
            throw std::invalid_argument("TracedKernel::bundle_pairs: vacío.");
        }
        std::vector<std::pair<State, State>> sp;
        sp.reserve(pairs.size());
        std::vector<std::string> in_names;
        in_names.reserve(2 * pairs.size());
        for (const auto& [r, f] : pairs) {
            sp.emplace_back(kernel_.ingest(r), kernel_.ingest(f));
            in_names.push_back(r);
            in_names.push_back(f);
        }
        State out = kernel_.bundle_pairs(sp);
        TraceEvent e;
        e.op = Op::BundlePairs;
        e.inputs = std::move(in_names);
        e.output_label = "bundle";
        record_state(e, out);
        events_.push_back(std::move(e));
        return out;
    }

    State query(const State& composite, const std::string& role) {
        const State& r = kernel_.ingest(role);
        State out = kernel_.query(composite, r);
        TraceEvent e;
        e.op = Op::Query;
        e.inputs = {role};
        e.output_label = "query(" + role + ")";
        record_state(e, out);
        events_.push_back(std::move(e));
        return out;
    }

    Distribution collapse(const State& s, const std::vector<std::string>& names) {
        Distribution d = kernel_.collapse(s, names);
        TraceEvent e;
        e.op = Op::Collapse;
        e.inputs = names;
        e.output_label = "collapse";
        e.has_dist = true;
        e.result_dist = d;
        e.dist_names = names;
        // Fisher-Rao paso a paso entre distribuciones consecutivas si las
        // dimensiones coinciden y ambas son válidas.
        if (last_dist_ && last_dist_->dim() == d.dim()) {
            try {
                e.fisher_step = fisher_rao_distance(*last_dist_, d);
            } catch (...) {
                e.fisher_step = std::nan("");
            }
        }
        last_dist_ = d;
        events_.push_back(std::move(e));
        return d;
    }

    std::string argmax_collapse(const State& s,
                                const std::vector<std::string>& names) {
        std::string winner = kernel_.argmax_collapse(s, names);
        TraceEvent e;
        e.op = Op::Argmax;
        e.inputs = names;
        e.output_label = winner;
        events_.push_back(std::move(e));
        return winner;
    }

    // --- Introspección expuesta -------------------------------------------

    /// Proyecta cualquier estado sobre la probe_codebook y devuelve
    /// la distribución resultante. NO graba evento (lectura pasiva).
    [[nodiscard]] Distribution probe(const State& s) const {
        if (probe_codebook_.empty()) {
            throw std::invalid_argument("probe: probe_codebook vacía.");
        }
        return kernel_.collapse(s, probe_codebook_);
    }

    /// Resumen agregado de la trayectoria (números puros, sin asignaciones
    /// dinámicas más allá del retorno).
    struct TraceSummary {
        std::size_t n_events     = 0;
        std::size_t n_collapses  = 0;
        std::size_t n_states     = 0;
        double      total_fisher_path = 0.0;  // suma de fisher_step válidos
        double      min_fidelity_prev = 1.0;  // mínima fidelidad entre estados
                                              // consecutivos (1.0 si ninguna).
    };

    [[nodiscard]] TraceSummary summarize() const {
        TraceSummary s;
        s.n_events = events_.size();
        bool any_fid = false;
        for (const auto& e : events_) {
            if (e.has_state) ++s.n_states;
            if (e.has_dist)  ++s.n_collapses;
            if (!std::isnan(e.fisher_step)) s.total_fisher_path += e.fisher_step;
            if (!std::isnan(e.fidelity_prev)) {
                if (!any_fid || e.fidelity_prev < s.min_fidelity_prev) {
                    s.min_fidelity_prev = e.fidelity_prev;
                }
                any_fid = true;
            }
        }
        if (!any_fid) s.min_fidelity_prev = 1.0;
        return s;
    }

    /// Exporta la trayectoria a JSON simple (sin librerías externas).
    [[nodiscard]] std::string to_json() const {
        std::string out = "{\"events\":[";
        for (std::size_t i = 0; i < events_.size(); ++i) {
            if (i) out += ",";
            out += event_to_json(events_[i]);
        }
        out += "]}";
        return out;
    }

private:
    QKernel kernel_;
    std::vector<TraceEvent> events_;
    std::vector<std::string> probe_codebook_;
    std::optional<Distribution> last_dist_;

    void record_state(TraceEvent& e, const State& s) {
        e.has_state = true;
        e.result_state = s;
        // Estados HD usan fases unitarias por componente: norm² ≈ dim.
        // Reportamos energía media por componente (∼1.0 para estados sanos).
        e.state_norm = (s.dim() > 0)
            ? s.norm_squared() / static_cast<double>(s.dim())
            : 0.0;
        // Fidelidad con el estado del último evento que llevaba State.
        for (long long i = static_cast<long long>(events_.size()) - 1; i >= 0; --i) {
            if (events_[static_cast<std::size_t>(i)].has_state) {
                e.fidelity_prev = fidelity(
                    s, events_[static_cast<std::size_t>(i)].result_state);
                break;
            }
        }
    }

    [[nodiscard]] static std::string esc(const std::string& s) {
        std::string o;
        o.reserve(s.size() + 2);
        for (char c : s) {
            switch (c) {
                case '"':  o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n";  break;
                case '\r': o += "\\r";  break;
                case '\t': o += "\\t";  break;
                default:   o += c;      break;
            }
        }
        return o;
    }

    [[nodiscard]] static std::string num_to_json(double x) {
        if (std::isnan(x)) return "null";
        if (std::isinf(x)) return x < 0 ? "\"-inf\"" : "\"inf\"";
        return std::to_string(x);
    }

    [[nodiscard]] static std::string event_to_json(const TraceEvent& e) {
        std::string s = "{";
        s += "\"op\":\"";       s += op_name(e.op); s += "\"";
        s += ",\"output\":\"";  s += esc(e.output_label); s += "\"";
        s += ",\"inputs\":[";
        for (std::size_t i = 0; i < e.inputs.size(); ++i) {
            if (i) s += ",";
            s += "\""; s += esc(e.inputs[i]); s += "\"";
        }
        s += "]";
        s += ",\"state_norm\":";    s += num_to_json(e.state_norm);
        s += ",\"fidelity_prev\":"; s += num_to_json(e.fidelity_prev);
        s += ",\"fisher_step\":";   s += num_to_json(e.fisher_step);
        if (e.has_dist) {
            s += ",\"distribution\":[";
            for (std::size_t i = 0; i < e.result_dist.dim(); ++i) {
                if (i) s += ",";
                s += "{\"name\":\""; s += esc(e.dist_names[i]);
                s += "\",\"p\":";   s += num_to_json(e.result_dist[i]); s += "}";
            }
            s += "]";
        }
        s += "}";
        return s;
    }
};

}  // namespace easyatom::introspect
