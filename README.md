# EasyAtom Engine v4.3

**Zero-shot inductive drug repurposing — 16-layer algebraic pipeline, no GPU, no neural networks.**

[![TRL 4](https://img.shields.io/badge/TRL-4-blue)](https://easyatom-engine.web.app)
[![Layers](https://img.shields.io/badge/pipeline-L0--L15-green)](https://easyatom-engine.web.app)
[![Audit](https://img.shields.io/badge/audit-public-brightgreen)](https://easyatom-engine.web.app/audit/)
[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](LICENSE)

> *"We don't store answers. We store the laws that generate them."*

## What it does

EasyAtom scores every (drug, disease) pair across a 16-layer algebraic pipeline derived from first principles — Ising models, hyperdimensional computing, DWPC causal paths, spectral graph analysis — and surfaces the most credible novel repurposing hypotheses.

**Benchmark (zero-shot inductive, 922 drugs × 108 diseases, 22,380 pairs):**

| Recall@1 | Recall@10 | Recall@50 | MRR   | NDCG@10 |
|----------|-----------|-----------|-------|---------|
| 4.0%     | **28.6%** | 54.7%     | 0.113 | **0.822** |

No pair from the benchmark was used as a training signal. No GPU required.

## Key outputs

| Output | Count |
|--------|-------|
| L12 repurposing candidates | 266,561 |
| Platinum Standard (L2 ∩ L7_HIGH ∩ L10_CRITICAL) | 325 |
| DDI-safe cocktails (L14) | 47 |
| N-of-1 protocols (L15) | 20 |
| Total knowledge shards | 6,595 KB (mobile-ready) |

**Top hypothesis:** loratadine → PDE4B → Alzheimer's disease  
(L2=1.460, L7 Jaccard=1.00, L10=0.175, L8 DWPC=0.084)

## Pipeline architecture

```
L0  HDC          Hyperdimensional computing (D=10,000 binary hypervectors)
L1  Causal Sym.  Symbolic transitive chains (length ≤3)
L2  Hamiltonian  Ising energy model (22,380 pairs)
L3  Attractor    Fixed-point attractor via message passing
L4  Spectra      Spectral subgraph analysis (top-k eigenvalues)
L5  Explain      Rule-based causal chain string generation
L6  Semantic     Token-level disease/drug context overlap
L7  GapQueue     Gap detection: 41,396 raw → 2,640 post-cap candidates
L8  Knockout     DWPC perturbation (44/50 CRITICAL/HIGH)
L9  Distillation int8 quantization → 10 domain shards, SHA-256 per shard
L10 WorldModel   Forward-chaining urgency (705 CRITICAL)
L11 PK Safety    PK/target-occupancy filter (22,380 pairs)
L12 Repurposing  Extended scan: 266,561 candidates
L13 Com.Synergy  Combinatorial cocktail scoring (4,264 SYNERGISTIC_POTENT)
L14 DDI Safe     Drug-drug interaction filter (47 viable)
L15 N-of-1       Personalized clinical protocol generation (20 protocols)
```

## Corpus sources

The knowledge graph (2.56M triples base + 1.05M augmented) is derived from:
- **DrugBank v5.x** — drug INN names, drug-gene targets, MoA annotations
- **OMIM** — gene-disease association triples (MeSH-normalized)
- **CTD** — curated chemical-gene and chemical-disease interactions
- **Hetionet v1.0** — integrated drug-gene-disease metapaths (DWPC methodology)
- **STRING v11** — protein-protein interaction pairs (confidence ≥ 700)

All sources are openly licensed. No proprietary databases.

## Auditability

All pipeline outputs are publicly downloadable at **[easyatom-engine.web.app/audit/](https://easyatom-engine.web.app/audit/)**:

| File | Layer | Size |
|------|-------|------|
| `audit_L01_corpus.json` | L0-L1 corpus stats | 0.5 KB |
| `audit_L02_L03_validation.tsv` | L2-L3 benchmark | 1.5 MB |
| `audit_L04_repurposing.tsv` | L4 spectral | 5.4 MB |
| `audit_L06_causal_primes.tsv` | L6 semantic | 2.9 MB |
| `phase_l7_gapqueue_raw.tsv` | L7 raw gaps | 8.4 MB |
| `audit_L08_knockout.tsv` | L8 DWPC | 7.2 KB |
| `audit_L09_distilled.tsv` | L9 shards + SHA-256 | — |
| `audit_L10_worldmodel.tsv` | L10 urgency | 765 KB |
| `audit_L13_com_synergy.tsv` | L13 cocktails | 20.5 KB |
| `audit_L14_ddi_safe.tsv` | L14 DDI filter | 13.3 KB |
| `audit_L15_n1p_index.tsv` | L15 protocols | 3.4 KB |
| `audit_MANIFEST.json` | SHA-256 per shard | 4.3 KB |

**Corpus fingerprint (64-bit):** `0ff11993fb8746a9`  
**Full SHA-256:** `0ff11993fb8746a9f1eb3dcf241e074c486acae5c27d77f0a4a0dd17a6fb9997`

## Build (C++20)

```bash
# Linux / macOS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Windows (MSVC)
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

**Requirements:** C++20 compiler (GCC 12+, Clang 14+, MSVC 19.x). No external libraries.

## Run the pipeline

```bash
# Full pipeline (Python orchestrator, calls C++ binaries)
python easyatom_full_pipeline.py --out-dir /path/to/EasyAtom_FullPipeline

# Blind test (5 held-out drug-disease pairs)
python blind_test_runner.py --out-dir /path/to/EasyAtom_FullPipeline

# Single query (C++ CLI)
./build/easyatom predict loratadine
```

## Partnership

EasyHelpCare LLC is seeking pharma and biotech partners for:
- **Corpus augmentation**: ingest proprietary preclinical data (under NDA), rerun pipeline
- **Co-validation**: wet-lab validation of Platinum Standard candidates
- **Engine licensing**: full C++20 source for therapeutic-area deployments

**Contact:** Adrian Riveron (CTO) — EasyHelpCare LLC, Naples, Florida, USA  
**Web:** https://easyatom-engine.web.app  
**Technical report:** `paper/easyatom_arxiv.tex` (arXiv submission in progress)

## License

Source code: [MIT License](LICENSE)  
Pipeline outputs and audit data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Corpus (DrugBank-derived subset): DrugBank Academic License applies to derived data.

```

## Roadmap de ladrillos (orden estricto)

| # | Ladrillo | Estado |
|---|---|---|
| 0 | Multivectores de Clifford `Cl(p,q)` + producto geométrico | ✅ |
| 1 | Espacio de Hilbert simulado `H_D` + estados normalizados | ✅ |
| 2 | Operadores fundamentales geométricos (bind/bundle/permute/unbind) | ✅ |
| 3 | Métrica: Fisher-Rao + α-conexiones de Amari | ✅ |
| 4 | Topología: homología persistente (Vietoris-Rips, H_0/b_1, bottleneck) | ✅ |
| 5 | Dinámica: operador de Koopman (EDMD) | ✅ |
| 6 | Compilación de leyes (SINDy / STLSQ) | ✅ |
| 7 | API pública del Q-Kernel: `ingest / compose / collapse` | ✅ |
| 8 | Bindings: C ABI (`include/easyatom/c_api.h`) + JNI Android | ✅ |
| 9 | Introspección: `TracedKernel` (trayectoria, fidelidad/fisher por paso, JSON auditable) | ✅ |
| 10 | Decisor geométrico (`decide`) con validación multi-criterio | ✅ |
| 11 | Decoder semántico (es-ES) — respuestas en idioma humano | ✅ |
| C | Qubits sintéticos reales: amplitudes complejas + unitarios + Born + entrelazamiento | ✅ |

## Reglas de construcción

1. **Cada ladrillo tiene tests numéricos exactos antes del siguiente.**
2. **Sin dependencias externas en el núcleo.** Tests permitidos solo con
   framework propio mínimo.
3. **C++20 estándar puro.** Nada específico de plataforma en `include/`.
4. **No alucinaciones por construcción.** Cada operación que no tiene
   resultado matemáticamente definido devuelve un error explícito, no
   una aproximación silenciosa.

## Build (host Windows con clang++ del NDK)

```powershell
cd easyatom-engine
cmake -B build -G Ninja `
  -DCMAKE_CXX_COMPILER="C:/Users/DnR/AppData/Local/Android/SDK/ndk/29.0.14033849/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe" `
  -DCMAKE_C_COMPILER="C:/Users/DnR/AppData/Local/Android/SDK/ndk/29.0.14033849/toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
cmake --build build
ctest --test-dir build --output-on-failure
```

Si Ninja no está disponible, usar generador `"MinGW Makefiles"` o
`"Unix Makefiles"`.
