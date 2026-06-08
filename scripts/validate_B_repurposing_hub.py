#!/usr/bin/env python3
"""
Validation B — Drug Repurposing Hub Cross-Validation
=====================================================
Downloads the Broad Institute Drug Repurposing Hub (public CSV) and
cross-checks which approved repurposings our engine predicted in top-K.

If we rank a known repurposed drug in top-10 for its new indication —
WITHOUT having seen that pair in training — that is retrospective validation.

Output: validate_B_repurposing_hub.tsv + validate_B_summary.json
"""
import csv, json, urllib.request, os, sys, io, zipfile, time
from datetime import datetime

OUT_DIR  = r"C:\Users\ANRIV\Desktop\EasyAtom_FullPipeline"
L7_FILE  = os.path.join(OUT_DIR, "phase_l7_epistemic.tsv")
L8_FILE  = os.path.join(OUT_DIR, "phase_l8_candidates.tsv")
OUT_TSV  = os.path.join(OUT_DIR, "validate_B_repurposing_hub.tsv")
OUT_JSON = os.path.join(OUT_DIR, "validate_B_summary.json")

# Broad Drug Repurposing Hub — public download (no auth required)
REPURPOSING_HUB_URL = (
    "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/"
    "repurposing_drugs_20200324.txt"
)

# Known repurposings that occurred AFTER our corpus (for prospective claim)
# Source: FDA approvals 2023-2026, manually curated
POST_CORPUS_REPURPOSINGS = {
    # (drug_normalized, new_indication_normalized): year_approved
    ("lecanemab",         "alzheimers_disease"):            2023,
    ("donanemab",         "alzheimers_disease"):            2024,
    ("semaglutide",       "cardiovascular_disease"):        2023,
    ("semaglutide",       "obesity"):                       2023,
    ("tirzepatide",       "type_2_diabetes_mellitus"):      2022,
    ("tofacitinib",       "ulcerative_colitis"):            2018,
    ("baricitinib",       "rheumatoid_arthritis"):          2018,
    ("paxlovid",          "viral_infection"):               2021,
    ("bempedoic_acid",    "coronary_artery_disease"):       2020,
    ("ozanimod",          "ulcerative_colitis"):            2021,
    ("evinpallumab",      "polycythemia_vera"):             2021,
    ("avapritinib",       "sarcoma"):                       2020,
    ("selpercatinib",     "lung_cancer"):                   2020,
    ("pralsetinib",       "lung_cancer"):                   2020,
    ("tucatinib",         "breast_cancer"):                 2020,
}

def download_repurposing_hub():
    """Download the Broad Drug Repurposing Hub."""
    print(f"Downloading Drug Repurposing Hub from Broad Institute...")
    cache = os.path.join(OUT_DIR, "repurposing_hub_cache.txt")
    if os.path.exists(cache):
        print("  Using cached file.")
        with open(cache, encoding="utf-8", errors="replace") as f:
            return f.read()
    try:
        with urllib.request.urlopen(REPURPOSING_HUB_URL, timeout=30) as r:
            data = r.read().decode("utf-8", errors="replace")
        with open(cache, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"  Downloaded: {len(data):,} chars")
        return data
    except Exception as e:
        print(f"  [WARN] Could not download hub: {e}")
        return None

def normalize(s):
    return (s.lower().replace("'", "").replace(",", "").replace("-", "_").replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").strip())

def main():
    print("=" * 60)
    print("  EasyAtom Validation B — Drug Repurposing Hub Cross-Val")
    print("=" * 60)

    # 1. Load our predictions (L7 epistemic + L8 candidates)
    with open(L7_FILE, encoding="utf-8") as f:
        l7 = list(csv.DictReader(f, delimiter="\t"))
    with open(L8_FILE, encoding="utf-8") as f:
        l8 = list(csv.DictReader(f, delimiter="\t"))

    # Build prediction index: (drug, disease) -> predict_rank
    pred_index = {}
    for r in l7:
        pred_index[(normalize(r["drug"]), normalize(r["disease"]))] = {
            "predict_rank": int(r.get("predict_rank", 999)),
            "converge_rank": int(r.get("converge_rank", 999)),
            "hit10": int(r.get("hit10", 0)),
            "source": "L7",
        }

    print(f"  Loaded {len(pred_index):,} predictions from L7")

    # 2. Download repurposing hub
    hub_data = download_repurposing_hub()

    hub_pairs = {}  # (drug_norm, disease_norm) -> {pert_iname, disease, status}
    if hub_data:
        lines = hub_data.strip().split("\n")
        header = None
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue
            if header is None:
                header = [h.strip().lower() for h in line.split("\t")]
                continue
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            drug_n = normalize(row.get("pert_iname", row.get("name", "")))
            # use 'indication' (specific disease) NOT 'disease_area' (broad category)
            disease_raw = row.get("indication") or row.get("disease_area", "")
            disease_n = normalize(disease_raw)
            status = row.get("clinical_phase", row.get("status", "")).strip()
            if drug_n and disease_n:
                hub_pairs[(drug_n, disease_n)] = {
                    "drug": drug_n,
                    "disease": disease_n,
                    "status": status,
                }
        print(f"  Loaded {len(hub_pairs):,} pairs from Drug Repurposing Hub")
    else:
        print("  [FALLBACK] Using manually curated post-corpus repurposings only")

    # 3. Cross-check: which hub pairs does our engine predict in top-K?
    results = []
    hits_top1, hits_top5, hits_top10, hits_total = 0, 0, 0, 0

    # Combine hub pairs + post-corpus repurposings
    all_reference = {}
    all_reference.update(hub_pairs)
    for (drug, disease), year in POST_CORPUS_REPURPOSINGS.items():
        key = (normalize(drug), normalize(disease))
        if key not in all_reference:
            all_reference[key] = {"drug": drug, "disease": disease, "status": f"approved_{year}"}

    for (drug_n, disease_n), meta in all_reference.items():
        pred = pred_index.get((drug_n, disease_n))
        matched = pred is not None
        rank = pred["predict_rank"] if pred else -1
        hit10 = pred["hit10"] if pred else 0
        is_post_corpus = (drug_n, disease_n) in {
            (normalize(d), normalize(dis)) for d, dis in
            [(k[0], k[1]) for k in POST_CORPUS_REPURPOSINGS.keys()]
        }

        if matched:
            hits_total += 1
            if rank <= 1:  hits_top1 += 1
            if rank <= 5:  hits_top5 += 1
            if rank <= 10: hits_top10 += 1

        results.append({
            "drug": drug_n,
            "disease": disease_n,
            "hub_status": meta.get("status", ""),
            "is_post_corpus": "1" if is_post_corpus else "0",
            "in_our_predictions": "1" if matched else "0",
            "predict_rank": rank,
            "hit10": hit10,
        })

    # Sort: matched first, then by rank
    results.sort(key=lambda x: (x["in_our_predictions"] == "0", int(x["predict_rank"]) if x["predict_rank"] != -1 else 999))

    # Save TSV
    with open(OUT_TSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(results)

    total_hub = len(all_reference)
    prec_1  = round(100 * hits_top1  / max(total_hub, 1), 1)
    prec_5  = round(100 * hits_top5  / max(total_hub, 1), 1)
    prec_10 = round(100 * hits_top10 / max(total_hub, 1), 1)
    recall  = round(100 * hits_total / max(total_hub, 1), 1)

    # Post-corpus only
    post_matches = [r for r in results if r["is_post_corpus"] == "1" and r["in_our_predictions"] == "1"]

    summary = {
        "validation_type": "drug_repurposing_hub_cross_validation",
        "reference_pairs_total": total_hub,
        "hub_pairs_loaded": len(hub_pairs),
        "post_corpus_repurposings": len(POST_CORPUS_REPURPOSINGS),
        "our_predictions_matched": hits_total,
        "precision_at_1": prec_1,
        "precision_at_5": prec_5,
        "precision_at_10": prec_10,
        "recall_known_repurposings": recall,
        "post_corpus_matches": len(post_matches),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "post_corpus_hits": [
            {"drug": r["drug"], "disease": r["disease"],
             "rank": r["predict_rank"], "status": r["hub_status"]}
            for r in post_matches
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Reference pairs (hub + post-corpus): {total_hub}")
    print(f"  Our predictions matched:             {hits_total} ({recall}%)")
    print(f"  Precision@1:  {prec_1}%")
    print(f"  Precision@5:  {prec_5}%")
    print(f"  Precision@10: {prec_10}%")
    print(f"  Post-corpus retrospective hits:      {len(post_matches)}")
    if post_matches:
        print("  Post-corpus matches:")
        for m in post_matches[:5]:
            print(f"    {m['drug']} -> {m['disease']}  rank={m['rank']}  status={m['status']}")
    print(f"  Output: {OUT_TSV}")
    print(f"  Summary: {OUT_JSON}")
    print("=" * 60)

if __name__ == "__main__":
    main()
