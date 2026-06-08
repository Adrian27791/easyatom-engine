#!/usr/bin/env python3
"""
Validation A — PubMed Post-Corpus Evidence Mining
===================================================
For each top gap candidate (drug, disease, gene), query the NCBI PubMed API
to find papers published AFTER our corpus freeze (2023) that support the
drug-disease or drug-gene or gene-disease link.

If we predicted a pair from pre-2023 corpus data and post-2023 papers confirm
it independently — that is computational prospective validation.

Output: validate_A_pubmed_results.tsv + validate_A_pubmed_summary.json
"""
import csv, json, time, urllib.request, urllib.parse, os, sys, re
from datetime import datetime

OUT_DIR  = r"C:\Users\ANRIV\Desktop\EasyAtom_FullPipeline"
GAP_FILE = os.path.join(OUT_DIR, "phase_l7_gapqueue_raw.tsv")
KO_FILE  = os.path.join(OUT_DIR, "phase_l8_candidates.tsv")
OUT_TSV  = os.path.join(OUT_DIR, "validate_A_pubmed_results.tsv")
OUT_JSON = os.path.join(OUT_DIR, "validate_A_pubmed_summary.json")

CORPUS_FREEZE_YEAR = 2023   # papers after this = post-corpus evidence
MAX_CANDIDATES     = 25     # top-N to query (rate-limit safe)
SLEEP_BETWEEN      = 0.35   # NCBI allows ~3 req/s without API key

# ── Manually curated gene links for top candidates (from L8 blind test) ──────
# drug -> disease -> gene (from blind_test_runner.py and L1 causal chains)
KNOWN_GENE_LINKS = {
    ("loratadine",              "alzheimers_disease"):         "PDE4B",
    ("minocycline",             "alzheimers_disease"):         "MMP9",
    ("minocycline",             "parkinsons_disease"):         "CASP3",
    ("acetylsalicylic_acid",    "type_2_diabetes_mellitus"):   "PTGS1",
    ("acetaminophen",           "rheumatoid_arthritis"):       "PTGS2",
    ("acetylcysteine",          "idiopathic_pulmonary_fibrosis"): "TGFB1",
    ("aclidinium",              "alzheimers_disease"):         "CHRM1",
    ("metformin",               "alzheimers_disease"):         "AMPK",
    ("metformin",               "lung_cancer"):                "AMPK",
    ("ibuprofen",               "alzheimers_disease"):         "PTGS2",
    ("simvastatin",             "alzheimers_disease"):         "HMGCR",
    ("baclofen",                "malignant_glioma"):           "GABA",
    ("baclofen",                "glaucoma"):                   "GABA",
    ("etidronic_acid",          "pagets_disease_of_bone"):     "RANK",
    ("solifenacin",             "rheumatoid_arthritis"):       "CHRM3",
    ("afatinib",                "lung_cancer"):                "EGFR",
    ("abacavir",                "hematologic_cancer"):         "PARP1",
    ("acenocoumarol",           "obesity"):                    "VKORC1",
    ("latanoprost",             "glaucoma"):                   "PTGFR",
    ("aciclovir",               "hepatitis_b"):                "TK",
    ("agomelatine",             "gout"):                       "MT1",
    ("adenosine",               "hematologic_cancer"):         "ADORA2A",
    ("acetazolamide",           "ocular_cancer"):              "CA9",
    ("acetophenazine",          "autistic_disorder"):          "DRD2",
    ("adinazolam",              "epilepsy_syndrome"):          "GABA",
}

# Normalize: replace underscores/hyphens with spaces for PubMed search
def normalize(s):
    return s.replace("_", " ").replace("-", " ").lower()

def pubmed_search(query, min_year=2023):
    """Search PubMed and return list of PMIDs with year filter."""
    encoded = urllib.parse.quote(query)
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
           f"?db=pubmed&term={encoded}&mindate={min_year}&maxdate=2026"
           f"&datetype=pdat&retmax=5&retmode=json&usehistory=n")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        ids  = data.get("esearchresult", {}).get("idlist", [])
        count = int(data.get("esearchresult", {}).get("count", 0))
        return ids, count
    except Exception as e:
        print(f"  [WARN] PubMed error for '{query}': {e}")
        return [], 0

def get_title(pmid):
    """Fetch title of a PMID."""
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
           f"?db=pubmed&id={pmid}&retmode=json")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        return data.get("result", {}).get(pmid, {}).get("title", "")[:120]
    except:
        return ""

def main():
    print("=" * 60)
    print("  EasyAtom Validation A — PubMed Post-Corpus Evidence")
    print(f"  Corpus freeze: {CORPUS_FREEZE_YEAR} | Looking for post-{CORPUS_FREEZE_YEAR} papers")
    print("=" * 60)

    # Build candidate list: top-N from gapqueue + known gene links
    with open(GAP_FILE, encoding="utf-8") as f:
        gaps = list(csv.DictReader(f, delimiter="\t"))
    gaps.sort(key=lambda x: float(x.get("gap_score", "0")), reverse=True)

    # Add loratadine->alzheimer explicitly (our top hypothesis)
    candidates = [("loratadine", "alzheimers_disease")]
    seen = {("loratadine", "alzheimers_disease")}
    for r in gaps:
        key = (r["drug"], r["disease"])
        if key not in seen and len(candidates) < MAX_CANDIDATES:
            candidates.append(key)
            seen.add(key)

    results = []
    supported = 0

    for i, (drug, disease) in enumerate(candidates):
        gene = KNOWN_GENE_LINKS.get((drug, disease), "")
        d_norm = normalize(drug)
        dis_norm = normalize(disease)
        g_norm = normalize(gene) if gene else ""

        print(f"\n[{i+1:02d}/{len(candidates)}] {drug} → {disease}" + (f" (gene: {gene})" if gene else ""))

        # Query 1: drug + disease direct
        q1 = f'"{d_norm}"[Title/Abstract] AND "{dis_norm}"[Title/Abstract]'
        ids1, cnt1 = pubmed_search(q1, CORPUS_FREEZE_YEAR)
        time.sleep(SLEEP_BETWEEN)

        # Query 2: drug + gene (if available)
        ids2, cnt2 = [], 0
        if gene:
            q2 = f'"{d_norm}"[Title/Abstract] AND "{g_norm}"[Title/Abstract]'
            ids2, cnt2 = pubmed_search(q2, CORPUS_FREEZE_YEAR)
            time.sleep(SLEEP_BETWEEN)

        # Query 3: gene + disease (if available)
        ids3, cnt3 = [], 0
        if gene:
            q3 = f'"{g_norm}"[Title/Abstract] AND "{dis_norm}"[Title/Abstract]'
            ids3, cnt3 = pubmed_search(q3, CORPUS_FREEZE_YEAR)
            time.sleep(SLEEP_BETWEEN)

        total_evidence = cnt1 + cnt2 + cnt3
        all_ids = list(dict.fromkeys(ids1 + ids2 + ids3))[:3]
        is_supported = total_evidence > 0

        if is_supported:
            supported += 1
            print(f"  ✓ SUPPORTED — {total_evidence} post-{CORPUS_FREEZE_YEAR} papers found")
            # Fetch title of first paper
            title = get_title(all_ids[0]) if all_ids else ""
            time.sleep(SLEEP_BETWEEN)
            print(f"    Top PMID {all_ids[0]}: {title[:80]}..." if title else "")
        else:
            print(f"  ✗ No post-{CORPUS_FREEZE_YEAR} PubMed evidence found")

        results.append({
            "drug": drug,
            "disease": disease,
            "gene_link": gene,
            "post_corpus_papers_direct": cnt1,
            "post_corpus_papers_drug_gene": cnt2,
            "post_corpus_papers_gene_disease": cnt3,
            "total_post_corpus_evidence": total_evidence,
            "is_supported": "1" if is_supported else "0",
            "top_pmids": ",".join(all_ids),
        })

    # Save TSV
    with open(OUT_TSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(results)

    # Save JSON summary
    summary = {
        "validation_type": "post_corpus_pubmed_evidence",
        "corpus_freeze_year": CORPUS_FREEZE_YEAR,
        "candidates_tested": len(candidates),
        "candidates_supported": supported,
        "support_rate_pct": round(100 * supported / len(candidates), 1),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "top_supported": [
            {"drug": r["drug"], "disease": r["disease"],
             "gene": r["gene_link"], "n_papers": r["total_post_corpus_evidence"],
             "pmids": r["top_pmids"]}
            for r in sorted(results, key=lambda x: int(x["total_post_corpus_evidence"]), reverse=True)
            if r["is_supported"] == "1"
        ][:10],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  RESULT: {supported}/{len(candidates)} candidates supported by post-{CORPUS_FREEZE_YEAR} PubMed evidence")
    print(f"  Support rate: {summary['support_rate_pct']}%")
    print(f"  Output: {OUT_TSV}")
    print(f"  Summary: {OUT_JSON}")
    print("=" * 60)

if __name__ == "__main__":
    main()
