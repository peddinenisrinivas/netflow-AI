"""
utils/diagnostics.py
━━━━━━━━━━━━━━━━━━━━
Standalone data diagnosis module — run this FIRST before training.
Identifies all root causes of poor model performance.

Usage (from backend/):
    python utils/diagnostics.py
    python utils/diagnostics.py --n 5000 --out diagnostics_report.txt
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from scipy.stats import pearsonr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pcap_parser import FEATURE_COLUMNS, _synthetic_fallback


def run_full_diagnosis(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Run full data diagnosis on *df*.
    Returns a dict with all findings; prints to stdout if verbose=True.
    """

    out_lines = []

    def p(line=""):
        if verbose:
            print(line)
        out_lines.append(line)

    p("=" * 65)
    p("  NETFLOW AI — DATA DIAGNOSIS REPORT")
    p(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    p("=" * 65)

    findings = {}

    # ── 1. Basic dataset summary ──────────────────────────────────────────────
    p("\n[1] DATASET SUMMARY")
    p("-" * 40)
    p(f"  Rows       : {len(df)}")
    p(f"  Columns    : {df.shape[1]}")
    p(f"  NaN cells  : {df.isnull().sum().sum()}")
    p(f"  Inf cells  : {np.isinf(df.select_dtypes(include='number')).sum().sum()}")

    # ── 2. Class distribution ─────────────────────────────────────────────────
    p("\n[2] CLASS DISTRIBUTION")
    p("-" * 40)
    if "label" in df.columns:
        vc = df["label"].value_counts()
        majority_cls  = vc.idxmax()
        majority_pct  = vc.max() / len(df) * 100
        imbalance_ratio = vc.max() / max(vc.min(), 1)
        for cls, cnt in vc.items():
            bar = "█" * int(cnt / len(df) * 30)
            p(f"  {cls:<15}: {cnt:>5} ({cnt/len(df)*100:5.1f}%)  {bar}")
        p(f"\n  Majority class   : {majority_cls!r} ({majority_pct:.1f}%)")
        p(f"  Imbalance ratio  : {imbalance_ratio:.1f}x  "
          f"{'⚠ HIGH — use SMOTE/class_weight' if imbalance_ratio > 2 else '✓ OK'}")
        p(f"\n  DUMMY CLASSIFIER BASELINE (predict always {majority_cls!r}):")
        p(f"    Accuracy  = {majority_pct:.1f}%")
        p(f"  → If your model gets ~{majority_pct:.0f}% accuracy, it is predicting")
        p(f"    the majority class for everything (imbalance not handled).")
        findings["majority_class"]     = majority_cls
        findings["majority_pct"]       = majority_pct
        findings["imbalance_ratio"]    = imbalance_ratio
        findings["imbalance_critical"] = imbalance_ratio > 2
    else:
        p("  No 'label' column found — unlabeled dataset.")
        findings["imbalance_critical"] = False

    # ── 3. Feature correlation ────────────────────────────────────────────────
    p("\n[3] FEATURE CORRELATION WITH LABEL (|Pearson r|)")
    p("-" * 40)
    feature_corrs = {}
    useless_feats = []
    if "label" in df.columns and _HAS_SCIPY:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df["label"].values)
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        for col in feature_cols:
            try:
                r, _ = pearsonr(df[col].fillna(0).values, y)
                feature_corrs[col] = abs(r)
            except Exception:
                feature_corrs[col] = 0.0
        ranked = sorted(feature_corrs.items(), key=lambda x: x[1], reverse=True)
        p("  Top 10 most informative:")
        for feat, corr in ranked[:10]:
            bar = "█" * int(corr * 28)
            p(f"    {feat:<25} {corr:.3f}  {bar}")
        p("  Weakest features (|r| < 0.05):")
        useless_feats = [(f, r) for f, r in ranked if r < 0.05]
        for feat, corr in useless_feats:
            p(f"    {feat:<25} {corr:.4f}  ⚠ NEAR-USELESS (adds noise)")
        findings["useless_features"] = [f for f, _ in useless_feats]
        findings["top_feature"]      = ranked[0][0]
        findings["top_corr"]         = ranked[0][1]
    else:
        p("  (scipy not available — skipping correlation analysis)")

    # ── 4. Label noise check ──────────────────────────────────────────────────
    p("\n[4] LABEL NOISE CHECK")
    p("-" * 40)
    if "label" in df.columns:
        from utils.pcap_parser import infer_labels, add_derived_features
        df2 = df.copy()
        if "suspicious_port" not in df2.columns:
            df2 = add_derived_features(df2)
        heuristic = infer_labels(df2)
        agree = (heuristic == df["label"].values).mean()
        p(f"  Heuristic vs true label agreement: {agree*100:.1f}%")
        if agree < 0.85:
            p(f"  ⚠ LABEL NOISE: {(1-agree)*100:.1f}% of labels disagree")
            p(f"    → May require manual review or stronger labeling rules")
        else:
            p(f"  ✓ Labels look consistent (>{agree*100:.0f}% agreement)")
        findings["label_noise_pct"] = (1 - agree) * 100

    # ── 5. Class separability ─────────────────────────────────────────────────
    p("\n[5] CLASS SEPARABILITY (per-class feature means)")
    p("-" * 40)
    if "label" in df.columns:
        key_feats = ["bytes_per_second", "avg_iat_ms", "flag_rst",
                     "pkt_count", "suspicious_port", "rst_intensity"]
        key_feats = [f for f in key_feats if f in df.columns]
        header = f"  {'Feature':<22}"
        classes = sorted(df["label"].unique())
        for c in classes:
            header += f"  {c[:10]:>12}"
        p(header)
        p("  " + "-" * (22 + len(classes) * 14))
        for f in key_feats:
            row = f"  {f:<22}"
            for c in classes:
                v = df[df["label"] == c][f].mean()
                row += f"  {v:>12.3f}"
            p(row)

    # ── 6. Vanilla model baseline ─────────────────────────────────────────────
    p("\n[6] VANILLA RANDOM FOREST BASELINE (no SMOTE, no tuning)")
    p("-" * 40)
    if "label" in df.columns:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from utils.pcap_parser import add_derived_features

        dfe = df.copy()
        if "suspicious_port" not in dfe.columns:
            dfe = add_derived_features(dfe)
        feat_cols = [c for c in FEATURE_COLUMNS if c in dfe.columns]
        le2 = LabelEncoder()
        y2  = le2.fit_transform(dfe["label"].values)
        X2  = dfe[feat_cols].fillna(0).values
        X2  = StandardScaler().fit_transform(X2)
        Xtr, Xvl, ytr, yvl = train_test_split(X2, y2, test_size=0.2,
                                               stratify=y2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xvl)
        vacc  = accuracy_score(yvl, yp)
        vprec = precision_score(yvl, yp, average="weighted", zero_division=0)
        vrec  = recall_score(yvl, yp,    average="weighted", zero_division=0)
        vf1   = f1_score(yvl, yp,        average="weighted", zero_division=0)
        p(f"  Accuracy  : {vacc:.4f}")
        p(f"  Precision : {vprec:.4f}")
        p(f"  Recall    : {vrec:.4f}")
        p(f"  F1-score  : {vf1:.4f}")
        findings["vanilla_accuracy"] = vacc
        findings["vanilla_f1"]       = vf1

    p("\n" + "=" * 65)
    p("  ROOT CAUSE SUMMARY")
    p("=" * 65)
    causes = []
    if findings.get("imbalance_critical"):
        causes.append(f"#1 CLASS IMBALANCE ({findings['majority_pct']:.0f}% majority)"
                      " → model predicts majority class only")
    if findings.get("useless_features"):
        causes.append(f"#2 NOISY FEATURES ({len(findings['useless_features'])} features"
                      f" with |r|<0.05): {findings['useless_features'][:3]}")
    if findings.get("label_noise_pct", 0) > 15:
        causes.append(f"#3 LABEL NOISE ({findings['label_noise_pct']:.1f}% mismatch)"
                      " → model trains on wrong targets")
    if not causes:
        causes.append("No critical issues found — apply standard tuning")

    for c in causes:
        p(f"  → {c}")
    p("")

    findings["report_lines"] = out_lines
    return findings


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=3000, help="Synthetic dataset size")
    parser.add_argument("--out", type=str, default=None, help="Save report to file")
    args = parser.parse_args()

    df = _synthetic_fallback(args.n)
    findings = run_full_diagnosis(df, verbose=True)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(findings["report_lines"]))
        print(f"\nReport saved to: {args.out}")
