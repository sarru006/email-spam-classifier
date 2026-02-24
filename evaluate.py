"""
evaluate.py
-----------
Run this script to reproduce all results from Chapter 4 of the project report.

Usage:
    python evaluate.py

What it does:
    1. Loads the dataset (SMS)
    2. Preprocesses and vectorizes the test split
    3. Loads all 3 trained models from models/
    4. Prints metrics for each model (accuracy, precision, recall, F1, MCC)
    5. Prints confusion matrices
    6. Shows top spam/ham features from Logistic Regression
    7. Saves all charts to evaluation_outputs/
"""

import os
import sys
import json
import time
import joblib
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# == Make sure project root is importable (works from any directory) ==
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # so relative paths like models/ always work

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    classification_report,
)

# Adapted imports for the current project structure
from data_loader import load_dataset, train_test_split_stratified
from preprocessor import preprocess_batch

# == Output folder ==
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "Naive Bayes":        "#e74c3c",
    "SVM":                "#2ecc71",
    "Logistic Regression":"#3498db",
}

MODEL_FILES = {
    "Naive Bayes":         "models/nb_model.pkl",
    "SVM":                 "models/svm_model.pkl",
    "Logistic Regression": "models/lr_model.pkl",
}

DIVIDER = "=" * 65


# ====================================================================
# STEP 1 - Load & split dataset
# ====================================================================

def load_test_set():
    print(DIVIDER)
    print("STEP 1 - Loading dataset")
    print(DIVIDER)

    df = load_dataset(deduplicate=True)
    print(f"  Total documents : {len(df):,}")
    print(f"  Spam            : {(df.binary_label == 1).sum():,}  "
          f"({(df.binary_label==1).mean()*100:.1f}%)")
    print(f"  Ham             : {(df.binary_label == 0).sum():,}  "
          f"({(df.binary_label==0).mean()*100:.1f}%)")

    _, test_df = train_test_split_stratified(df)
    print(f"\n  Test split (20%) : {len(test_df):,} documents")
    print(f"  Test spam        : {(test_df.binary_label==1).sum():,}")
    print(f"  Test ham         : {(test_df.binary_label==0).sum():,}")

    return test_df


# ====================================================================
# STEP 2 - Preprocess + vectorize
# ====================================================================

def vectorize(test_df):
    print(f"\n{DIVIDER}")
    print("STEP 2 - Preprocessing & TF-IDF Vectorization")
    print(DIVIDER)

    t0 = time.time()
    test_texts = preprocess_batch(test_df["text"].tolist())
    print(f"  Preprocessing done in {time.time()-t0:.1f}s")

    vectorizer_path = "models/tfidf_vectorizer.pkl"
    if not os.path.exists(vectorizer_path):
        print(f"  ERROR: {vectorizer_path} not found. Run training first.")
        sys.exit(1)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    X_test = vectorizer.transform(test_texts)
    y_test = test_df["binary_label"].values

    print(f"  Vocabulary size  : {len(vectorizer.get_feature_names_out()):,} features")
    print(f"  Feature matrix   : {X_test.shape[0]:,} docs x {X_test.shape[1]:,} features")

    return X_test, y_test, vectorizer


# ====================================================================
# STEP 3 - Evaluate each model
# ====================================================================

def evaluate_models(X_test, y_test):
    print(f"\n{DIVIDER}")
    print("STEP 3 - Model Evaluation")
    print(DIVIDER)

    results = {}

    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            print(f"  SKIP {name}: {path} not found.")
            continue

        print(f"\n  -- {name} --")
        clf    = joblib.load(path)
        t0     = time.time()
        y_pred = clf.predict(X_test)
        ms     = (time.time() - t0) * 1000 / len(y_test)

        try:    y_prob = clf.predict_proba(X_test)[:, 1]
        except: y_prob = None

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec   = recall_score(y_test, y_pred,    pos_label=1, zero_division=0)
        f1    = f1_score(y_test, y_pred,         pos_label=1, zero_division=0)
        mcc   = matthews_corrcoef(y_test, y_pred)

        print(f"    Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"    Spam Precision  : {prec:.4f}")
        print(f"    Spam Recall     : {rec:.4f}")
        print(f"    Spam F1-Score   : {f1:.4f}")
        print(f"    MCC             : {mcc:.4f}")
        print(f"    Inference speed : {ms:.4f} ms / email")
        print(f"\n    Confusion Matrix:")
        print(f"      TN={tn:4d}  FP={fp:4d}")
        print(f"      FN={fn:4d}  TP={tp:4d}")
        print(f"\n    Classification Report:")
        print("    " + classification_report(
            y_test, y_pred, target_names=["Ham", "Spam"]
        ).replace("\n", "\n    "))

        results[name] = {
            "y_pred": y_pred, "y_prob": y_prob,
            "cm": cm,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "mcc": mcc, "ms": ms,
        }

    return results


# ====================================================================
# STEP 4 - Feature importance (Logistic Regression)
# ====================================================================

def feature_importance(vectorizer):
    print(f"\n{DIVIDER}")
    print("STEP 4 - Top TF-IDF Features (Logistic Regression Coefficients)")
    print(DIVIDER)

    lr_path = MODEL_FILES["Logistic Regression"]
    if not os.path.exists(lr_path):
        print("  LR model not found, skipping.")
        return None, None

    lr  = joblib.load(lr_path)
    coef = lr.coef_[0]
    features = vectorizer.get_feature_names_out()

    top_spam_idx = np.argsort(coef)[-15:][::-1]
    top_ham_idx  = np.argsort(coef)[:15]

    spam_feats = [(features[i], round(float(coef[i]), 4)) for i in top_spam_idx]
    ham_feats  = [(features[i], round(float(coef[i]), 4)) for i in top_ham_idx]

    print("\n  Top 15 SPAM indicator words:")
    for word, score in spam_feats:
        scaled_score = max(0, int(score / 0.5))
        bar = "#" * scaled_score
        print(f"    {word:<18} {score:+.4f}  {bar}")

    print("\n  Top 15 HAM indicator words:")
    for word, score in ham_feats:
        scaled_score = max(0, int(abs(score) / 0.3))
        bar = "#" * scaled_score
        print(f"    {word:<18} {score:+.4f}  {bar}")

    return spam_feats, ham_feats


# ====================================================================
# STEP 5 - Save charts
# ====================================================================

def save_charts(results, y_test, spam_feats, ham_feats):
    print(f"\n{DIVIDER}")
    print("STEP 5 - Saving Charts")
    print(DIVIDER)

    names = list(results.keys())
    if not names:
        return
        
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                          "axes.spines.right": False})

    # Chart 1: Confusion Matrices
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1: axes = [axes]
    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")

    for ax, name in zip(axes, names):
        r  = results[name]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, linewidths=1, linecolor="white")
        ax.set_title(f"{name}\nAcc={r['accuracy']:.2%} | MCC={r['mcc']:.4f}",
                     fontweight="bold")
        ax.set_xticklabels(["Pred HAM", "Pred SPAM"])
        ax.set_yticklabels(["Actual HAM", "Actual SPAM"], rotation=0)
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() * 0.6 else "black"
                ax.text(j + 0.5, i + 0.5,
                        f"{labels[i][j]}\n{val:,}",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "1_confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Chart 2: Metric Comparison
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    labels  = ["Accuracy", "Spam\nPrecision", "Spam\nRecall", "Spam\nF1", "MCC"]
    x = np.arange(len(metrics)); w = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Model Metric Comparison", fontsize=15, fontweight="bold")

    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * w, vals, w, label=name,
                      color=COLORS.get(name, "gray"), alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x + w * (len(names)-1)/2); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.axhline(0.95, ls="--", color="gray", alpha=0.4, lw=1)
    ax.legend(fontsize=10); plt.tight_layout()

    path = os.path.join(OUT_DIR, "2_metric_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Chart 3: ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ROC & Precision-Recall Curves", fontsize=15, fontweight="bold")

    for name in names:
        r = results[name]
        if r["y_prob"] is None:
            continue
        y_prob = r["y_prob"]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=COLORS.get(name, "gray"), lw=2,
                     label=f"{name} (AUC={roc_auc:.4f})")

        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec_c, prec_c)
        axes[1].plot(rec_c, prec_c, color=COLORS.get(name, "gray"), lw=2,
                     label=f"{name} (AUC={pr_auc:.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=9)

    axes[1].axhline(y_test.mean(), color="gray", ls="--", lw=1,
                    label=f"Baseline (spam={y_test.mean():.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "3_roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Chart 4: Feature Importance
    if spam_feats and ham_feats:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle("Top TF-IDF Features",
                     fontsize=14, fontweight="bold")

        words_s = [f[0] for f in spam_feats]
        coef_s  = [f[1] for f in spam_feats]
        axes[0].barh(words_s[::-1], coef_s[::-1],
                     color="#e74c3c", alpha=0.85, edgecolor="white")
        axes[0].set_xlabel("LR Coefficient (positive -> spam)")
        axes[0].set_title("Top 15 SPAM Indicator Words")
        for i, (w, v) in enumerate(zip(words_s[::-1], coef_s[::-1])):
            axes[0].text(v + 0.05, i, f"{v:.2f}", va="center",
                         fontsize=9, fontweight="bold")

        words_h = [f[0] for f in ham_feats]
        coef_h  = [abs(f[1]) for f in ham_feats]
        axes[1].barh(words_h[::-1], coef_h[::-1],
                     color="#2ecc71", alpha=0.85, edgecolor="white")
        axes[1].set_xlabel("|LR Coefficient| (negative -> ham)")
        axes[1].set_title("Top 15 HAM Indicator Words")
        for i, (w, v) in enumerate(zip(words_h[::-1], coef_h[::-1])):
            axes[1].text(v + 0.02, i, f"{v:.2f}", va="center",
                         fontsize=9, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(OUT_DIR, "4_feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ====================================================================
# STEP 6 - Save JSON results
# ====================================================================

def save_results_json(results):
    out = {}
    for name, r in results.items():
        out[name] = {
            "accuracy":   round(float(r["accuracy"]),  4),
            "precision":  round(float(r["precision"]), 4),
            "recall":     round(float(r["recall"]),    4),
            "f1":         round(float(r["f1"]),        4),
            "mcc":        round(float(r["mcc"]),       4),
            "ms_per_email": round(float(r["ms"]),      6),
            "confusion_matrix": {
                "TN": int(r["tn"]), "FP": int(r["fp"]),
                "FN": int(r["fn"]), "TP": int(r["tp"]),
            },
        }
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")


# ====================================================================
# SUMMARY TABLE
# ====================================================================

def print_summary(results):
    print(f"\n{DIVIDER}")
    print("SUMMARY TABLE")
    print(DIVIDER)
    print(f"  {'Model':<22} {'Accuracy':>10} {'MCC':>8} {'Spam Prec':>11} "
          f"{'Spam Rec':>10} {'Spam F1':>9} {'ms/email':>10}")
    print("  " + "-" * 83)
    for name, r in results.items():
        marker = "  <- best" if name == "SVM" else ""
        print(f"  {name:<22} {r['accuracy']:>10.4f} {r['mcc']:>8.4f} "
              f"{r['precision']:>11.4f} {r['recall']:>10.4f} "
              f"{r['f1']:>9.4f} {r['ms']:>9.4f}{marker}")
    print()


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    print("\n" + DIVIDER)
    print(" EMAIL SPAM CLASSIFIER - EVALUATION SCRIPT")
    print(DIVIDER + "\n")

    test_df              = load_test_set()
    X_test, y_test, vectorizer  = vectorize(test_df)
    results              = evaluate_models(X_test, y_test)
    spam_feats, ham_feats = feature_importance(vectorizer)

    save_charts(results, y_test, spam_feats, ham_feats)
    save_results_json(results)
    print_summary(results)

    print(f"All outputs saved to: {OUT_DIR}/")
    print("Done.\n")
