"""
Standalone training script to generate the portable `mbti_predictor.joblib`
bundle locally from `mbti_1.csv.zip` or `mbti_1.csv`.

Usage:
    python train.py --data path/to/mbti_1.csv.zip
"""

from __future__ import annotations

import argparse
import os
import zipfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from model_engine import DIMENSIONS, clean_text_single, extract_stylometry_single

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGBM = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def load_dataset(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            csv_name = [f for f in zf.namelist() if f.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)

    df = df.dropna(subset=["type", "posts"]).reset_index(drop=True)
    df["type"] = df["type"].str.strip().str.upper()
    for col, pos_l, _, _ in DIMENSIONS:
        idx = {"IE": 0, "NS": 1, "TF": 2, "JP": 3}[col]
        df[col] = (df["type"].str[idx] == pos_l).astype(int)
    return df


def make_gbdt():
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_features(texts, remove_mbti_words=True, fitted=None):
    cleaned = [clean_text_single(t, remove_mbti_words=remove_mbti_words) for t in texts]
    stylo_raw = np.array([extract_stylometry_single(t)[0] for t in texts], dtype=np.float32)

    if fitted is None:
        word_tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            max_features=35000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=5,
            max_df=0.90,
            max_features=25000,
            sublinear_tf=True,
        )
        stylo_scaler = StandardScaler()
        stylo_norm = Normalizer(norm="l2")
        svd = TruncatedSVD(n_components=120, random_state=RANDOM_STATE)

        X_word = word_tfidf.fit_transform(cleaned)
        X_char = char_tfidf.fit_transform(cleaned)
        X_stylo_scaled = stylo_scaler.fit_transform(stylo_raw)
        X_stylo_sp = sparse.csr_matrix(stylo_norm.fit_transform(X_stylo_scaled) * 0.35)
        X_svd = svd.fit_transform(sparse.hstack([X_word, X_char], format="csr")).astype(np.float32)

        fitted = {
            "word_tfidf": word_tfidf,
            "char_tfidf": char_tfidf,
            "stylo_scaler": stylo_scaler,
            "stylo_norm": stylo_norm,
            "svd": svd,
        }
    else:
        X_word = fitted["word_tfidf"].transform(cleaned)
        X_char = fitted["char_tfidf"].transform(cleaned)
        X_stylo_scaled = fitted["stylo_scaler"].transform(stylo_raw)
        X_stylo_sp = sparse.csr_matrix(fitted["stylo_norm"].transform(X_stylo_scaled) * 0.35)
        X_svd = fitted["svd"].transform(sparse.hstack([X_word, X_char], format="csr")).astype(np.float32)

    X_sp = sparse.hstack([X_word, X_char, X_stylo_sp], format="csr")
    X_dn = np.hstack([X_svd, X_stylo_scaled]).astype(np.float32)
    return X_sp, X_dn, fitted


def fit_axis(X_tr_sp, X_tr_dn, y_tr, X_va_sp=None, X_va_dn=None, y_va=None, fixed_w=None, fixed_thr=None):
    sw_tr = compute_sample_weight("balanced", y_tr)

    lr = LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_tr_sp, y_tr)

    svc = CalibratedClassifierCV(
        estimator=LinearSVC(C=0.25, max_iter=3000, random_state=RANDOM_STATE),
        method="sigmoid",
        cv=3,
    )
    svc.fit(X_tr_sp, y_tr, sample_weight=sw_tr)

    sgd = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )
    sgd.fit(X_tr_sp, y_tr)

    gbdt = make_gbdt()
    gbdt.fit(X_tr_dn, y_tr)

    models = {"lr": lr, "svc": svc, "sgd": sgd, "gbdt": gbdt}
    if fixed_w is not None and fixed_thr is not None:
        return {"models": models, "weights": fixed_w, "threshold": fixed_thr}

    p_val = np.column_stack([
        lr.predict_proba(X_va_sp)[:, 1],
        svc.predict_proba(X_va_sp)[:, 1],
        sgd.predict_proba(X_va_sp)[:, 1],
        gbdt.predict_proba(X_va_dn)[:, 1],
    ])
    sw_va = compute_sample_weight("balanced", y_va)

    def obj(w):
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-12)
        p = np.clip(p_val @ w, 1e-6, 1 - 1e-6)
        return log_loss(y_va, p, sample_weight=sw_va)

    res = minimize(
        obj,
        x0=np.array([0.40, 0.30, 0.15, 0.15]),
        bounds=[(0.0, 1.0)] * 4,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
    )
    best_w = res.x / res.x.sum()
    p_blend = p_val @ best_w

    best_thr, best_f1 = 0.50, -1.0
    for thr in np.linspace(0.30, 0.70, 81):
        score = f1_score(y_va, (p_blend >= thr).astype(int), average="macro")
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)

    return {"models": models, "weights": best_w, "threshold": best_thr}


def main():
    parser = argparse.ArgumentParser(description="Train portable MBTI ensemble artifact.")
    parser.add_argument("--data", default="mbti_1.csv.zip", help="Path to mbti_1.csv.zip or mbti_1.csv")
    parser.add_argument("--out", default="mbti_predictor.joblib", help="Output joblib path")
    parser.add_argument("--keep-mbti-words", action="store_true", help="Keep explicit MBTI type mentions")
    args = parser.parse_args()

    remove_mbti = not args.keep_mbti_words
    print(f"Loading dataset from {args.data} (remove_mbti_words={remove_mbti})...")
    df = load_dataset(args.data)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["type"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df["type"])

    print("Extracting training features...")
    X_tr_sp, X_tr_dn, fitted = build_features(train_df["posts"], remove_mbti_words=remove_mbti)
    X_va_sp, X_va_dn, _ = build_features(val_df["posts"], remove_mbti_words=remove_mbti, fitted=fitted)
    X_te_sp, X_te_dn, _ = build_features(test_df["posts"], remove_mbti_words=remove_mbti, fitted=fitted)

    tuned = {}
    for col, pos_l, neg_l, desc in DIMENSIONS:
        print(f"Tuning {desc}...")
        bundle = fit_axis(
            X_tr_sp, X_tr_dn, train_df[col].values,
            X_va_sp, X_va_dn, val_df[col].values,
        )
        tuned[col] = bundle

        m = bundle["models"]
        p_te = np.column_stack([
            m["lr"].predict_proba(X_te_sp)[:, 1],
            m["svc"].predict_proba(X_te_sp)[:, 1],
            m["sgd"].predict_proba(X_te_sp)[:, 1],
            m["gbdt"].predict_proba(X_te_dn)[:, 1],
        ]) @ bundle["weights"]
        y_te = test_df[col].values
        y_pred = (p_te >= bundle["threshold"]).astype(int)
        print(
            f"  Test {pos_l}/{neg_l} -> AUC: {roc_auc_score(y_te, p_te):.4f} | "
            f"Macro-F1: {f1_score(y_te, y_pred, average='macro'):.4f} | "
            f"Balanced Acc: {balanced_accuracy_score(y_te, y_pred):.4f}"
        )

    print("Refitting production bundle on 100% of dataset...")
    X_full_sp, X_full_dn, full_fitted = build_features(df["posts"], remove_mbti_words=remove_mbti)
    full_axes = {}
    for col, _, _, desc in DIMENSIONS:
        print(f"  Refitting {desc}...")
        full_axes[col] = fit_axis(
            X_full_sp, X_full_dn, df[col].values,
            fixed_w=tuned[col]["weights"],
            fixed_thr=tuned[col]["threshold"],
        )

    artifact = {
        "format_version": 1,
        "remove_mbti_words": remove_mbti,
        "dimensions": DIMENSIONS,
        **full_fitted,
        "axis_bundles": full_axes,
    }
    joblib.dump(artifact, args.out, compress=3)
    print(f"Saved portable model bundle to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
