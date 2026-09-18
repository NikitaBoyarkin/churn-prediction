"""Train + evaluate the churn model with a leakage-free time split.

Pipeline:
  1. Build datasets at three chronological snapshots (train → val → test).
  2. Fit LightGBM with categorical features passed natively.
  3. Evaluate: ROC-AUC, PR-AUC, recall@top-decile, lift@top-decile, Brier.
  4. Compare against a logistic-regression baseline.
  5. Native LightGBM SHAP values (pred_contrib=True) → feature importance.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from lightgbm import LGBMClassifier

from . import config as C
from .features import build_dataset

REPORTS = pathlib.Path(__file__).parent.parent / "reports"


def _prep(X: pd.DataFrame):
    X = X.copy()
    drop = ["user_id"]
    feats = X.drop(columns=[c for c in drop if c in X.columns])
    for c in C.CATEGORICAL:
        if c in feats.columns:
            feats[c] = feats[c].astype("category")
    return feats


def _metrics(y, p) -> dict:
    df = pd.DataFrame({"y": y.values, "p": p})
    n = len(df)
    decile_n = max(1, n // 10)
    top = df.sort_values("p", ascending=False).head(decile_n)
    base_rate = float(y.mean())
    top_recall = float(top["y"].sum()) / float(max(1, y.sum()))
    lift = top_recall / base_rate if base_rate else 0.0
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": round(base_rate, 4),
        "recall_at_top_decile": round(top_recall, 4),
        "lift_at_top_decile": round(lift, 2),
        "precision_at_top_decile": round(float(top["y"].mean()), 4),
    }


def run_pipeline(plot: bool = True) -> dict:
    REPORTS.mkdir(exist_ok=True)
    Xtr, ytr = build_dataset(C.SNAPSHOT_TRAIN)
    Xva, yva = build_dataset(C.SNAPSHOT_VAL)
    Xte, yte = build_dataset(C.SNAPSHOT_TEST)
    print(f"train={len(Xtr)} (churn {ytr.mean():.3f})  "
          f"val={len(Xva)} (churn {yva.mean():.3f})  "
          f"test={len(Xte)} (churn {yte.mean():.3f})")

    Ftr, Fva, Fte = _prep(Xtr), _prep(Xva), _prep(Xte)
    cat = C.CATEGORICAL

    model = LGBMClassifier(**C.MODEL_PARAMS)
    model.fit(
        Ftr, ytr,
        eval_set=[(Fva, yva)],
        categorical_feature=cat,
        callbacks=[],
    )

    # Baseline: scaled logistic regression on one-hot features.
    base = LogisticRegression(max_iter=400, class_weight="balanced", n_jobs=-1)
    Ftr_ohe = pd.get_dummies(Ftr, columns=cat, drop_first=True)
    Fte_ohe = pd.get_dummies(Fte, columns=cat, drop_first=True)
    Fte_ohe = Fte_ohe.reindex(columns=Ftr_ohe.columns, fill_value=0)
    base.fit(Ftr_ohe, ytr)
    p_base = base.predict_proba(Fte_ohe)[:, 1]

    p_te = model.predict_proba(Fte)[:, 1]
    p_va = model.predict_proba(Fva)[:, 1]

    results = {
        "split": {"train_snapshot": C.SNAPSHOT_TRAIN, "val_snapshot": C.SNAPSHOT_VAL,
                  "test_snapshot": C.SNAPSHOT_TEST, "label_window_days": C.LABEL_WINDOW_DAYS},
        "n": {"train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
        "lgbm_val": _metrics(yva, p_va),
        "lgbm_test": _metrics(yte, p_te),
        "logreg_test": _metrics(yte, p_base),
    }

    # Native SHAP (TreeSHAP via LightGBM, no `shap` package).
    shap_vals = model.predict_proba(Fte, pred_contrib=True)  # (n, n_feat+1)
    expl = pd.DataFrame(shap_vals[:, :-1], columns=Fte.columns)
    mean_abs = expl.abs().mean().sort_values(ascending=False)
    results["shap_top10"] = {k: round(float(v), 4) for k, v in mean_abs.head(10).items()}

    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    _print(results)

    if plot:
        try:
            _plots(yte, p_te, mean_abs)
        except Exception as e:  # plotting is optional (headless CI)
            print(f"[plot skipped: {e}]")
    return results


def _print(r: dict) -> None:
    print("\n=== Test metrics ===")
    print(f"{'metric':<24}{'lgbm':>10}{'logreg':>10}")
    for m in ("roc_auc", "pr_auc", "recall_at_top_decile", "lift_at_top_decile", "brier"):
        print(f"{m:<24}{r['lgbm_test'][m]:>10.3f}{r['logreg_test'][m]:>10.3f}")
    print("\n=== Top-10 SHAP (mean |contribution|) ===")
    for k, v in r["shap_top10"].items():
        print(f"  {k:<28}{v}")


def _plots(y, p, mean_abs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", alpha=0.4)
    ax[0].set_title(f"ROC (AUC={roc_auc_score(y, p):.3f})"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[1].plot(rec, prec); ax[1].set_title(f"PR (AP={average_precision_score(y, p):.3f})")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    top = mean_abs.head(12)[::-1]
    ax[2].barh(top.index, top.values); ax[2].set_title("SHAP mean |contribution|")
    fig.tight_layout()
    out = REPORTS / "evaluation.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[saved {out}]")


if __name__ == "__main__":
    run_pipeline()
