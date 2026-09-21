"""Train, tune, compare and save the churn model.

Flow of my project: load-> clean -> split (80/20) -> feature engineering + preprocessing (Pipeline)
                        -> tune 5 models with 5-fold CV (training set only) -> class imbalance
                        -> threshold tuning on out-of-fold predictions -> pick best model by CV
                        -> evaluate ONCE on the untouched test set -> save model + outputs.
"""
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import add_features, basic_features, load_data

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_JOBS = 1          
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = load_data()                            

feature_cols = [c for c in df.columns if c != "CustomerID"]
df = df.drop_duplicates(subset=feature_cols).dropna()

customer_ids = df["CustomerID"]

y = df["churn"].astype(str).str.strip().str.lower().map(
    {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}).astype(int)
X = df.drop(columns=["CustomerID", "churn"])
assert "CustomerID" not in X.columns and "churn" not in X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Remember which customers are in the unseen test set (X keeps the original row index)
test_customers = pd.DataFrame({"CustomerID": customer_ids.loc[X_test.index].values,
                               "actual_churn": y_test.values}).sort_values("CustomerID")
test_customers.to_csv("outputs/test_customer_ids.csv", index=False)


def build_pipeline(model, feature_func):
    features = feature_func(X_train)
    binary_cols = [c for c in features.columns if set(features[c].unique()) <= {0, 1}]
    numeric_cols = [c for c in features.columns if c not in binary_cols]
    prep = ColumnTransformer([("scale", StandardScaler(), numeric_cols)], remainder="passthrough")
    return Pipeline([("features", FunctionTransformer(feature_func)),
                     ("prep", prep),
                     ("model", model)])

fe_rows = []
for name, clf in [("Logistic Regression", LogisticRegression(max_iter=2000)),
                  ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))]:
    for label, func in [("Without feature engineering", basic_features),
                        ("With feature engineering", add_features)]:
        scores = cross_val_score(build_pipeline(clf, func), X_train, y_train,
                                 cv=cv, scoring="f1", n_jobs=N_JOBS)
        fe_rows.append({"Model": name, "Features": label, "CV F1": round(scores.mean(), 4)})
pd.DataFrame(fe_rows).to_csv("outputs/feature_engineering_comparison.csv", index=False)

models = [
    ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
     {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"],
      "class_weight": [None, "balanced"]}, "grid"),
    ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE),
     {"max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 10, 20],
      "min_samples_leaf": [1, 5, 10], "class_weight": [None, "balanced"]}, "grid"),
    ("Random Forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
     {"n_estimators": [200, 300, 500], "max_depth": [None, 10, 15, 20],
      "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
      "class_weight": [None, "balanced", "balanced_subsample"]}, "random"),
    ("KNN", KNeighborsClassifier(),
     {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"], "p": [1, 2]}, "grid"),
    ("Extra Trees", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1),
     {"n_estimators": [200, 300, 500], "max_depth": [None, 10, 20],
      "min_samples_leaf": [1, 2, 4], "class_weight": [None, "balanced"]}, "random"),
]


def best_threshold(y_true, proba):
    """Threshold with the best F1 on the given (validation) predictions."""
    scores = [f1_score(y_true, proba >= t) for t in THRESHOLDS]
    return THRESHOLDS[int(np.argmax(scores))]


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {"Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(y_true, pred),
            "Recall": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred),
            "ROC-AUC": roc_auc_score(y_true, proba)}

results, best_params, fitted, oof_probas = [], {}, {}, {}
for name, estimator, grid, search_type in models:
    pipe = build_pipeline(estimator, add_features)
    param_grid = {f"model__{k}": v for k, v in grid.items()}
    if search_type == "grid":
        search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", n_jobs=N_JOBS)
    else:
        search = RandomizedSearchCV(pipe, param_grid, n_iter=15, cv=cv, scoring="f1",
                                    n_jobs=N_JOBS, random_state=RANDOM_STATE)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    proba = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=N_JOBS)[:, 1]
    threshold = best_threshold(y_train, proba)

    results.append({"Model": name, "CV F1": search.best_score_, "Threshold": threshold,
                    **metrics_at(y_train, proba, threshold)})
    best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    fitted[name], oof_probas[name] = best, proba

comparison = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
comparison.round(4).to_csv("outputs/model_comparison.csv", index=False)
with open("outputs/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2, default=str)

best_name = comparison.loc[0, "Model"]
best_model = fitted[best_name]
best_thr = float(comparison.loc[0, "Threshold"])

test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_thr).astype(int)
test_metrics = metrics_at(y_test, test_proba, best_thr)

with open("outputs/test_metrics.json", "w") as f:
    json.dump({"best_model": best_name, "threshold": best_thr,
               "train_rows": len(X_train), "test_rows": len(X_test),
               "churn_rate": round(float(y.mean()), 4),
               **{k: round(v, 4) for k, v in test_metrics.items()}}, f, indent=2)

joblib.dump({"pipeline": best_model, "threshold": best_thr, "model_name": best_name},
            "models/churn_model.pkl")

ConfusionMatrixDisplay.from_predictions(y_test, test_pred, display_labels=["No Churn", "Churn"],
                                        cmap="Blues")
plt.title(f"Confusion Matrix - {best_name} (test set)")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()

RocCurveDisplay.from_predictions(y_test, test_proba, name=best_name)
plt.plot([0, 1], [0, 1], "k--", label="Random guess"); plt.legend()
plt.title("ROC Curve (test set)")
plt.tight_layout(); plt.savefig("outputs/roc_curve.png", dpi=150); plt.close()

oof = oof_probas[best_name]
grid_t = np.arange(0.30, 0.701, 0.05)
plt.figure(figsize=(7, 4.5))
for label, fn in [("F1", f1_score), ("Precision", precision_score), ("Recall", recall_score)]:
    plt.plot(grid_t, [fn(y_train, oof >= t) for t in grid_t], marker="o", label=label)
plt.axvline(best_thr, color="gray", linestyle="--", label=f"Chosen = {best_thr:.2f}")
plt.xlabel("Threshold"); plt.ylabel("Score (5-fold validation)")
plt.title(f"Threshold tuning - {best_name}"); plt.legend()
plt.tight_layout(); plt.savefig("outputs/threshold_tuning.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, ["Accuracy", "F1", "ROC-AUC"]):
    ordered = comparison.sort_values(metric)
    ax.barh(ordered["Model"], ordered[metric], color="#3b6fb6")
    ax.set_title(metric); ax.set_xlim(max(0, ordered[metric].min() - 0.05), 1)
    for i, v in enumerate(ordered[metric]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
fig.suptitle("Model comparison (5-fold cross-validation, training set)")
plt.tight_layout(); plt.savefig("outputs/model_comparison.png", dpi=150); plt.close()

for metric, fname in [("F1", "f1_comparison.png"), ("Accuracy", "accuracy_comparison.png")]:
    ordered = comparison.sort_values(metric)
    plt.figure(figsize=(7, 4.5))
    plt.barh(ordered["Model"], ordered[metric], color="#3b6fb6")
    plt.xlim(max(0, ordered[metric].min() - 0.05), 1)
    for i, v in enumerate(ordered[metric]):
        plt.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    plt.title(f"{metric} comparison (cross-validation)")
    plt.tight_layout(); plt.savefig(f"outputs/{fname}", dpi=150); plt.close()

print(f"Best Model : {best_name} (threshold {best_thr:.2f})")
for k, v in test_metrics.items():
    print(f"{k:<10} : {v:.4f}")
