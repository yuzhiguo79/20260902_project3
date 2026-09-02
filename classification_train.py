"""離線訓練分類模型；直接執行此檔會將模型匯出至 models。"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["order_count", "total_spend", "avg_order_value", "total_units", "session_count", "recency_days", "tenure_days"]
CATEGORICAL = ["acquisition_channel", "city"]

def _pipeline(name: str) -> Pipeline:
    estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42) if name == "LogisticRegression" else RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("preprocessor", prep), ("model", estimator)])

def train_classifiers(csv_path: str) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(csv_path); x, y = frame[NUMERIC + CATEGORICAL], frame["is_vip"]
    folds = min(5, int(y.value_counts().min()))
    if folds < 2:
        raise ValueError("每個分類至少需要 2 筆資料才能交叉驗證。")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    artifacts, rows = {}, []
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1"}
    for name in ("LogisticRegression", "RandomForest"):
        model = _pipeline(name); scores = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=-1)
        metrics = {"accuracy": scores["test_accuracy"].mean(), "roc_auc": scores["test_roc_auc"].mean(), "f1": scores["test_f1"].mean()}
        model.fit(x, y)
        artifacts[name] = {"task": "classification", "model_name": name, "model": model, "features": NUMERIC + CATEGORICAL, "metrics": metrics}
        rows.append({"模型": name, "CV Accuracy": metrics["accuracy"], "CV AUC": metrics["roc_auc"], "CV F1": metrics["f1"]})
    return artifacts, pd.DataFrame(rows)

def export_classifiers() -> pd.DataFrame:
    """建立最新客戶特徵、訓練兩個分類模型並匯出。"""
    from operations_data import build_processed_csvs

    root = Path(__file__).resolve().parent
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = build_processed_csvs()
    artifacts, comparison = train_classifiers(str(csv_paths["customers"]))
    filenames = {
        "LogisticRegression": "classification_logistic_regression.joblib",
        "RandomForest": "classification_random_forest.joblib",
    }
    for name, artifact in artifacts.items():
        joblib.dump(artifact, model_dir / filenames[name])
    comparison.to_csv(model_dir / "classification_metrics.csv", index=False)
    return comparison


if __name__ == "__main__":
    result = export_classifiers()
    print(f"Exported {len(result)} classification models to models/")
