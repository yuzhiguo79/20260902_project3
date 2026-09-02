"""離線訓練迴歸模型；直接執行此檔會將模型匯出至 models。"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["units", "avg_discount", "product_count", "order_month", "order_quarter"]
CATEGORICAL = ["segment", "acquisition_channel", "city", "payment_type"]

def train_regressors(csv_path: str) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(csv_path); 
    x, y = frame[NUMERIC + CATEGORICAL], frame["order_amount"]
    xt, xv, yt, yv = train_test_split(x, y, test_size=.25, random_state=42)
    estimators = {
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=8,  random_state=42, n_jobs=-1), #min_samples_leaf=3,
        "Ridge": Ridge(alpha=10.0), 
        "Lasso": Lasso(alpha=10.0, max_iter=10000)
        }
    prep = lambda: ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    artifacts, rows = {}, []
    for name, estimator in estimators.items():
        model = Pipeline([("preprocessor", prep()), ("model", estimator)]); 
        model.fit(xt, yt); 
        prediction = np.maximum(model.predict(xv), 0)
        metrics = {"mae": mean_absolute_error(yv, prediction), 
                   "r2": r2_score(yv, prediction)}
        model.fit(x, y); 
        artifacts[name] = {"task": "regression", 
                           "model_name": name, 
                           "model": model, 
                           "features": NUMERIC + CATEGORICAL, 
                           "metrics": metrics}
        rows.append({"模型": name, "MAE": metrics["mae"], "R²": metrics["r2"]})
    return artifacts, pd.DataFrame(rows)

def export_regressors() -> pd.DataFrame:
    """建立最新訂單彙整、訓練三個迴歸模型並匯出。"""
    from operations_data import build_processed_csvs

    root = Path(__file__).resolve().parent
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = build_processed_csvs()
    artifacts, comparison = train_regressors(str(csv_paths["orders"]))
    filenames = {
        "RandomForest": "regression_random_forest.joblib",
        "Ridge": "regression_ridge.joblib",
        "Lasso": "regression_lasso.joblib",
    }
    for name, artifact in artifacts.items():
        joblib.dump(artifact, model_dir / filenames[name])
    comparison.to_csv(model_dir / "regression_metrics.csv", index=False)
    return comparison


if __name__ == "__main__":
    result = export_regressors()
    print(f"Exported {len(result)} regression models to models/")
