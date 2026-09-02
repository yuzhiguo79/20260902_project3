"""依模型名稱載入預先訓練的迴歸模型並預測。"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

def render_regression_prediction(csv_paths: dict[str, Path]) -> None:
    st.title("迴歸預測")
    st.caption("選擇預先訓練的模型，預測訂單金額。")
    model_files = {
        "RandomForest": Path(__file__).resolve().parent / "models" / "regression_random_forest.joblib",
        "Ridge": Path(__file__).resolve().parent / "models" / "regression_ridge.joblib",
        "Lasso": Path(__file__).resolve().parent / "models" / "regression_lasso.joblib",
    }
    model_name = st.selectbox("迴歸模型", list(model_files))
    model_path = model_files[model_name]
    if not model_path.exists():
        st.error(f"找不到模型檔 {model_path.name}，請先執行 regression_train.py。"); return
    try:
        artifact = joblib.load(model_path)
        if artifact.get("task") != "regression": raise ValueError("這不是迴歸模型檔。")
    except Exception as error:
        st.error(f"模型載入失敗：{error}"); return
    orders = pd.read_csv(csv_paths["orders"])
    with st.form("regression_prediction"):
        a, b, c = st.columns(3)
        values = {"units": a.number_input("商品總數", 1, 1000, int(orders["units"].median())), "avg_discount": b.slider("平均折扣率", 0.0, 1.0, float(orders["avg_discount"].median()), .01), "product_count": c.number_input("商品種類數", 1, 100, int(orders["product_count"].median())), "order_month": a.selectbox("訂單月份", list(range(1, 13))), "order_quarter": b.selectbox("訂單季度", [1, 2, 3, 4]), "segment": c.selectbox("客戶分群", sorted(orders["segment"].dropna().unique())), "acquisition_channel": a.selectbox("獲客管道", sorted(orders["acquisition_channel"].dropna().unique())), "city": b.selectbox("城市", sorted(orders["city"].dropna().unique())), "payment_type": c.selectbox("付款方式", sorted(orders["payment_type"].dropna().unique()))}
        submitted = st.form_submit_button("預測訂單金額", type="primary")
    if submitted:
        prediction = max(float(artifact["model"].predict(pd.DataFrame([values]))[0]), 0)
        st.metric(f"{artifact['model_name']} 預測訂單金額", f"NT$ {prediction:,.0f}")
        st.caption(f"模型測試 MAE：NT$ {artifact['metrics']['mae']:,.0f}")
