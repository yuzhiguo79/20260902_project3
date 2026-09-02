"""依模型名稱載入預先訓練的分類模型並預測。"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

def render_classification_prediction(csv_paths: dict[str, Path]) -> None:
    st.title("分類預測")
    st.caption("選擇預先訓練的模型，預測客戶成為 VIP 的機率。")
    model_files = {
        "LogisticRegression": Path(__file__).resolve().parent / "models" / "classification_logistic_regression.joblib",
        "RandomForest": Path(__file__).resolve().parent / "models" / "classification_random_forest.joblib",
    }
    model_name = st.selectbox("分類模型", list(model_files))
    model_path = model_files[model_name]
    if not model_path.exists():
        st.error(f"找不到模型檔 {model_path.name}，請先執行 classification_train.py。")
        return
    try:
        artifact = joblib.load(model_path)
        if artifact.get("task") != "classification": raise ValueError("這不是分類模型檔。")
    except Exception as error:
        st.error(f"模型載入失敗：{error}"); return
    frame = pd.read_csv(csv_paths["customers"]); defaults = frame.median(numeric_only=True)
    with st.form("classification_prediction"):
        a, b, c = st.columns(3)
        order_count = a.number_input("歷史訂單數", 0, 10000, int(defaults["order_count"])); total_spend = b.number_input("累計消費金額", 0.0, value=float(defaults["total_spend"]), step=1000.0)
        values = {"order_count": order_count, "total_spend": total_spend, "avg_order_value": total_spend / order_count if order_count else 0.0, "total_units": c.number_input("累計商品數", 0, 100000, int(defaults["total_units"])), "session_count": a.number_input("工作階段數", 0, 100000, int(defaults["session_count"])), "recency_days": b.number_input("距上次訂單天數", 0, 10000, int(defaults["recency_days"])), "tenure_days": c.number_input("客戶年資天數", 0, 10000, int(defaults["tenure_days"])), "acquisition_channel": a.selectbox("獲客管道", sorted(frame["acquisition_channel"].dropna().unique())), "city": b.selectbox("城市", sorted(frame["city"].dropna().unique()))}
        submitted = st.form_submit_button("預測 VIP 機率", type="primary")
    if submitted:
        probability = float(artifact["model"].predict_proba(pd.DataFrame([values]))[0, 1])
        st.progress(probability, text=f"{artifact['model_name']} 預測 VIP 機率：{probability:.1%}")
        st.success("預測為 VIP") if probability >= .5 else st.info("預測為非 VIP")
