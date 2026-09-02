"""銷售資料分析與預測的 Streamlit 主程式。"""
import streamlit as st
from classification_predict import render_classification_prediction
from data_review import render_data_review
from operations_data import build_processed_csvs
from operations_overview import render_operations_overview
from regression_predict import render_regression_prediction

st.set_page_config(page_title="銷售營運分析與預測", page_icon="📊", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.5rem;padding-bottom:2rem}[data-testid="stMetric"]{background:#f7f9fc;border:1px solid #e5eaf1;border-radius:12px;padding:16px 18px;min-height:110px}div[data-testid="stSidebar"]{background:#f8fafc}</style>""", unsafe_allow_html=True)
PAGES = ["1｜營運總覽", "2｜資料檢視", "3｜分類預測", "4｜迴歸預測"]
with st.sidebar:
    st.title("📊 銷售分析平台")
    page = st.radio("功能選單", PAGES)
try:
    csv_paths = build_processed_csvs()
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"資料彙整失敗：{error}")
    st.stop()
if page == PAGES[0]:
    render_operations_overview(csv_paths)
elif page == PAGES[1]:
    render_data_review(csv_paths)
elif page == PAGES[2]:
    render_classification_prediction(csv_paths)
else:
    render_regression_prediction(csv_paths)
st.divider()
st.caption("原始資料：data/raw｜彙整資料：data/processed｜模型：models")
