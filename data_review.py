"""資料檢視：讀取彙整 CSV 並以頁籤呈現。"""
from pathlib import Path
import pandas as pd
import streamlit as st

def render_data_review(csv_paths: dict[str, Path]) -> None:
    st.title("資料檢視"); st.caption("從彙整 CSV 檢視客戶、流量及每月訂單統計。")
    orders = pd.read_csv(csv_paths["orders"], parse_dates=["order_date"]); sessions = pd.read_csv(csv_paths["sessions"], parse_dates=["session_start"])
    customer_tab, traffic_tab, monthly_tab = st.tabs(["客戶與訂單", "流量與訂單", "每月訂單營收"])
    with customer_tab:
        stats = orders.groupby("segment", as_index=False).agg(訂單數=("order_id", "nunique"), 營收=("order_amount", "sum"), 平均訂單金額=("order_amount", "mean"), 客戶數=("customer_id", "nunique"))
        left, right = st.columns(2); left.bar_chart(stats, x="segment", y="訂單數"); right.bar_chart(stats, x="segment", y="營收"); st.dataframe(stats, hide_index=True, width="stretch")
    with traffic_tab:
        counts = sessions.groupby(["customer_id", "traffic_source"], as_index=False).agg(工作階段數=("session_id", "nunique")); order_counts = orders.groupby("customer_id", as_index=False).agg(訂單數=("order_id", "nunique"))
        traffic = counts.merge(order_counts, on="customer_id", how="left").fillna({"訂單數": 0}); correlation = traffic[["工作階段數", "訂單數"]].corr().iloc[0, 1]
        st.metric("工作階段數與訂單數相關係數", f"{correlation:.3f}"); st.scatter_chart(traffic, x="工作階段數", y="訂單數", color="traffic_source", size=60); st.dataframe(traffic.groupby("traffic_source", as_index=False)[["工作階段數", "訂單數"]].sum(), hide_index=True, width="stretch")
    with monthly_tab:
        monthly = orders.assign(月份=orders["order_date"].dt.to_period("M").dt.to_timestamp()).groupby("月份", as_index=False).agg(訂單數=("order_id", "nunique"), 營收=("order_amount", "sum"), 平均訂單金額=("order_amount", "mean"))
        st.line_chart(monthly, x="月份", y="營收"); st.dataframe(monthly, hide_index=True, width="stretch")
