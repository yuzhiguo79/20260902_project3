"""1.營運總覽：讀取彙整 CSV，呈現統計與視覺化。"""
from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["order_date", "month"])

def render_operations_overview(csv_paths: dict[str, Path]) -> None:
    st.title("營運總覽"); 
    st.caption("彙整原始資料為 CSV，再由 CSV 顯示營運指標與圖表。")
    facts = _read(str(csv_paths['facts']))
    with st.sidebar:
        st.subheader("總覽選擇：")
        segments = st.multiselect("客戶分群", sorted(facts['segment'].dropna().unique()),
                                  default=sorted(facts['segment'].dropna().unique()))
        channels = st.multiselect("獲客管道", sorted(facts['acquisition_channel'].dropna().unique()),
                                  default=sorted(facts['acquisition_channel'].dropna().unique()))

    shown = facts[ facts['segment'].isin(segments) &  facts['acquisition_channel'].isin(channels)  ]
    if shown.empty:
        st.warning("目前無資料")
        return
    money = lambda value: f"NT${value:,.0f}"

    a,b,c = st.columns(3)
    a.metric("營收", money(shown['line_revenue'].sum()))
    b.metric("訂單數", f'{shown['order_id'].nunique():,}')
    c.metric("均單額",money(shown.groupby('order_id')['line_revenue'].sum().mean()))

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("每月營收")
        st.line_chart(shown.groupby('month',as_index=False)['line_revenue'].sum(), 
                        x='month', y='line_revenue')
    with right:
        st.subheader("客群營收")
        st.bar_chart(shown.groupby('segment', as_index=False)['line_revenue'].sum(), 
                        x="segment", y='line_revenue')

    left, right = st.columns(2)
    with left:
        st.subheader("獲客管道")
        st.bar_chart(shown.groupby('acquisition_channel', as_index=False)['line_revenue'].sum(), 
                                    x="acquisition_channel", y='line_revenue')
    with right:
        st.subheader("產品類別營收")
        st.bar_chart(shown.groupby('category', as_index=False)['line_revenue'].sum(), 
                        x="category", y='line_revenue')
    
