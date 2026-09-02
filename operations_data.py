"""營運資料彙整：讀取原始 CSV，建立分析用 CSV。"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR, PROCESSED_DIR = ROOT / "data" / "raw", ROOT / "data" / "processed"

def build_processed_csvs(force: bool = False) -> dict[str, Path]:
    """產生交易明細、訂單、客戶特徵及工作階段 CSV，並回傳路徑。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {"facts": PROCESSED_DIR / "sales_facts.csv", "orders": PROCESSED_DIR / "order_summary.csv", "customers": PROCESSED_DIR / "customer_features.csv", "sessions": PROCESSED_DIR / "sessions.csv"}
    sources = [RAW_DIR / name for name in ("customers.csv", "orders.csv", "order_items.csv", "products.csv", "sessions.csv")]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"找不到原始資料：{', '.join(missing)}")
    fresh = all(path.exists() for path in paths.values()) and max(p.stat().st_mtime for p in sources) <= min(p.stat().st_mtime for p in paths.values())
    if fresh and not force:
        return paths
    customers = pd.read_csv(sources[0], parse_dates=["signup_date"])
    orders = pd.read_csv(sources[1], parse_dates=["order_date"])
    items = pd.read_csv(sources[2])
    products = pd.read_csv(sources[3]).rename(columns={"unit_price": "list_price"})
    sessions = pd.read_csv(sources[4], parse_dates=["session_start"])
    items["line_revenue"] = items["quantity"] * items["unit_price"] * (1 - items["discount_rate"])
    facts = items.merge(products, on="product_id", how="left", validate="many_to_one").merge(orders, on="order_id", how="left", validate="many_to_one").merge(customers, on="customer_id", how="left", validate="many_to_one")
    facts = facts.loc[facts["status"].eq("completed")].copy()
    facts["month"] = facts["order_date"].dt.to_period("M").dt.to_timestamp()
    order_summary = facts.groupby("order_id", as_index=False).agg(customer_id=("customer_id", "first"), order_date=("order_date", "first"), segment=("segment", "first"), acquisition_channel=("acquisition_channel", "first"), city=("city", "first"), payment_type=("payment_type", "first"), order_amount=("line_revenue", "sum"), units=("quantity", "sum"), avg_discount=("discount_rate", "mean"), product_count=("product_id", "nunique"))
    order_summary["order_month"] = order_summary["order_date"].dt.month
    order_summary["order_quarter"] = order_summary["order_date"].dt.quarter
    cutoff = max(order_summary["order_date"].max(), sessions["session_start"].max())
    spend = order_summary.groupby("customer_id", as_index=False).agg(order_count=("order_id", "nunique"), total_spend=("order_amount", "sum"), avg_order_value=("order_amount", "mean"), total_units=("units", "sum"), last_order=("order_date", "max"))
    activity = sessions.groupby("customer_id", as_index=False).agg(session_count=("session_id", "nunique"), last_session=("session_start", "max"))
    features = customers.merge(spend, on="customer_id", how="left").merge(activity, on="customer_id", how="left")
    numeric = ["order_count", "total_spend", "avg_order_value", "total_units", "session_count"]
    features[numeric] = features[numeric].fillna(0)
    features["recency_days"] = (cutoff - features["last_order"]).dt.days.fillna(999)
    features["tenure_days"] = (cutoff - features["signup_date"]).dt.days.clip(lower=0)
    features["is_vip"] = features["segment"].eq("vip").astype(int)
    facts.to_csv(paths["facts"], index=False); order_summary.to_csv(paths["orders"], index=False)
    features.to_csv(paths["customers"], index=False); sessions.to_csv(paths["sessions"], index=False)
    return paths
