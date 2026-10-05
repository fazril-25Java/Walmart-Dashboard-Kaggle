import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# 0. Set dynamic base directory relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from predict import engineer_features, train_and_save_model

# 1. Page Configuration
st.set_page_config(
    page_title="Walmart Monthly Sales Dashboard (Stores 1-3)",
    page_icon="🛒",
    layout="wide"
)

# 2. Load & Clean Data (Aggregated Monthly)
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "walmart_sales.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")
        
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    
    # Filter strictly Stores 1, 2, and 3
    df = df[df["store"].isin([1, 2, 3])].copy()
    
    # Resample weekly data to monthly
    df_monthly = (
        df.groupby(["store", pd.Grouper(key="date", freq="MS")])
        .agg({
            "weekly_sales": "sum",
            "holiday_flag": "max",
            "temperature": "mean",
            "fuel_price": "mean",
            "cpi": "mean",
            "unemployment": "mean"
        })
        .reset_index()
    )
    df_monthly.rename(columns={"weekly_sales": "monthly_sales"}, inplace=True)
    df_monthly["holiday_label"] = df_monthly["holiday_flag"].map({1: "Holiday Month", 0: "Regular Month"})
    return df_monthly

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"`walmart_sales.csv` not found in `{BASE_DIR}`. Please check file placement.")
    st.stop()

# 3. Sidebar Filters
st.sidebar.header("Dashboard Controls 🎛️")

selected_store = st.sidebar.radio(
    "Select Store View",
    options=["All Stores (1, 2 & 3)", "Store 1", "Store 2", "Store 3"],
    index=0
)

factor_map = {
    "Temperature (°F)": "temperature",
    "Unemployment Rate (%)": "unemployment",
    "Fuel Price ($)": "fuel_price",
    "CPI": "cpi"
}
selected_factor_label = st.sidebar.selectbox("Economic Indicator", options=list(factor_map.keys()))
selected_factor = factor_map[selected_factor_label]

if "All Stores" in selected_store:
    filtered_df = df.copy()
else:
    store_id = int(selected_store.split()[-1])
    filtered_df = df[df["store"] == store_id]

# 4. Header Section
st.title("🛒 Walmart Monthly Sales Dashboard (Stores 1, 2 & 3)")
st.markdown("Monthly performance analysis and ML predictions for Stores 1, 2, and 3.")
st.divider()

tab_analytics, tab_ml = st.tabs(["📊 Analytics Dashboard", "🤖 Machine Learning Predictions"])

with tab_analytics:
    # 5. Executive KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    total_sales = filtered_df["monthly_sales"].sum()
    avg_sales = filtered_df["monthly_sales"].mean()
    max_sales = filtered_df["monthly_sales"].max()

    reg_sales_mean = filtered_df[filtered_df["holiday_flag"] == 0]["monthly_sales"].mean()
    hol_sales_mean = filtered_df[filtered_df["holiday_flag"] == 1]["monthly_sales"].mean()
    holiday_lift = ((hol_sales_mean - reg_sales_mean) / reg_sales_mean) * 100 if reg_sales_mean else 0

    col1.metric("Total Revenue", f"${total_sales:,.2f}")
    col2.metric("Avg Monthly Sales", f"${avg_sales:,.2f}")
    col3.metric("Peak Monthly Revenue", f"${max_sales:,.2f}")
    col4.metric("Holiday Lift", f"+{holiday_lift:.1f}%")

    st.divider()

    # 6. Charts Grid
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📈 Monthly Sales Trends")
        fig_trend = px.line(
            filtered_df,
            x="date",
            y="monthly_sales",
            color="store" if "All Stores" in selected_store else None,
            labels={"monthly_sales": "Monthly Sales ($)", "date": "Date", "store": "Store"},
            template="plotly_white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with chart_col2:
        st.subheader("🏬 Store Revenue Benchmark")
        store_totals = df.groupby("store")["monthly_sales"].sum().reset_index()
        store_totals["store_name"] = store_totals["store"].apply(lambda x: f"Store {x}")
        
        fig_bar = px.bar(
            store_totals,
            x="store_name",
            y="monthly_sales",
            color="store_name",
            text_auto=".3s",
            labels={"monthly_sales": "Total Revenue ($)", "store_name": "Store"},
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader(f"📊 Impact of {selected_factor_label}")
        fig_scatter = px.scatter(
            filtered_df,
            x=selected_factor,
            y="monthly_sales",
            color="holiday_label",
            labels={"monthly_sales": "Monthly Sales ($)", selected_factor: selected_factor_label},
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with chart_col4:
        st.subheader("🎄 Holiday vs Regular Month Sales")
        fig_box = px.box(
            filtered_df,
            x="holiday_label",
            y="monthly_sales",
            color="holiday_label",
            labels={"monthly_sales": "Monthly Sales ($)", "holiday_label": "Month Type"},
            template="plotly_white"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("🔍 View Raw Monthly Data (Stores 1, 2 & 3)"):
        st.dataframe(filtered_df.sort_values(by="date", ascending=False), use_container_width=True)

with tab_ml:
    @st.cache_resource
    def load_trained_model():
        model_path = os.path.join(BASE_DIR, "walmart_model.joblib")
        if not os.path.exists(model_path):
            st.info("⚙️ Model package not found. Training model now...")
            train_and_save_model()
            
        return joblib.load(model_path)

    model_pkg = load_trained_model()
    model = model_pkg["model"]
    feature_cols = model_pkg["feature_cols"]

    @st.cache_data
    def get_prediction_data():
        csv_path = os.path.join(BASE_DIR, "walmart_sales.csv")
        raw_df = pd.read_csv(csv_path)
        df_engineered = engineer_features(raw_df).dropna().reset_index(drop=True)
        df_engineered["predicted_sales"] = model.predict(df_engineered[feature_cols])
        return df_engineered

    pred_df = get_prediction_data()

    st.subheader("🤖 Actual vs. Predicted Monthly Sales")

    pred_store_select = st.selectbox("Filter Prediction View", ["Store 1", "Store 2", "Store 3"], key="pred_store_select")
    pred_store_id = int(pred_store_select.split()[-1])
    store_pred_df = pred_df[pred_df["store"] == pred_store_id]

    # Grouped Bar Chart for Monthly Sales
    fig_pred = go.Figure()

    fig_pred.add_trace(
        go.Bar(
            x=store_pred_df["date"].dt.strftime("%Y-%m"),  # Format date as YYYY-MM
            y=store_pred_df["monthly_sales"],
            name="Actual Sales",
            marker_color="#1f77b4"
        )
    )

    fig_pred.add_trace(
        go.Bar(
            x=store_pred_df["date"].dt.strftime("%Y-%m"),
            y=store_pred_df["predicted_sales"],
            name="Predicted Sales",
            marker_color="#ff7f0e"
        )
    )

    fig_pred.update_layout(
        barmode="group",
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Monthly Sales ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_pred, use_container_width=True)