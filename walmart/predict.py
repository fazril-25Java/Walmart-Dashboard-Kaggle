import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def engineer_features(df):
    """Resample weekly data to monthly and extract time-series features."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df = df.sort_values(by=["store", "date"]).reset_index(drop=True)

    # 1. Resample Weekly to Monthly per Store
    # Group by store and month start
    df_monthly = (
        df.groupby(["store", pd.Grouper(key="date", freq="MS")])
        .agg({
            "weekly_sales": "sum",          # Sum weekly sales into total monthly sales
            "holiday_flag": "max",          # 1 if any week in the month was a holiday
            "temperature": "mean",          # Average monthly temperature
            "fuel_price": "mean",           # Average monthly fuel price
            "cpi": "mean",                  # Average monthly CPI
            "unemployment": "mean"          # Average monthly unemployment
        })
        .reset_index()
    )
    df_monthly.rename(columns={"weekly_sales": "monthly_sales"}, inplace=True)

    # 2. Extract Monthly Calendar Features
    df_monthly["year"] = df_monthly["date"].dt.year
    df_monthly["month"] = df_monthly["date"].dt.month

    # 3. Monthly Lag Features (1-month and 2-month prior sales)
    df_monthly["lag_1"] = df_monthly.groupby("store")["monthly_sales"].shift(1)
    df_monthly["lag_2"] = df_monthly.groupby("store")["monthly_sales"].shift(2)

    # 3-month rolling average
    df_monthly["rolling_3_avg"] = df_monthly.groupby("store")["monthly_sales"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    return df_monthly

def train_and_save_model():
    print("📥 Loading raw CSV dataset...")
    raw_df = pd.read_csv("walmart_sales.csv")

    print("🛠️ Resampling data to monthly & engineering features...")
    df = engineer_features(raw_df)
    
    # Drop initial NaN rows caused by lag shifts
    clean_df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "store", "holiday_flag", "temperature", "fuel_price", 
        "cpi", "unemployment", "year", "month",
        "lag_1", "lag_2", "rolling_3_avg"
    ]
    target_col = "monthly_sales"

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    # Chronological Split (80% Train, 20% Test)
    split_index = int(len(clean_df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("🤖 Training Random Forest Regressor on Monthly Sales...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n✅ Monthly Model Training Complete!")
    print(f"📊 Test MAE : ${mae:,.2f}")
    print(f"📊 Test R² Score : {r2:.4f}")

    # Save model package
    package = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": {"mae": mae, "r2": r2}
    }
    
    joblib.dump(package, "walmart_model.joblib")
    print("💾 Saved model package to 'walmart_model.joblib'")

if __name__ == "__main__":
    train_and_save_model()