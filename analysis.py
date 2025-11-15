import pandas as pd
import json
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import KMeans
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from prophet import Prophet  # pip install prophet
import warnings
warnings.filterwarnings("ignore")

REQUIRED_COLUMNS = ["Product", "Quantity", "Unit_Price", "Date", "Region", "Category"]

def validate_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    if df["Quantity"].isnull().any() or df["Unit_Price"].isnull().any():
        return False, "Quantity or Unit_Price contains missing values."
    return True, "Data is valid."


def analyze_basic(df):
    total_sales = (df["Quantity"] * df["Unit_Price"]).sum()
    top_product = df.groupby("Product")["Quantity"].sum().idxmax()
    return {"total_sales": total_sales, "top_product": top_product}

def sales_by_category(df):
    df["Revenue"] = df["Quantity"] * df["Unit_Price"]
    result = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    return {"categories": result.index.tolist(), "values": result.values.tolist()}

def sales_over_time(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Revenue"] = df["Quantity"] * df["Unit_Price"]
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()
    monthly.index = monthly.index.astype(str)
    return {"months": monthly.index.tolist(), "values": monthly.values.tolist()}

def profit_by_category(df):
    if 'Profit' in df.columns and 'Category' in df.columns:
        data = df.groupby('Category')['Profit'].sum().reset_index()
        return {
            "categories": data['Category'].tolist(),
            "values": data['Profit'].tolist()
        }
    return None

def top_customers(df):
    """
    Return ALL customers sorted by total Sales descending.
    Front-end will decide how many to show.
    """
    if 'Customer' not in df.columns or 'Sales' not in df.columns:
        return None

    grouped = (
        df.groupby('Customer')['Sales']
          .sum()
          .sort_values(ascending=False)
    )

    return {
        "customers": grouped.index.tolist(),
        "values": grouped.values.tolist()
    }

def profit_efficiency(df):
    """Profitability efficiency by category (bubble chart)."""
    if {'Category', 'Profit', 'Quantity', 'Unit_Price'}.issubset(df.columns):
        df["Sales"] = df["Quantity"] * df["Unit_Price"]
        df["Margin %"] = (df["Profit"] / df["Sales"]) * 100
        agg = df.groupby("Category")[["Sales", "Profit", "Margin %"]].mean().reset_index()

        return {
            "categories": agg["Category"].tolist(),
            "sales": agg["Sales"].tolist(),
            "profit": agg["Profit"].tolist(),
            "margin": agg["Margin %"].tolist()
        }
    return None

def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.empty:
        return None

    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")

    def convert_ndarrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_ndarrays(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_ndarrays(v) for v in obj]
        return obj

    return convert_ndarrays(fig.to_dict())

def forecast_sales(df, months_ahead=3):
    """Basic sales forecast using Exponential Smoothing."""
    if 'Date' not in df.columns or 'Quantity' not in df.columns or 'Unit_Price' not in df.columns:
        return None

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Revenue'] = df['Quantity'] * df['Unit_Price']

    # Aggregate by month
    monthly = df.groupby(df['Date'].dt.to_period('M'))['Revenue'].sum().to_timestamp()
    if len(monthly) < 2:
        return {"history": [], "values": [], "forecast": [], "forecast_values": []}

    model = ExponentialSmoothing(monthly, trend="add", seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(months_ahead)

    # 🔧 FIX: Create proper datetime index for forecast
    last_date = monthly.index[-1]
    future_index = pd.date_range(last_date, periods=months_ahead + 1, freq='M')[1:]
    forecast.index = future_index

    return {
        "history": monthly.index.strftime("%Y-%m").tolist(),
        "values": monthly.values.tolist(),
        "forecast": forecast.index.strftime("%Y-%m").tolist(),
        "forecast_values": forecast.values.tolist()
    }

def rfm_segmentation(df, k=4):
    if not {'Customer', 'Date', 'Sales'}.issubset(df.columns):
        return None

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Sales'])

    snapshot = df['Date'].max()
    rfm = df.groupby('Customer').agg({
        'Date': lambda x: (snapshot - x.max()).days,
        'Customer': 'count',
        'Sales': 'sum'
    }).rename(columns={'Date': 'Recency', 'Customer': 'Frequency', 'Sales': 'Monetary'})

    if len(rfm) < 2:
        # Not enough customers for clustering
        return rfm.reset_index().to_dict(orient='records')

    rfm = rfm.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = (rfm - rfm.min()) / (rfm.max() - rfm.min())
    scaled = scaled.fillna(0)

    k = min(k, len(scaled))  # don’t exceed available rows
    if k < 2:
        k = 1

    try:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        rfm['Cluster'] = model.fit_predict(scaled)
    except Exception:
        rfm['Cluster'] = 0

    return rfm.reset_index().to_dict(orient='records')

def sales_by_region(df):
    if {'Region', 'Quantity', 'Unit_Price'}.issubset(df.columns):
        df['Revenue'] = df['Quantity'] * df['Unit_Price']
        data = df.groupby('Region')['Revenue'].sum().reset_index()
        fig = px.choropleth(data, locations="Region", locationmode="country names",
                            color="Revenue", color_continuous_scale="Blues",
                            title="Sales by Region")
        return json.loads(fig.to_json())
    return None

def forecast_sales_prophet(df, months_ahead=6):
    """Advanced forecast using Facebook Prophet."""
    if not {'Date', 'Quantity', 'Unit_Price'}.issubset(df.columns):
        return None
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Revenue'] = df['Quantity'] * df['Unit_Price']
    monthly = df.groupby(df['Date'].dt.to_period('M'))['Revenue'].sum().to_timestamp().reset_index()
    monthly.columns = ['ds', 'y']

    if len(monthly) < 3:
        return None

    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(monthly)
    future = model.make_future_dataframe(periods=months_ahead, freq='M')
    forecast = model.predict(future)

    return {
        "history": monthly['ds'].dt.strftime("%Y-%m").tolist(),
        "values": monthly['y'].tolist(),
        "forecast": forecast['ds'].dt.strftime("%Y-%m").tolist(),
        "forecast_values": forecast['yhat'].tolist(),
    }


def churn_prediction(df):
    """Predicts customer churn using RandomForest (simulated binary target)."""
    if not {'Customer', 'Date', 'Sales'}.issubset(df.columns):
        return None

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    snapshot_date = df['Date'].max()
    customer_df = df.groupby('Customer').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'Sales': 'sum',
        'Customer': 'count'
    }).rename(columns={'Date': 'Recency', 'Customer': 'Frequency', 'Sales': 'Monetary'})

    # Simulate churn labels (for demo)
    customer_df['Churned'] = (customer_df['Recency'] > customer_df['Recency'].median()).astype(int)

    X = customer_df[['Recency', 'Frequency', 'Monetary']]
    y = customer_df['Churned']
    if len(X) < 4:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    feature_importance = dict(zip(X.columns, model.feature_importances_))

    return {
        "accuracy": round(report['accuracy'], 3),
        "feature_importance": feature_importance,
        "churn_rate": round(customer_df['Churned'].mean(), 3)
    }


def product_demand_forecast(df, months_ahead=3):
    """Forecast demand (Quantity) for each product."""
    # --- fix column typo ---
    df.columns = [c.strip().title() for c in df.columns]

    if not {'Product', 'Date', 'Quantity'}.issubset(df.columns):
        return None

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    result = {}

    for product, group in df.groupby('Product'):
        monthly = group.groupby(group['Date'].dt.to_period('M'))['Quantity'].sum().to_timestamp()

        if len(monthly) < 2:
            continue

        try:
            model = ExponentialSmoothing(monthly, trend="add", seasonal=None)
            fit = model.fit()
            forecast = fit.forecast(months_ahead)

            # 🔧 FIX: Create proper future monthly dates
            last_date = monthly.index[-1]
            future_index = pd.date_range(last_date, periods=months_ahead + 1, freq='M')[1:]
            forecast.index = future_index

            result[product] = {
                "history": monthly.index.strftime("%Y-%m").tolist(),
                "values": monthly.values.tolist(),
                "forecast": forecast.index.strftime("%Y-%m").tolist(),
                "forecast_values": forecast.values.tolist()
            }
        except Exception as e:
            print(f"Error forecasting {product}: {e}")

    return result if result else None