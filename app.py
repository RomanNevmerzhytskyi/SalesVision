# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin, login_required, current_user
from flask_security.utils import hash_password
from flask_login import current_user
from extensions import db
from datetime import timedelta
from models import User, Role, Upload, UserPreference
import os, uuid
import pandas as pd
from analysis import validate_data, analyze_basic, sales_by_category, sales_over_time, profit_by_category, top_customers
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import KMeans
from flask_mailman import Mail
from analysis import (
    validate_data,
    analyze_basic,
    sales_by_category,
    sales_over_time,
    profit_by_category,
    top_customers,
    profit_efficiency,
    correlation_heatmap,
    forecast_sales,
    rfm_segmentation,
    sales_by_region,
    forecast_sales_prophet,
    churn_prediction,
    product_demand_forecast,
)

import logging
logging.basicConfig(level=logging.INFO)


app = Flask(__name__)

mail = Mail()

app.config.update(
    SECRET_KEY="super-secret-key",
    SECURITY_PASSWORD_SALT="super-secret-salt",
    SQLALCHEMY_DATABASE_URI="sqlite:///salesvision.db",
    UPLOAD_FOLDER="data",

    # Flask-Security
    SECURITY_REGISTERABLE=True,
    SECURITY_SEND_REGISTER_EMAIL=False,
    SECURITY_CONFIRMABLE=False,
    SECURITY_PASSWORD_CONFIRMABLE=True,
    SECURITY_POST_REGISTER_VIEW='upload',
    SECURITY_POST_LOGIN_VIEW='upload',
    SECURITY_PASSWORD_HASH='argon2',
    SECURITY_REGISTER_USER_TEMPLATE='/security/register_user.html',
    SECURITY_CHANGEABLE=True,
    SECURITY_RECOVERABLE=True,
    SECURITY_REMEMBER_ME=True,
    SECURITY_LOGIN_WITHOUT_CONFIRMATION=True,
    SECURITY_SEND_PASSWORD_CHANGE_EMAIL=False,
    SECURITY_EMAIL_SENDER=("SalesVision", "noreply@example.test"),

    # Mailman (file backend)
    MAIL_BACKEND="file",
    MAIL_FILE_PATH="mail_outbox",
    MAIL_DEFAULT_SENDER=("SalesVision", "noreply@example.test"),
    MAIL_SUPPRESS_SEND=False,  # <-- only once
)

app.config["REMEMBER_COOKIE_PATH"] = "/"

db.init_app(app)
mail.init_app(app)

# Setup Flask-Security
user_datastore = SQLAlchemyUserDatastore(db, User, Role)
# security = Security(app, user_datastore)
from forms import ExtendedLoginForm
security = Security(app, user_datastore, login_form=ExtendedLoginForm)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files['file']
        description = (request.form.get('description') or '').strip()
        if file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            if not description:
                try:
                    df = pd.read_excel(filepath) if file.filename.endswith('.xlsx') else pd.read_csv(filepath)
                    description=auto_describe_df(df, fallback=file.filename)
                except Exception:
                    description=file.filename

            upload_entry = Upload(filename=file.filename, user_id=current_user.id, description=description)
            db.session.add(upload_entry)
            db.session.commit()
            return redirect(url_for('dashboard', filename=file.filename))
    return render_template('upload.html')

def auto_describe_df(df, fallback=""):
    # Normalize
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    rows = len(df)
    cols = list(df.columns)
    parts = []

    # Date span
    if 'Date' in df.columns and df['Date'].notna().any():
        dmin = df['Date'].min().date()
        dmax = df['Date'].max().date()
        parts.append(f"{dmin}–{dmax}")

    # Revenue / Sales
    if {'Quantity','Unit_Price'}.issubset(df.columns):
        revenue = float((df['Quantity'] * df['Unit_Price']).sum())
        parts.append(f"€{revenue:,.0f} revenue")

    # Top product
    if 'Product' in df.columns and 'Quantity' in df.columns and df['Quantity'].notna().any():
        try:
            top_product = df.groupby('Product')['Quantity'].sum().idxmax()
            parts.append(f"top: {top_product}")
        except Exception:
            pass

    # Category diversity
    if 'Category' in df.columns:
        parts.append(f"{df['Category'].nunique()} categories")

    # Regions
    if 'Region' in df.columns:
        parts.append(f"{df['Region'].nunique()} regions")

    # Compose
    head = f"{rows} rows" + (f" • {', '.join(cols[:4])}…" if cols else "")
    tail = " • ".join(parts) if parts else (fallback or "Uploaded dataset")
    text = f"{head} • {tail}"

    # Keep it short for tables
    return (text[:180] + "…") if len(text) > 180 else text

@app.route('/files/<int:file_id>/description', methods=['POST'])
@login_required
def update_file_description(file_id):
    upload = Upload.query.filter_by(id=file_id, user_id=current_user.id).first_or_404()
    new_desc = (request.form.get('description') or '').strip()
    upload.description = new_desc or upload.description
    db.session.commit()
    flash('Description updated.', 'info')
    return redirect(url_for('files'))

@app.route('/dashboard/<filename>')
@login_required
def dashboard(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_excel(filepath) if filename.endswith('.xlsx') else pd.read_csv(filepath)

    valid, message = validate_data(df)
    if not valid:
        return render_template('dashboard.html', filename=filename, error=message)

    # --- Core analytics ---
    results = analyze_basic(df)
    category_data = sales_by_category(df)
    trend_data = sales_over_time(df)
    profit_data = profit_by_category(df) if 'Profit' in df.columns else None
    profit_efficiency_data = profit_efficiency(df) if 'Profit' in df.columns else None
    top_customers_data = top_customers(df) if 'Customer' in df.columns else None
    corr_data = correlation_heatmap(df)
    forecast_data = forecast_sales(df)
    rfm_data = rfm_segmentation(df)
    region_data = sales_by_region(df)

    # --- NEW advanced analytics ---
    prophet_forecast_data = forecast_sales_prophet(df)
    churn_data = churn_prediction(df)
    demand_data = product_demand_forecast(df)

    # --- Generate general summary text ---
    summary_parts = []
    summary_parts.append(f"📊 The dataset contains {len(df):,} rows and {len(df.columns)} columns.")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].notna().any():
        dmin, dmax = df['Date'].min().date(), df['Date'].max().date()
        summary_parts.append(f"🗓️ Date range: {dmin} – {dmax}.")
    if 'Region' in df.columns:
        summary_parts.append(f"🌍 Covered regions: {df['Region'].nunique()}.")
    if 'Category' in df.columns:
        summary_parts.append(f"🛒 Product categories: {df['Category'].nunique()}.")
    if 'Customer' in df.columns:
        summary_parts.append(f"👥 Unique customers: {df['Customer'].nunique()}.")
    if 'Sales' in df.columns:
        total_sales = df['Sales'].sum()
        summary_parts.append(f"💰 Total sales: €{total_sales:,.2f}.")
    summary_text = " ".join(summary_parts) or "No summary available."


    # --- DEBUG LOGS ---
    logging.info(f"Category data: {type(category_data)}")
    logging.info(f"Trend data: {type(trend_data)}")
    logging.info(f"Profit data: {bool(profit_data)}")
    logging.info(f"Top customers: {bool(top_customers_data)}")
    logging.info(f"Correlation: {bool(corr_data)}")
    logging.info(f"Forecast: {bool(forecast_data)}")
    logging.info(f"RFM: {bool(rfm_data)}")
    logging.info(f"Region: {bool(region_data)}")
    logging.info(f"Prophet forecast: {bool(prophet_forecast_data)}")
    logging.info(f"Churn analysis: {bool(churn_data)}")
    logging.info(f"Demand forecast: {bool(demand_data)}")

    # --- Render the dashboard ---
    return render_template(
        'dashboard.html',
        filename=filename,
        total_sales=results['total_sales'],
        top_product=results['top_product'],
        category_data=category_data,
        trend_data=trend_data,
        profit_data=profit_data,
        profit_efficiency_data=profit_efficiency_data,
        top_customers_data=top_customers_data,
        corr_data=corr_data,
        forecast_data=forecast_data,
        rfm_data=rfm_data,
        region_data=region_data,
        prophet_forecast_data=prophet_forecast_data,
        churn_data=churn_data,
        demand_data=demand_data,
        summary_text=summary_text
    )


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    # Ensure the user has a preference row
    pref = current_user.preference
    if not pref:
        pref = UserPreference(user_id=current_user.id)
        db.session.add(pref)
        db.session.commit()

    if request.method == 'POST':
        # Basic sanitization/constraints
        theme = request.form.get('theme', 'light')
        default_tab = request.form.get('default_dashboard_tab', 'overview')
        try:
            charts_per_row = int(request.form.get('charts_per_row', 2))
        except ValueError:
            charts_per_row = 2
        charts_per_row = max(1, min(charts_per_row, 4))  # clamp 1..4

        homepage_bg = request.form.get('homepage_bg', '#f1f3f6').strip()
        if not homepage_bg.startswith('#') or len(homepage_bg) not in (4, 7, 9):
            homepage_bg = '#f1f3f6'

        # Save
        pref.theme = theme
        pref.default_dashboard_tab = default_tab
        pref.charts_per_row = charts_per_row
        pref.homepage_bg = homepage_bg
        db.session.commit()
        flash('Settings saved.', 'info')
        return redirect(url_for('settings'))

    return render_template('settings.html', pref=pref)

    prophet_forecast_data = forecast_sales_prophet(df)
    churn_data = churn_prediction(df)
    demand_data = product_demand_forecast(df)
    
    logging.info(f"Demand forecast result: {None if demand_data is None else list(demand_data.keys())}")

@app.route('/files')
@login_required
def files():
    uploads = Upload.query.filter_by(user_id=current_user.id).order_by(Upload.uploaded_at.desc()).all()
    return render_template('files.html', uploads=uploads)

@app.route('/files/delete/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    upload = Upload.query.filter_by(id=file_id, user_id=current_user.id).first()
    if not upload:
        flash("File not found or unauthorized.", "danger")
        return redirect(url_for('files'))
    
    # Remove the physical file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    db.session.delete(upload)
    db.session.commit()
    flash("File deleted successfully.", "info")
    return redirect(url_for('files'))

@app.route('/forgot')
def forgot_router():
    if current_user.is_authenticated:
        flash("You're already logged in. Use Change Password instead", "info")
        return redirect(url_for('security.change_password'))
    return redirect(url_for('security.forgot_password'))

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    with app.app_context():
        db.create_all()
        if not user_datastore.find_user(email="admin@salesvision.com"):
            user_datastore.create_user(email="admin@salesvision.com", password=hash_password("123456"))
        db.session.commit()
    app.run(debug=True)
