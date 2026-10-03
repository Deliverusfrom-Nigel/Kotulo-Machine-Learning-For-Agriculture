"""Kotulo — SAFEX White Maize Forecaster (Streamlit UI).
Run:  streamlit run app.py
"""

import io
import textwrap
import warnings
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================== page config ==
st.set_page_config(
    page_title="Kotulo | White Maize Forecaster",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================== farm theme ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Nunito+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Nunito Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Lora', serif !important; color: #F5EFE0 !important; }

/* ===================== DARK BACKGROUND FORCED EVERYWHERE ===================== */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHeader"],
[data-testid="stVerticalBlock"] {
    background: linear-gradient(180deg, #2D4A2B 0%, #1F3520 100%) !important;
    color: #F0EBDC !important;
}
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ===================== LIGHT MODE FIX ===================== */
/* Force all text to light colour regardless of theme */
.stApp, .stApp * {
    color-scheme: dark;
}
.stMarkdown, .stMarkdown p, label, .stCaption, p, span, div, li, ul, ol {
    color: #F0EBDC;
}
.stMetric label, .stMetric div { color: #F0EBDC !important; }
[data-testid="stMetricValue"] { color: #F0EBDC !important; }
[data-testid="stMetricLabel"] { color: #C9C4B0 !important; }

/* --- Text inputs --- */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
input[type="text"],
input[type="number"] {
    background-color: rgba(31, 53, 32, 0.75) !important;
    color: #F0EBDC !important;
    border: 1px solid rgba(201, 169, 97, 0.45) !important;
}
.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: rgba(240, 235, 220, 0.55) !important;
}

/* --- Number input stepper buttons --- */
.stNumberInput button {
    background-color: rgba(107, 142, 35, 0.5) !important;
    color: #F0EBDC !important;
    border: none !important;
}
.stNumberInput button svg { fill: #F0EBDC !important; }

/* --- Selectbox / multiselect --- */
[data-baseweb="select"] > div:first-child {
    background-color: rgba(31, 53, 32, 0.75) !important;
    color: #F0EBDC !important;
    border: 1px solid rgba(201, 169, 97, 0.45) !important;
}
[data-baseweb="select"] svg { fill: #F0EBDC !important; }
[data-baseweb="tag"] {
    background-color: #6B8E23 !important;
    color: #F5EFE0 !important;
}
[data-baseweb="popover"],
[role="listbox"],
[role="option"],
[data-baseweb="menu"] {
    background-color: #2D4A2B !important;
    color: #F0EBDC !important;
}
[role="option"]:hover { background-color: #6B8E23 !important; }

/* --- File uploader --- */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(31, 53, 32, 0.65) !important;
    color: #F0EBDC !important;
    border: 1px dashed rgba(201, 169, 97, 0.55) !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] div {
    color: #F0EBDC !important;
}
[data-testid="stFileUploader"] button {
    background-color: #6B8E23 !important;
    color: #F5EFE0 !important;
    border: none !important;
}

/* --- Checkbox --- */
.stCheckbox label,
.stCheckbox p { color: #F0EBDC !important; }

/* --- Slider --- */
.stSlider label,
.stSlider [data-testid="stTickBar"],
.stSlider div { color: #F0EBDC !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #C9A961 !important;
}

/* --- DataFrames --- */
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"] {
    background-color: rgba(245, 239, 224, 0.06) !important;
    border: 1px solid rgba(201, 169, 97, 0.25) !important;
}
[data-testid="stDataFrame"] * { color: #F0EBDC !important; }
[data-testid="stDataFrame"] th {
    background-color: rgba(107, 142, 35, 0.55) !important;
    color: #F5EFE0 !important;
}
[data-testid="stDataFrame"] td {
    background-color: rgba(31, 53, 32, 0.35) !important;
}

/* --- Alerts (info, success, warning, error) --- */
[data-testid="stAlert"],
[data-testid="stAlertContainer"] {
    background-color: rgba(245, 239, 224, 0.10) !important;
    border-left: 4px solid #C9A961 !important;
}
[data-testid="stAlert"] *,
[data-testid="stAlertContainer"] * {
    color: #F0EBDC !important;
    fill: #F0EBDC !important;
}

/* --- Code blocks --- */
code {
    background-color: rgba(0, 0, 0, 0.4) !important;
    color: #C9A961 !important;
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
}
pre {
    background-color: rgba(0, 0, 0, 0.4) !important;
    color: #F0EBDC !important;
    border: 1px solid rgba(201, 169, 97, 0.3) !important;
}

/* --- Buttons --- */
.stButton > button {
    background-color: #6B8E23 !important;
    color: #F5EFE0 !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background-color: #8FBC8F !important;
    color: #1F3520 !important;
}
.stButton > button[kind="primary"] {
    background-color: #6B8E23 !important;
    color: #F5EFE0 !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #8FBC8F !important;
    color: #1F3520 !important;
}
.stDownloadButton > button {
    background-color: #6B8E23 !important;
    color: #F5EFE0 !important;
    border: none !important;
}
.stDownloadButton > button:hover {
    background-color: #8FBC8F !important;
    color: #1F3520 !important;
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] { background-color: transparent !important; }
.stTabs [data-baseweb="tab"] { color: #C9C4B0 !important; }
.stTabs [aria-selected="true"] {
    color: #C9A961 !important;
    border-bottom-color: #C9A961 !important;
}

/* --- Expander --- */
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
    background-color: rgba(245, 239, 224, 0.05) !important;
    color: #F0EBDC !important;
    border: 1px solid rgba(201, 169, 97, 0.3) !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: rgba(245, 239, 224, 0.10) !important;
}
[data-testid="stExpander"] summary svg { fill: #F0EBDC !important; }
[data-testid="stExpander"] summary p {
    color: #F5EFE0 !important;
    font-weight: 600 !important;
}

/* --- Expander titles --- */
[data-testid="stExpander"] summary p {
    font-family: 'Lora', serif !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #F5EFE0 !important;
}

/* --- Status widget --- */
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] * {
    background-color: rgba(31, 53, 32, 0.6) !important;
    color: #F0EBDC !important;
}

/* --- Popover / tooltip --- */
[data-testid="stTooltipContent"] {
    background-color: #2D4A2B !important;
    color: #F0EBDC !important;
}

/* --- Markdown links --- */
.stMarkdown a, a { color: #C9A961 !important; }

/* ===================== HERO, CARDS, HELPERS ===================== */
.hero {
    background: linear-gradient(135deg, #6B8E23 0%, #8FBC8F 100%);
    color: #1F3520; padding: 2.5rem 2rem; border-radius: 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.35); margin-bottom: 1.5rem;
    text-align: center;
}
.hero h1 { color: #1F3520 !important; margin: 0; font-size: 2.4rem; }
.hero p  { color: #1F3520; font-size: 1.1rem; margin-top: 0.6rem; }

.card {
    background: #F5EFE0; padding: 1.5rem 1.8rem; border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    border-left: 8px solid #6B8E23; margin-bottom: 1.2rem;
}
.card, .card * { color: #2D2D2D !important; }
.card-sell { background: #FFEBEE; border-left-color: #C62828; }
.card-hold { background: #E8F5E9; border-left-color: #2E7D32; }
.card-demo { border-left-color: #6B8E23; }
.card-real { border-left-color: #C9A961; }

.rec-big    { font-size: 3rem; font-weight: 700; margin: 0.1rem 0; letter-spacing: 0.02em; }
.rec-sell   { color: #C62828 !important; }
.rec-hold   { color: #2E7D32 !important; }
.rec-reason { font-size: 1.05rem; color: #3A3A3A !important; line-height: 1.55; margin-top: 0.4rem; }
.rec-conf   { font-size: 0.95rem; color: #6B6B6B !important; margin-top: 0.9rem; }

.model-note {
    font-size: 0.92rem; color: #D8D3C4; font-style: italic;
    border-top: 1px dashed #C9A961; padding-top: 0.7rem; margin-top: 0.9rem;
    line-height: 1.55;
}
.model-note a { color: #C9A961 !important; }
.explain {
    background: rgba(245, 239, 224, 0.10);
    border-left: 4px solid #C9A961;
    padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.6rem 0;
    font-size: 0.97rem; color: #F0EBDC; line-height: 1.55;
}
.step-badge {
    display: inline-block; background: #6B8E23; color: white;
    width: 32px; height: 32px; line-height: 32px; border-radius: 50%;
    text-align: center; font-weight: 700; margin-right: 0.6rem;
}
.help-h3 {
    font-family: 'Lora', serif; color: #C9A961; font-size: 1.25rem;
    margin: 1.4rem 0 0.5rem 0; font-weight: 600;
}
.help-h4 {
    color: #F5EFE0; font-size: 1.02rem; font-weight: 700;
    margin: 1rem 0 0.2rem 0;
}
.help-body {
    color: #E5DCC4; font-size: 0.97rem; line-height: 1.6;
    margin: 0 0 0.6rem 0;
}
.help-box {
    background: rgba(245, 239, 224, 0.06);
    border-radius: 10px; padding: 1rem 1.3rem; margin: 0.6rem 0 1.2rem 0;
    border-left: 3px solid #8FBC8F;
}
</style>
""", unsafe_allow_html=True)


# ============================================================ translations ===
LANGUAGES = {
    "English":   "en",
    "Afrikaans": "af",
    "isiXhosa":  "xh",
    "isiZulu":   "zu",
    "Sesotho":   "st",
}

TRANSLATIONS = {
    # ============================================================= ENGLISH
    "en": {
        "app_name": "Kotulo",
        "hero_sub": "A friendly guide to when to sell your SAFEX white maize.",
        "tab_home": "Home",
        "tab_forecast": "Forecast (3 models)",
        "tab_recommend": "What should I do?",
        "tab_help": "Get Help",
        "language_label": "Language",
        "home_explainer": (
            "Welcome to Kotulo, where tech meets agriculture. Our aim is to bring "
            "machine-learning-powered algorithms to our local farmers, so they know "
            "when the right time is to sell — based on years of historical test data. "
            "Small farmers often miss out on the market because they do not have access "
            "to certain information. We aspire to be a solution to this problem, using "
            "three of the best machine learning algorithms: SARIMAX, XGBoost and Seq2Seq."
        ),
        "choose_how": "Choose how you want to use the app",
        "demo_card_title": "🧪 Demo — see how it works",
        "demo_card_body": (
            "Try the app with realistic sample data. No files needed. The models will run, "
            "the recommendation will appear, and you can see the whole flow in under a minute."
        ),
        "demo_btn": "▶️ Try the Demo",
        "real_card_title": "📊 Get Real Price Predictions",
        "real_card_body": (
            "Upload your own SAFEX maize price file and external drivers (rand, rainfall, fuel) "
            "and get a recommendation based on your own data."
        ),
        "real_btn": "📈 Use My Own Data",
        "demo_mode_banner": "🧪 Demo mode — using realistic sample data",
        "advanced_settings": "Advanced settings",
        "train_split_label": "Train / test split",
        "train_split_help": "80% of the history trains the models; the most recent 20% tests them.",
        "province_label": "Province (for holding cost)",
        "holding_cost_label": "Cost of holding (per month, %)",
        "holding_cost_help": "Defaults to your province's estimate.",
        "external_drivers_label": "External drivers",
        "your_data_glance": "Your data at a glance",
        "business_days_metric": "Business days",
        "date_range_metric": "Date range",
        "today_price": "Today's price",
        "province_metric": "Province",
        "show_data": "Show the data",
        "how_to_use": "How to use it",
        "step_1": "Open <b>{forecast_tab}</b> to see what each model says.",
        "step_2": "Open <b>{recommend_tab}</b> and press the big button.",
        "pick_option": "Pick an option above to begin.",
        "upload_section_title": "📊 Upload your data",
        "upload_explainer": (
            "You need two CSV files. The first contains the SAFEX white maize prices "
            "(with a Date column and a Close column in R/ton). The second contains the "
            "external drivers (with a Date column and any of ZAR_USD, Rainfall_mm, Fuel_Price)."
        ),
        "upload_prices_label": "Upload safex_prices.csv",
        "upload_factors_label": "Upload sa_factors.csv",
        "dayfirst_label": "My dates are dd/mm/yyyy (European format)",
        "upload_both_info": "Upload **both** CSV files above to continue.",
        "data_loaded_success": "✓ Data loaded: **{n} business days**",
        "forecast_hero_sub": (
            "Three different models look at the data in three different ways. "
            "Press Run on any of them to see what they say. The recommendation tab "
            "does the hard work for you."
        ),
        "settings_expander": "Settings",
        "run_sarimax_btn": "Run SARIMAX",
        "run_xgboost_btn": "Run XGBoost",
        "run_seq2seq_btn": "Run Seq2Seq",
        "sarimax_p_label": "AR order (p)",
        "sarimax_d_label": "Differencing (d)",
        "sarimax_q_label": "MA order (q)",
        "sarimax_n_label": "Forecast days",
        "xgb_horizon_label": "Horizon (days)",
        "xgb_predict_label": "Predict",
        "xgb_price_lags_label": "Price lags",
        "xgb_driver_lags_label": "Driver lags",
        "xgb_iter_label": "Search iterations",
        "s2s_enc_label": "Encoder length",
        "s2s_pred_label": "Days to predict",
        "s2s_latent_label": "Latent size",
        "s2s_epochs_label": "Max epochs",
        "forecast_success_day": "Forecast for day +{n}: **R {price:,.2f} / ton**",
        "forecast_success_range": "Next {n}-day forecast: **R {price:,.2f} / ton**",
        "sarimax_explainer": (
            "<b>What it does:</b> SARIMAX is a classic statistical model. It studies the pattern "
            "of past prices and how they move with external factors (exchange rate, rainfall, fuel).<br>"
            "<b>How to read it:</b> If the MAE (average error) is smaller than the naive benchmark, it's useful."
        ),
        "xgb_explainer": (
            "<b>What it does:</b> XGBoost is a tree-based machine-learning model. It learns rules like "
            "\"if the rand weakened and fuel rose, the price usually goes up the next day\".<br>"
            "<b>How to read it:</b> Usually the strongest short-term forecaster. Tells you which inputs matter most."
        ),
        "s2s_explainer": (
            "<b>What it does:</b> Seq2Seq reads the last 60 days of history and writes out the next 5 days, "
            "one day at a time.<br>"
            "<b>How to read it:</b> The further ahead it looks, the less accurate it gets — like any forecast."
        ),
        "translate_explanation": "🌍 Translate this explanation",
        "recommend_hero_sub": (
            "Press the button and the app will run all three models, weight them by how well they "
            "performed historically, and give you one clear recommendation."
        ),
        "horizon_label": "Look ahead how many trading days?",
        "horizon_help": "Five days is roughly one trading week.",
        "btn_recommend": "🌟 Get my recommendation",
        "rec_header": "Our recommendation",
        "decision_hold": "HOLD",
        "decision_sell": "SELL",
        "confidence": "Confidence",
        "confidence_line": "{up} of {total} models forecast a price rise. ({down} forecast a fall.)",
        "reason_hold": (
            "The models expect the price to rise by about {pct:.1f}% over the next {days} trading days. "
            "That is comfortably more than the cost of storing your maize, so waiting is likely to pay off."
        ),
        "reason_sell_down": (
            "The models expect the price to fall by about {pct:.1f}% over the next {days} trading days. "
            "Selling now is likely to be better than waiting."
        ),
        "reason_sell_weak": (
            "The models do not see a strong rise in the price over the next {days} trading days. "
            "Because small farms cannot afford risk, we recommend selling now."
        ),
        "avg_forecast": "Average forecast",
        "expected_change": "Expected change",
        "how_each_voted": "How each model voted",
        "table_model": "Model",
        "table_forecast": "Forecast (R/ton)",
        "table_change": "Change",
        "table_direction": "Direction",
        "table_weight": "Weight",
        "table_test_mae": "Test MAE (R/ton)",
        "direction_up": "📈 up",
        "direction_down": "📉 down",
        "direction_flat": "➖ flat",
        "what_means": "What does this mean? (in plain language)",
        "what_means_hold": "{hold} means the models expect the price to rise enough to cover storage costs. If you can wait, waiting is likely to earn you more.",
        "what_means_sell": "{sell} means the models do not see a strong rise in the price. Selling now is likely to be safer.",
        "not_advice": "No forecast is certain. Use this as one input, together with your own knowledge of your farm.",
        "press_button": "Press the button above to get your recommendation.",
        "unlock_panel": "Pick **Demo** or **Real predictions** on the Home tab to unlock this panel.",
        "running_models_status": "Running the three models…",
        "fitting_sarimax": "Fitting SARIMAX…",
        "fitting_xgb": "Fitting XGBoost…",
        "training_s2s": "Training Seq2Seq (~1 min)…",
        "all_finished": "All three models finished ✓",
        "skipped_warning": "{model} skipped: {err}",
        "pdf_download": "📄 Download recommendation as PDF",
        "pdf_unavailable": "📄 PDF download is unavailable — install reportlab with `pip install reportlab`.",
        "sarimax_failed": "SARIMAX failed: {err}",
        "xgb_failed": "XGBoost failed: {err}",
        "s2s_failed": "Seq2Seq failed: {err}",
        "error_could_not_read": "Could not read your data files.",
        "error_common_causes": (
            "**Common causes:**\n"
            "- The dates in your file are not in a format the app recognises.\n"
            "  - If they look like 2024-01-05, **untick** the dd/mm/yyyy box.\n"
            "  - If they look like 05/01/2024, **tick** it.\n"
            "- The file has no Date column, or the column is named differently."
        ),
        "footer": "🌽 Kotulo — an open forecasting tool for SAFEX white maize. Not financial advice.",

        # ---- HELP PANEL ----
        "help_hero_sub": "Plain-language explanations of every term, setting and outcome in the app.",
        "help_intro": (
            "If you are not sure what something means, look it up here. Every term the app uses is "
            "explained in plain language below."
        ),
        "help_models_title": "Our 3 models",
        "help_models_intro": (
            "Kotulo uses three different computer models. Each one looks at the data a little "
            "differently, so together they give a more reliable picture than any one alone."
        ),
        "help_sarimax_title": "SARIMAX",
        "help_sarimax_desc": (
            "A classic statistical model. It studies the pattern of past prices and how those "
            "prices moved when the rand, rainfall or fuel changed. Strong at short-term forecasts "
            "and easy to interpret."
        ),
        "help_xgboost_title": "XGBoost",
        "help_xgboost_desc": (
            "A machine-learning model built from decision trees. It learns rules like \"if the rand "
            "weakened and fuel rose, the price usually goes up the next day\". Usually the strongest "
            "short-term forecaster of the three."
        ),
        "help_seq2seq_title": "Seq2Seq",
        "help_seq2seq_desc": (
            "A neural network that reads the last 60 days of history and writes out the next 5 days, "
            "one day at a time. It is the only model of the three that forecasts a whole week in one go."
        ),
        "help_advanced_title": "Explaining the Advanced Settings",
        "help_advanced_intro": (
            "These settings change how the models are trained. If you are not sure, leave them at "
            "their default values."
        ),
        "help_train_split_title": "Train / test split",
        "help_train_split_desc": (
            "How much of the history the models learn from, versus how much is held back to test "
            "them. The default 80/20 means the models learn from the first 80% of days and are "
            "tested on the most recent 20%."
        ),
        "help_province_title": "Province",
        "help_province_desc": (
            "Your province sets the default cost of storing maize. Free State, North West and "
            "Mpumalanga are the three main maize-producing provinces, and each has a slightly "
            "different cost of transport and silo storage."
        ),
        "help_holding_cost_title": "Cost of holding (per month, %)",
        "help_holding_cost_desc": (
            "How much it costs to keep one tonne of maize in storage for one month, as a percentage "
            "of the maize price. It includes silo fees and the interest on money tied up in the crop. "
            "Higher costs make waiting less worthwhile."
        ),
        "help_drivers_title": "External drivers",
        "help_drivers_desc": (
            "The extra signals the models use to make their forecasts. You can turn each one on or "
            "off. Turning them all on gives the models the most information."
        ),
        "help_driver_explain_title": "Explaining the drivers",
        "help_driver_explain_intro": (
            "Drivers are the outside factors that affect the maize price. Kotulo uses three."
        ),
        "help_zar_title": "ZAR/USD (exchange rate)",
        "help_zar_desc": (
            "How many rand it takes to buy one US dollar. A weaker rand (a higher number) usually "
            "lifts local maize prices, because it makes South African maize more competitive in "
            "export markets."
        ),
        "help_rain_title": "Rainfall (mm)",
        "help_rain_desc": (
            "Rainfall in the maize-growing regions. More rain during the growing season usually "
            "means a bigger crop, which pushes prices down. Too little rain means a smaller crop "
            "and higher prices."
        ),
        "help_fuel_title": "Fuel Price",
        "help_fuel_desc": (
            "The diesel and petrol price. Fuel feeds into the cost of planting, harvesting and "
            "transporting maize. Higher fuel costs can lift the price farmers need to receive to "
            "break even."
        ),
        "help_settings_title": "Explaining the settings of each model",
        "help_settings_intro": (
            "Each model has its own settings. You can leave them all at their defaults — they are "
            "already tuned. This section explains what each one does."
        ),
        "help_sarimax_settings_title": "SARIMAX settings",
        "help_sarimax_p_desc": (
            "AR order (p) — How many past days the model looks back at when predicting today. "
            "Higher values give the model more memory of the past."
        ),
        "help_sarimax_d_desc": (
            "Differencing (d) — How many times the price series is transformed before the model "
            "sees it. A value of 1 is standard for prices and removes the long-term trend."
        ),
        "help_sarimax_q_desc": (
            "MA order (q) — How many past forecast errors the model corrects for. Higher values "
            "make the model more responsive to recent mistakes."
        ),
        "help_sarimax_n_desc": (
            "Forecast days — How many days ahead the model forecasts. Five days is roughly one "
            "trading week."
        ),
        "help_xgb_settings_title": "XGBoost settings",
        "help_xgb_horizon_desc": (
            "Horizon (days) — How many days ahead the model predicts. Use 1 for tomorrow's price "
            "or 5 for next week."
        ),
        "help_xgb_predict_desc": (
            "Predict — Whether the model predicts the price change (diff) or the raw price (level). "
            "\"diff\" is more robust because prices can climb beyond what the model saw in training."
        ),
        "help_xgb_plags_desc": (
            "Price lags — How many previous days of price changes the model uses as inputs."
        ),
        "help_xgb_dlags_desc": (
            "Driver lags — How many previous days of each external driver (rand, rain, fuel) the "
            "model uses as inputs."
        ),
        "help_xgb_iter_desc": (
            "Search iterations — How many combinations of settings the tuner tries. Higher values "
            "take longer but often find a slightly better model."
        ),
        "help_s2s_settings_title": "Seq2Seq settings",
        "help_s2s_enc_desc": (
            "Encoder length — How many days of history the encoder reads before making a forecast."
        ),
        "help_s2s_pred_desc": (
            "Days to predict — How many future days the decoder writes out, one at a time."
        ),
        "help_s2s_latent_desc": (
            "Latent size — The size of the model's internal memory. Higher values give the model "
            "more capacity, but too high can lead to overfitting."
        ),
        "help_s2s_epochs_desc": (
            "Max epochs — The maximum number of training passes. The model usually stops early "
            "when it stops improving."
        ),
        "help_rec_title": "Explaining the Get my recommendation section",
        "help_rec_intro": (
            "When you press the button, Kotulo runs all three models, weights them by historical "
            "accuracy, and gives you one decision. Here is what each part of the result means."
        ),
        "help_rec_hold_title": "HOLD (green)",
        "help_rec_hold_desc": (
            "The models expect the price to rise by more than the cost of storing your maize, "
            "plus a safety margin. Waiting is likely to earn you more money. HOLD is only "
            "recommended when at least 60% of the models agree on the rise."
        ),
        "help_rec_sell_title": "SELL (red)",
        "help_rec_sell_desc": (
            "The models do not see a strong rise in the price. Selling now is likely to be safer "
            "than waiting. This is the default for small farms that cannot afford to take risk."
        ),
        "help_rec_conf_title": "Confidence line",
        "help_rec_conf_desc": (
            "Shows how many of the three models voted for a price rise and how many voted for a "
            "fall. If all three agree, the recommendation is more reliable."
        ),
        "help_rec_votes_title": "How each model voted",
        "help_rec_votes_desc": (
            "A table showing what each model forecast, its predicted change from today, its "
            "direction (up, down or flat), its weight in the final forecast, and its test error "
            "(MAE). Models with a lower test error get more weight."
        ),
        "help_rec_metrics_title": "The three numbers below the decision",
        "help_rec_metrics_desc": (
            "Today's price is the current SAFEX price. The average forecast is what the "
            "weighted models expect the price to be in the future horizon. The expected change "
            "is the difference, shown as a percentage."
        ),
        "help_rec_pdf_title": "Download recommendation as PDF",
        "help_rec_pdf_desc": (
            "Saves the recommendation as a one-page PDF you can print or take to the co-op."
        ),
    },

    # ============================================================= AFRIKAANS
    "af": {
        "app_name": "Kotulo",
        "hero_sub": "'n Vriendelike gids oor wanneer om jou SAFEX witmielies te verkoop.",
        "tab_home": "Tuis",
        "tab_forecast": "Voorspelling (3 modelle)",
        "tab_recommend": "Wat moet ek doen?",
        "tab_help": "Kry Hulp",
        "language_label": "Taal",
        "home_explainer": (
            "Welkom by Kotulo, waar tegnologie en landbou mekaar ontmoet. Ons doel is om "
            "masjienleer-aangedrewe algoritmes na ons plaaslike boere te bring, sodat hulle "
            "weet wanneer die regte tyd is om te verkoop — gebaseer op jare se historiese "
            "toetsdata. Kleinboere verloor dikwels toegang tot die mark omdat hulle nie "
            "toegang tot sekere inligting het nie. Ons streef daarna om 'n oplossing vir "
            "hierdie probleem te wees, met drie van die beste masjienleer-algoritmes: "
            "SARIMAX, XGBoost en Seq2Seq."
        ),
        "choose_how": "Kies hoe jy die app wil gebruik",
        "demo_card_title": "🧪 Demo — sien hoe dit werk",
        "demo_card_body": (
            "Probeer die app met realistiese voorbeelddata. Geen lêers nodig nie. Die modelle "
            "hardloop, die aanbeveling verskyn, en jy kan die hele vloei in minder as 'n minuut sien."
        ),
        "demo_btn": "▶️ Probeer die Demo",
        "real_card_title": "📊 Kry Ware Prysvoorspellings",
        "real_card_body": (
            "Laai jou eie SAFEX-mielieprys-lêer en eksterne drywers (rand, reënval, brandstof) op "
            "en kry 'n aanbeveling gebaseer op jou eie data."
        ),
        "real_btn": "📈 Gebruik My Eie Data",
        "demo_mode_banner": "🧪 Demo-modus — gebruik realistiese voorbeelddata",
        "advanced_settings": "Gevorderde instellings",
        "train_split_label": "Opvoeding / toets-verdeling",
        "train_split_help": "80% van die geskiedenis lei die modelle op; die jongste 20% toets hulle.",
        "province_label": "Provinsie (vir stoor-koste)",
        "holding_cost_label": "Koste van stoor (per maand, %)",
        "holding_cost_help": "Verstek na jou provinsie se skatting.",
        "external_drivers_label": "Eksterne drywers",
        "your_data_glance": "Jou data in 'n oogopslag",
        "business_days_metric": "Sakedae",
        "date_range_metric": "Datumreeks",
        "today_price": "Vandag se prys",
        "province_metric": "Provinsie",
        "show_data": "Wys die data",
        "how_to_use": "Hoe om dit te gebruik",
        "step_1": "Maak <b>{forecast_tab}</b> oop om te sien wat elke model sê.",
        "step_2": "Maak <b>{recommend_tab}</b> oop en druk die groot knoppie.",
        "pick_option": "Kies 'n opsie hierbo om te begin.",
        "upload_section_title": "📊 Laai jou data op",
        "upload_explainer": (
            "Jy benodig twee CSV-lêers. Die eerste bevat die SAFEX witmielie-pryse "
            "(met 'n Datum-kolom en 'n Close-kolom in R/ton). Die tweede bevat die eksterne "
            "drywers (met 'n Datum-kolom en enige van ZAR_USD, Rainfall_mm, Fuel_Price)."
        ),
        "upload_prices_label": "Laai safex_prices.csv op",
        "upload_factors_label": "Laai sa_factors.csv op",
        "dayfirst_label": "My datums is dd/mm/jjjj (Europese formaat)",
        "upload_both_info": "Laai **beide** CSV-lêers hierbo op om voort te gaan.",
        "data_loaded_success": "✓ Data gelaai: **{n} sakedae**",
        "forecast_hero_sub": (
            "Drie verskillende modelle kyk op drie verskillende maniere na die data. "
            "Druk Hardloop op enige van hulle om te sien wat hulle sê. Die aanbeveling-tab "
            "doen die harde werk vir jou."
        ),
        "settings_expander": "Instellings",
        "run_sarimax_btn": "Hardloop SARIMAX",
        "run_xgboost_btn": "Hardloop XGBoost",
        "run_seq2seq_btn": "Hardloop Seq2Seq",
        "sarimax_p_label": "AR-orde (p)",
        "sarimax_d_label": "Verskil (d)",
        "sarimax_q_label": "MA-orde (q)",
        "sarimax_n_label": "Voorspellingsdae",
        "xgb_horizon_label": "Horison (dae)",
        "xgb_predict_label": "Voorspel",
        "xgb_price_lags_label": "Prys-vertragings",
        "xgb_driver_lags_label": "Drywer-vertragings",
        "xgb_iter_label": "Soek-iterasies",
        "s2s_enc_label": "Enkodeerder-lengte",
        "s2s_pred_label": "Dae om te voorspel",
        "s2s_latent_label": "Latente grootte",
        "s2s_epochs_label": "Maks epogge",
        "forecast_success_day": "Voorspelling vir dag +{n}: **R {price:,.2f} / ton**",
        "forecast_success_range": "Volgende {n}-dag voorspelling: **R {price:,.2f} / ton**",
        "sarimax_explainer": (
            "<b>Wat dit doen:</b> SARIMAX is 'n klassieke statistiese model. Dit bestudeer die "
            "patroon van vorige pryse en hoe hulle beweeg met eksterne faktore.<br>"
            "<b>Hoe om dit te lees:</b> As die MAE kleiner is as die naïewe maatstaf, is dit nuttig."
        ),
        "xgb_explainer": (
            "<b>Wat dit doen:</b> XGBoost is 'n boom-gebaseerde masjienleer-model. Dit leer reëls "
            "soos \"as die rand verswak en brandstof styg, gaan die prys gewoonlik op\".<br>"
            "<b>Hoe om dit te lees:</b> Gewoonlik die sterkste korttermyn-voorspeller."
        ),
        "s2s_explainer": (
            "<b>Wat dit doen:</b> Seq2Seq lees die laaste 60 dae se geskiedenis en skryf die "
            "volgende 5 dae uit, een dag op 'n slag.<br>"
            "<b>Hoe om dit te lees:</b> Hoe verder dit vorentoe kyk, hoe minder akkuraat word dit."
        ),
        "translate_explanation": "🌍 Vertaal hierdie verduideliking",
        "recommend_hero_sub": (
            "Druk die knoppie en die app sal al drie modelle hardloop, hulle weeg volgens hoe goed "
            "hulle histories gevaar het, en vir jou een duidelike aanbeveling gee."
        ),
        "horizon_label": "Hoeveel sakedae vorentoe kyk?",
        "horizon_help": "Vyf dae is omtrent een handelsweek.",
        "btn_recommend": "🌟 Kry my aanbeveling",
        "rec_header": "Ons aanbeveling",
        "decision_hold": "HOU",
        "decision_sell": "VERKOOP",
        "confidence": "Vertroue",
        "confidence_line": "{up} van {total} modelle voorspel 'n prysstyging. ({down} voorspel 'n daling.)",
        "reason_hold": (
            "Die modelle verwag die prys sal met ongeveer {pct:.1f}% styg oor die volgende "
            "{days} handelsdae. Dit is meer as die koste om jou mielies te stoor, so dit behoort "
            "te loon om te wag."
        ),
        "reason_sell_down": (
            "Die modelle verwag die prys sal met ongeveer {pct:.1f}% daal oor die volgende "
            "{days} handelsdae. Om nou te verkoop is waarskynlik beter as om te wag."
        ),
        "reason_sell_weak": (
            "Die modelle sien nie 'n sterk styging in die prys oor die volgende {days} handelsdae "
            "nie. Omdat klein plase nie risiko kan bekostig nie, beveel ons aan om nou te verkoop."
        ),
        "avg_forecast": "Gemiddelde voorspelling",
        "expected_change": "Verwagte verandering",
        "how_each_voted": "Hoe elke model gestem het",
        "table_model": "Model",
        "table_forecast": "Voorspelling (R/ton)",
        "table_change": "Verandering",
        "table_direction": "Rigting",
        "table_weight": "Gewig",
        "table_test_mae": "Toets-MAE (R/ton)",
        "direction_up": "📈 op",
        "direction_down": "📉 af",
        "direction_flat": "➖ plat",
        "what_means": "Wat beteken dit? (in eenvoudige taal)",
        "what_means_hold": "<b>{hold}</b> beteken die modelle verwag die prys sal genoeg styg om stookoste te dek. As jy kan wag, sal wag waarskynlik meer verdien.",
        "what_means_sell": "<b>{sell}</b> beteken die modelle sien nie 'n sterk styging in die prys nie. Om nou te verkoop is waarskynlik veiliger.",
        "not_advice": "Geen voorspelling is seker nie. Gebruik dit saam met jou eie kennis van jou plaas.",
        "press_button": "Druk die knoppie hierbo om jou aanbeveling te kry.",
        "unlock_panel": "Kies **Demo** of **Ware voorspellings** op die Tuis-tab om hierdie paneel oop te sluit.",
        "running_models_status": "Hardloop die drie modelle…",
        "fitting_sarimax": "Pas SARIMAX aan…",
        "fitting_xgb": "Pas XGBoost aan…",
        "training_s2s": "Lei Seq2Seq op (~1 min)…",
        "all_finished": "Al drie modelle klaar ✓",
        "skipped_warning": "{model} oorgeslaan: {err}",
        "pdf_download": "📄 Laai aanbeveling as PDF af",
        "pdf_unavailable": "📄 PDF-aflaai is nie beskikbaar nie — installeer reportlab met `pip install reportlab`.",
        "sarimax_failed": "SARIMAX het gefaal: {err}",
        "xgb_failed": "XGBoost het gefaal: {err}",
        "s2s_failed": "Seq2Seq het gefaal: {err}",
        "error_could_not_read": "Kon nie jou datalêers lees nie.",
        "error_common_causes": (
            "**Algemene oorsake:**\n"
            "- Die datums in jou lêer is nie in 'n formaat wat die app herken nie.\n"
            "  - As dit lyk soos 2024-01-05, **ontmerk** die dd/mm/jjjj-boks.\n"
            "  - As dit lyk soos 05/01/2024, **merk** dit.\n"
            "- Die lêer het nie 'n Datum-kolom nie, of die kolom het 'n ander naam."
        ),
        "footer": "🌽 Kotulo — 'n oop voorspellingsinstrument vir SAFEX witmielies. Nie finansiële advies nie.",

        "help_hero_sub": "Eenvoudige verduidelikings van elke term, instelling en uitkoms in die app.",
        "help_intro": "As jy nie seker is wat iets beteken nie, soek dit hier op. Elke term wat die app gebruik, word hieronder verduidelik.",
        "help_models_title": "Ons 3 modelle",
        "help_models_intro": (
            "Kotulo gebruik drie verskillende rekenaarmodelle. Elkeen kyk effens anders na die data, "
            "so saam gee hulle 'n meer betroubare beeld as een alleen."
        ),
        "help_sarimax_title": "SARIMAX",
        "help_sarimax_desc": (
            "’n Klassieke statistiese model. Dit bestudeer die patroon van vorige pryse en hoe "
            "daardie pryse beweeg het toe die rand, reënval of brandstof verander het. Sterk met "
            "korttermynvoorspellings en maklik om te interpreteer."
        ),
        "help_xgboost_title": "XGBoost",
        "help_xgboost_desc": (
            "’n Masjienleer-model wat uit besluitnemingsbome gebou is. Dit leer reëls soos "
            "\"as die rand verswak en brandstof styg, styg die prys gewoonlik die volgende dag\". "
            "Gewoonlik die sterkste korttermynvoorspeller van die drie."
        ),
        "help_seq2seq_title": "Seq2Seq",
        "help_seq2seq_desc": (
            "’n Neurale netwerk wat die laaste 60 dae se geskiedenis lees en die volgende 5 dae "
            "uit skryf, een dag op ’n slag. Dit is die enigste model wat ’n hele week op een slag "
            "voorspel."
        ),
        "help_advanced_title": "Verduideliking van die Gevorderde Instellings",
        "help_advanced_intro": (
            "Hierdie instellings verander hoe die modelle opgelei word. As jy onseker is, los hulle "
            "op hul verstekwaardes."
        ),
        "help_train_split_title": "Opvoeding / toets-verdeling",
        "help_train_split_desc": (
            "Hoeveel van die geskiedenis die modelle leer, teenoor hoeveel teruggehou word om hulle "
            "te toets. Die verstek 80/20 beteken die modelle leer van die eerste 80% van die dae en "
            "word getoets op die jongste 20%."
        ),
        "help_province_title": "Provinsie",
        "help_province_desc": (
            "Jou provinsie stel die verstekkoste van die stoor van mielies. Vrystaat, Noordwes en "
            "Mpumalanga is die drie hoof-mielieproduserende provinsies."
        ),
        "help_holding_cost_title": "Koste van stoor (per maand, %)",
        "help_holding_cost_desc": (
            "Hoeveel dit kos om een ton mielies vir een maand te stoor, as ’n persentasie van die "
            "mielieprys. Dit sluit silo-geld en rente op geld wat in die oes vasgelê is in. Hoër "
            "koste maak wag minder die moeite werd."
        ),
        "help_drivers_title": "Eksterne drywers",
        "help_drivers_desc": (
            "Die ekstra seine wat die modelle gebruik om voorspellings te maak. Jy kan elkeen aan "
            "of af skakel."
        ),
        "help_driver_explain_title": "Verduideliking van die drywers",
        "help_driver_explain_intro": "Drywers is die eksterne faktore wat die mielieprys beïnvloed. Kotulo gebruik drie.",
        "help_zar_title": "ZAR/USD (wisselkoers)",
        "help_zar_desc": (
            "Hoeveel rand dit neem om een Amerikaanse dollar te koop. ’n Swakker rand (hoër getal) "
            "lig gewoonlik plaaslike mieliepryse."
        ),
        "help_rain_title": "Reënval (mm)",
        "help_rain_desc": (
            "Reënval in die mielieproduserende streke. Meer reën beteken gewoonlik ’n groter oes, "
            "wat pryse afdwing. Te min reën beteken ’n kleiner oes en hoër pryse."
        ),
        "help_fuel_title": "Brandstofprys",
        "help_fuel_desc": (
            "Die diesel- en petrolprys. Brandstof voed in die koste van plant, oes en vervoer van "
            "mielies. Hoër brandstofkoste kan die prys lig wat produsente moet ontvang om gelyk te breek."
        ),
        "help_settings_title": "Verduideliking van elke model se instellings",
        "help_settings_intro": (
            "Elke model het sy eie instellings. Jy kan hulle almal op verstek los — hulle is reeds "
            "opgestel. Hierdie afdeling verduidelik wat elkeen doen."
        ),
        "help_sarimax_settings_title": "SARIMAX-instellings",
        "help_sarimax_p_desc": "AR-orde (p) — Hoeveel vorige dae die model terugkyk wanneer dit vandag voorspel.",
        "help_sarimax_d_desc": "Verskil (d) — Hoeveel keer die prysreeks getransformeer word voordat die model dit sien. ’n Waarde van 1 is standaard.",
        "help_sarimax_q_desc": "MA-orde (q) — Hoeveel vorige voorspellingsfoute die model regstel.",
        "help_sarimax_n_desc": "Voorspellingsdae — Hoeveel dae vorentoe die model voorspel.",
        "help_xgb_settings_title": "XGBoost-instellings",
        "help_xgb_horizon_desc": "Horison (dae) — Hoeveel dae vorentoe die model voorspel. Gebruik 1 vir môre of 5 vir volgende week.",
        "help_xgb_predict_desc": "Voorspel — Of die model die prysverandering (diff) of die rou prys (level) voorspel. \"diff\" is meer robuust.",
        "help_xgb_plags_desc": "Prys-vertragings — Hoeveel vorige dae se prysveranderinge die model as insette gebruik.",
        "help_xgb_dlags_desc": "Drywer-vertragings — Hoeveel vorige dae van elke eksterne drywer die model as insette gebruik.",
        "help_xgb_iter_desc": "Soek-iterasies — Hoeveel kombinasies van instellings die tuner probeer.",
        "help_s2s_settings_title": "Seq2Seq-instellings",
        "help_s2s_enc_desc": "Enkodeerder-lengte — Hoeveel dae se geskiedenis die enkodeerder lees voordat dit ’n voorspelling maak.",
        "help_s2s_pred_desc": "Dae om te voorspel — Hoeveel toekomstige dae die dekodeerder uitskryf, een op ’n slag.",
        "help_s2s_latent_desc": "Latente grootte — Die grootte van die model se interne geheue.",
        "help_s2s_epochs_desc": "Maks epogge — Die maksimum aantal opvoedingsrondtes.",
        "help_rec_title": "Verduideliking van Kry my aanbeveling",
        "help_rec_intro": (
            "Wanneer jy die knoppie druk, hardloop Kotulo al drie modelle, weeg hulle volgens "
            "historiese akkuraatheid, en gee jou een besluit. Hier is wat elke deel beteken."
        ),
        "help_rec_hold_title": "HOU (groen)",
        "help_rec_hold_desc": (
            "Die modelle verwag die prys sal styg met meer as die koste om jou mielies te stoor, "
            "plus ’n veiligheidsmarge. Wag sal waarskynlik meer geld verdien. HOU word slegs "
            "aanbeveel wanneer minstens 60% van die modelle saamstem."
        ),
        "help_rec_sell_title": "VERKOOP (rooi)",
        "help_rec_sell_desc": (
            "Die modelle sien nie ’n sterk styging in die prys nie. Om nou te verkoop is "
            "waarskynlik veiliger. Dit is die verstek vir klein plase."
        ),
        "help_rec_conf_title": "Vertroue-lyn",
        "help_rec_conf_desc": (
            "Wys hoeveel van die drie modelle vir ’n prysstyging gestem het en hoeveel vir ’n daling."
        ),
        "help_rec_votes_title": "Hoe elke model gestem het",
        "help_rec_votes_desc": (
            "’n Tabel wat wys wat elke model voorspel het, die voorspelde verandering, rigting, "
            "gewig in die finale voorspelling, en toetsfout (MAE)."
        ),
        "help_rec_metrics_title": "Die drie nommers onder die besluit",
        "help_rec_metrics_desc": (
            "Vandag se prys is die huidige SAFEX-prys. Die gemiddelde voorspelling is wat die "
            "geweegde modelle verwag. Die verwagte verandering is die verskil as ’n persentasie."
        ),
        "help_rec_pdf_title": "Laai aanbeveling as PDF af",
        "help_rec_pdf_desc": "Stoor die aanbeveling as ’n eenblad-PDF wat jy kan druk of saamneem co-op toe.",
    },

    # ============================================================= ISIXHOSA
    "xh": {
        "app_name": "Kotulo",
        "hero_sub": "Isikhokelo esilula sokuba uthengise nini ummbona wakho we-SAFEX.",
        "tab_home": "Ikhaya",
        "tab_forecast": "Uqikelelo (iimodeli ezi-3)",
        "tab_recommend": "Ndingenza ntoni?",
        "tab_help": "Fumana Uncedo",
        "language_label": "Ulwimi",
        "home_explainer": (
            "Wamkelekile kwi-Kotulo, apho itekhnoloji idibana nezolimo. Injongo yethu "
            "kukuzisa ii-algorithms eziqhutywa kukufunda ngomatshini kubalimi bethu basekhaya, "
            "ukuze bazi ukuba leliphi ixesha elifanelekileyo lokuthengisa — ngokusekelwe "
            "kwiminyaka yedatha yovavanyo lwangaphambili. Ambalwa amafama amancinci aphulukana "
            "nethuba lokungena emarikeni ngenxa yokungabi nalo ulwazi oluthile. Sifuna ukuba "
            "sisisombululo sale ngxaki, sisebenzisa ezintathu kwezona algorithms zibalaseleyo "
            "zokufunda ngomatshini: i-SARIMAX, i-XGBoost kunye ne-Seq2Seq."
        ),
        "choose_how": "Khetha indlela ofuna ukuyisebenzisa ngayo le app",
        "demo_card_title": "🧪 Umboniso — bona indlela esebenza ngayo",
        "demo_card_body": "Zama i-app ngedatha yomzekelo. Akukho fayile zifunekayo. Iimodeli ziya kusebenza, ingcebiso iya kuvela.",
        "demo_btn": "▶️ Zama uMboniso",
        "real_card_title": "📊 Fumana uQikelelo lweXabiso lokwenene",
        "real_card_body": "Faka ifayile yakho yexabiso lommbona we-SAFEX kunye neempembelelo zangaphandle kwaye ufumane ingcebiso.",
        "real_btn": "📈 Sebenzisa iDatha Yam",
        "demo_mode_banner": "🧪 Imowudi yomboniso — kusetyenziswa idatha yomzekelo",
        "advanced_settings": "Iisetingi eziphambili",
        "train_split_label": "Ukwahlula koqeqesho / kovavanyo",
        "train_split_help": "I-80% yembali iqeqesha iimodeli; i-20% yamva nje iyazivavanya.",
        "province_label": "Iphondo (kwiindleko zokugcina)",
        "holding_cost_label": "Iindleko zokugcina (ngenyanga, %)",
        "holding_cost_help": "Imilinganiselo yephondo lakho ngokuzenzekelayo.",
        "external_drivers_label": "Iimpembelelo zangaphandle",
        "your_data_glance": "Idatha yakho ngedwa",
        "business_days_metric": "Iintsuku zokusebenza",
        "date_range_metric": "Uluhlu lwemihla",
        "today_price": "Ixabiso lanamhlanje",
        "province_metric": "Iphondo",
        "show_data": "Bonisa idatha",
        "how_to_use": "Indlela yokuyisebenzisa",
        "step_1": "Vula <b>{forecast_tab}</b> ukubona into ethethwa yimodeli nganye.",
        "step_2": "Vula <b>{recommend_tab}</b> kwaye ucofe iqhosha elikhulu.",
        "pick_option": "Khetha ukhetho olungasentla ukuqala.",
        "upload_section_title": "📊 Faka idatha yakho",
        "upload_explainer": "Udinga iifayile ezimbini ze-CSV. Eyokuqala iqulethe amaxabiso ommbona we-SAFEX. Eyesibini iqulethe iimpembelelo zangaphandle.",
        "upload_prices_label": "Faka safex_prices.csv",
        "upload_factors_label": "Faka sa_factors.csv",
        "dayfirst_label": "Iimihla zam nge-dd/mm/yyyy",
        "upload_both_info": "Faka **zombini** iifayile ze-CSV apha ngasentla ukuqhubeka.",
        "data_loaded_success": "✓ Idatha ilayishiwe: **{n} iintsuku zokusebenza**",
        "forecast_hero_sub": "Iimodeli ezintathu zijonga idatha ngeendlela ezintathu ezahlukeneyo. Cofa u-Qalisa kuyo nayiphi na.",
        "settings_expander": "Iisetingi",
        "run_sarimax_btn": "Qalisa i-SARIMAX",
        "run_xgboost_btn": "Qalisa i-XGBoost",
        "run_seq2seq_btn": "Qalisa i-Seq2Seq",
        "sarimax_p_label": "I-AR order (p)",
        "sarimax_d_label": "Umahluko (d)",
        "sarimax_q_label": "I-MA order (q)",
        "sarimax_n_label": "Iintsuku zokuqikelela",
        "xgb_horizon_label": "Umda (iintsuku)",
        "xgb_predict_label": "Qikelela",
        "xgb_price_lags_label": "Ukulibaziseka kwexabiso",
        "xgb_driver_lags_label": "Ukulibaziseka kweempembelelo",
        "xgb_iter_label": "Uphando lokuphinda",
        "s2s_enc_label": "Ubude be-encoder",
        "s2s_pred_label": "Iintsuku zokuqikelela",
        "s2s_latent_label": "Ubungakanani obufihlakeleyo",
        "s2s_epochs_label": "Ubuninzi be-epoch",
        "forecast_success_day": "Uqikelelo losuku +{n}: **R {price:,.2f} / ton**",
        "forecast_success_range": "Uqikelelo lweentsuku ezi-{n}: **R {price:,.2f} / ton**",
        "sarimax_explainer": "<b>Into eyenzayo:</b> I-SARIMAX yimodeli yezibalo yakudala. Ifunda ipatheni yamaxabiso adlulileyo.<br><b>Indlela yokuyifunda:</b> Ukuba i-MAE incinci kunesiseko esilula, iyaluncedo.",
        "xgb_explainer": "<b>Into eyenzayo:</b> I-XGBoost yimodeli yokufunda ngomatshini. Ifunda imithetho malunga nokuhamba kwamaxabiso.<br><b>Indlela yokuyifunda:</b> Ngokuqhelekileyo eyona nto inamandla.",
        "s2s_explainer": "<b>Into eyenzayo:</b> I-Seq2Seq ifunda iintsuku ezi-60 zokugqibela kwaye ibhale iintsuku ezi-5 ezilandelayo.<br><b>Indlela yokuyifunda:</b> Okukhona ijonga phambili, kokukhona ingachanekanga.",
        "translate_explanation": "🌍 Guqulela le ngcaciso",
        "recommend_hero_sub": "Cofa iqhosha kwaye i-app iya kuqalisa iimodeli zontathu, izilinganise, kwaye ikunike ingcebiso enye.",
        "horizon_label": "Jonga phambili iintsuku ezingaphi?",
        "horizon_help": "Iintsuku ezintlanu phantse iveki enye yorhwebo.",
        "btn_recommend": "🌟 Fumana ingcebiso yam",
        "rec_header": "Ingcebiso yethu",
        "decision_hold": "GCINA",
        "decision_sell": "THENGISA",
        "confidence": "Ukuzithemba",
        "confidence_line": "{up} kwiimodeli ezi-{total} ziqikelela ukunyuka. ({down} ziqikelela ukuhla.)",
        "reason_hold": "Iimodeli zilindele ukuba ixabiso linyuke nge-{pct:.1f}% kwiintsuku ezi-{days} ezizayo. Oku kungaphezulu kweendleko zokugcina ummbona.",
        "reason_sell_down": "Iimodeli zilindele ukuba ixabiso lehle nge-{pct:.1f}% kwiintsuku ezi-{days} ezizayo. Ukuthengisa ngoku kungcono kunokulinda.",
        "reason_sell_weak": "Iimodeli aziboni kunyuka okunamandla kwiintsuku ezi-{days} ezizayo. Kuba iifama ezincinci azinakho ukuthatha umngcipheko, sicebisa ukuba uthengise ngoku.",
        "avg_forecast": "Uqikelelo oluphakathi",
        "expected_change": "Utshintsho olulindelekileyo",
        "how_each_voted": "Indlela imodeli nganye evote ngayo",
        "table_model": "Imodeli",
        "table_forecast": "Uqikelelo (R/ton)",
        "table_change": "Utshintsho",
        "table_direction": "Indlela",
        "table_weight": "Ubunzima",
        "table_test_mae": "Uvavanyo lwe-MAE (R/ton)",
        "direction_up": "📈 phezulu",
        "direction_down": "📉 phantsi",
        "direction_flat": "➖ tyaba",
        "what_means": "Oku kuthetha ntoni?",
        "what_means_hold": "<b>{hold}</b> kuthetha ukuba iimodeli zilindele ukuba ixabiso linyuke ngokwaneleyo ukuhlawulela iindleko zokugcina.",
        "what_means_sell": "<b>{sell}</b> kuthetha ukuba iimodeli aziboni kunyuka okunamandla. Ukuthengisa ngoku kunokukhuselekile.",
        "not_advice": "Akukho qikelelo liqinisekileyo. Sebenzisa oku kunye nolwazi lwakho ngefu yakho.",
        "press_button": "Cofa iqhosha elingasentla ukufumana ingcebiso yakho.",
        "unlock_panel": "Khetha **uMboniso** okanye **Uqikelelo lokwenene** kwiThebhu yeKhaya ukuvula eli paneli.",
        "running_models_status": "Kuqhutywa iimodeli ezintathu…",
        "fitting_sarimax": "Kulungiswa i-SARIMAX…",
        "fitting_xgb": "Kulungiswa i-XGBoost…",
        "training_s2s": "Kuqeqeshwa i-Seq2Seq (~1 min)…",
        "all_finished": "Zonke iimodeli zontathu zigqibile ✓",
        "skipped_warning": "{model} itsityiwe: {err}",
        "pdf_download": "📄 Khuphela ingcebiso njenge-PDF",
        "pdf_unavailable": "📄 Ukukhuphela i-PDF akufumaneki — faka i-reportlab.",
        "sarimax_failed": "I-SARIMAX yehlile: {err}",
        "xgb_failed": "I-XGBoost yehlile: {err}",
        "s2s_failed": "I-Seq2Seq yehlile: {err}",
        "error_could_not_read": "Ayingakwazanga ukufunda iifayile zakho zedatha.",
        "error_common_causes": "**Oonobangela abaqhelekileyo:**\n- Iimihla aziqondakali.\n  - 2024-01-05: **susa** ibhokisi ye-dd/mm/yyyy.\n  - 05/01/2024: **yikhonkcoze**.\n- Ifayile ayinayo ikholamu ye-Date.",
        "footer": "🌽 Kotulo — isixhobo esivulekileyo sokuqikelela ummbona we-SAFEX. Ayingcebiso yemali.",

        "help_hero_sub": "Iinkcazo zolwimi olulula zawo onke amagama, iisetingi, neziphumo kwi-app.",
        "help_intro": "Ukuba awuqinisekanga ukuba into ethile ithetha ntoni, khangela apha. Onke amagama asetyenziswa yi-app achazwe ngezantsi.",
        "help_models_title": "Iimodeli zethu ezi-3",
        "help_models_intro": "I-Kotulo isebenzisa iimodeli ezintathu ezahlukeneyo zekhompyutha. Zonke zijonga idatha ngeendlela ezahlukeneyo.",
        "help_sarimax_title": "SARIMAX",
        "help_sarimax_desc": "Imodeli yezibalo yakudala. Ifunda ipatheni yamaxabiso adlulileyo kunye nendlela ahamba ngayo xa i-rand, imvula okanye amafutha etshintsha.",
        "help_xgboost_title": "XGBoost",
        "help_xgboost_desc": "Imodeli yokufunda ngomatshini eyakhiwe ngezihlahla. Ifunda imithetho efana nokuthi \"xa i-rand iba buthaka, ixabiso linyuka\".",
        "help_seq2seq_title": "Seq2Seq",
        "help_seq2seq_desc": "Inethiwekhi ye-neural efunda iintsuku ezi-60 zokugqibela kwaye ibhale iintsuku ezi-5 ezilandelayo, usuku ngolunye.",
        "help_advanced_title": "Inkcazo yeeSetingi eziPhambili",
        "help_advanced_intro": "Ezi setingi zitshintsha indlela iimodeli eziqeqeshwa ngayo. Ukuba awuqinisekanga, shiya kwi-default.",
        "help_train_split_title": "Ukwahlula koqeqesho / kovavanyo",
        "help_train_split_desc": "Mingaphi amasiko iimodeli ezifunda kuwo, ngokuchasene nokususa eceleni ukuvavanya. I-80/20 yindlela eqhelekileyo.",
        "help_province_title": "Iphondo",
        "help_province_desc": "Iphondo lakho liseta iindleko zokugcina ummbona. IFree State, North West, Mpumalanga zezona phondo ziphambili zokulima ummbona.",
        "help_holding_cost_title": "Iindleko zokugcina (ngenyanga, %)",
        "help_holding_cost_desc": "Yimalini ukugcina itoni enye yombona inyanga enye. Iquka iifizi zokugcina kunye nenzala yemali ebotshelelweyo esivunweni.",
        "help_drivers_title": "Iimpembelelo zangaphandle",
        "help_drivers_desc": "Iimpawu ezongezelelweyo ezisetyenziswa ziimodeli ukwenza uqikelelo. Ungavula okanye uvala nganye.",
        "help_driver_explain_title": "Inkcazo yeempembelelo",
        "help_driver_explain_intro": "Iimpembelelo zezona zinto zangaphandle ezichaphazela ixabiso lombona. I-Kotulo isebenzisa ezintathu.",
        "help_zar_title": "ZAR/USD (izinga lotshintshiselwano)",
        "help_zar_desc": "Mangaphi ama-rand okuthenga idola enye yaseMelika. I-rand ebutheleleyo inyusa amaxabiso ombona wasekhaya.",
        "help_rain_title": "Imvula (mm)",
        "help_rain_desc": "Imvula kwiindawo zokulima umbona. Imvula eninzi ithetha isivuno esikhulu, esinciphisa amaxabiso.",
        "help_fuel_title": "Ixabiso lamafutha",
        "help_fuel_desc": "Ixabiso le-diesel nepetroli. Amafutha abiza kakhulu anyusa iindleko zokulima nokuhambisa umbona.",
        "help_settings_title": "Inkcazo yeesetingi zemodeli nganye",
        "help_settings_intro": "Imodeli nganye ineesetingi zayo. Ungazishiya kwi-default — sezilungisiwe.",
        "help_sarimax_settings_title": "Iisetingi ze-SARIMAX",
        "help_sarimax_p_desc": "I-AR order (p) — Zingaphi iintsuku ezidlulileyo imodeli ejonga kuzo xa iqikelela namhlanje.",
        "help_sarimax_d_desc": "Umahluko (d) — Kukangaphi uluhlu lwamaxabiso lutshintshwa ngaphambi kokuba imodeli ilubone. 1 yinto eqhelekileyo.",
        "help_sarimax_q_desc": "I-MA order (q) — Zingaphi iimpazamo zokuqikelela ezidlulileyo imodeli ezilungisayo.",
        "help_sarimax_n_desc": "Iintsuku zokuqikelela — Zingaphi iintsuku phambili imodeli eqikelelayo.",
        "help_xgb_settings_title": "Iisetingi ze-XGBoost",
        "help_xgb_horizon_desc": "Umda (iintsuku) — Zingaphi iintsuku phambili imodeli eqikelela. 1 ngomso, 5 iveki ezayo.",
        "help_xgb_predict_desc": "Qikelela — Nokuba imodeli iqikelela utshintsho (diff) okanye ixabiso elipheleleyo (level).",
        "help_xgb_plags_desc": "Ukulibaziseka kwexabiso — Zingaphi iintsuku zokutshintsha kwexabiso imodeli ezisebenzisayo.",
        "help_xgb_dlags_desc": "Ukulibaziseka kweempembelelo — Zingaphi iintsuku zempembelelo nganye imodeli ezisebenzisayo.",
        "help_xgb_iter_desc": "Uphando lokuphinda — Zingaphi iindibaniselwano i-tuner ezizamayo.",
        "help_s2s_settings_title": "Iisetingi ze-Seq2Seq",
        "help_s2s_enc_desc": "Ubude be-encoder — Zingaphi iintsuku zembali i-encoder ezifundayo ngaphambi kokuqikelela.",
        "help_s2s_pred_desc": "Iintsuku zokuqikelela — Zingaphi iintsuku zexesha elizayo i-decoder ezibhalayo.",
        "help_s2s_latent_desc": "Ubungakanani obufihlakeleyo — Ubungakanani bememori yangaphakathi yemodeli.",
        "help_s2s_epochs_desc": "Ubuninzi be-epoch — Inani eliphezulu lokudlula koqeqesho.",
        "help_rec_title": "Inkcazo yeFumana ingcebiso yam",
        "help_rec_intro": "Xa ucofa iqhosha, i-Kotulo iqalisa iimodeli zontathu, izilinganise, kwaye ikunike isigqibo esinye.",
        "help_rec_hold_title": "GCINA (luhlaza)",
        "help_rec_hold_desc": "Iimodeli zilindele ukuba ixabiso linyuke ngaphezulu kweendleko zokugcina ummbona, kunye nomda wokhuseleko. Ukulinda kuya kukunceda.",
        "help_rec_sell_title": "THENGISA (bomvu)",
        "help_rec_sell_desc": "Iimodeli aziboni kunyuka okunamandla. Ukuthengisa ngoku kunokukhuselekile. Le yindlela yefama ezincinci.",
        "help_rec_conf_title": "Umgca wokuzithemba",
        "help_rec_conf_desc": "Ibonisa ukuba zingaphi kwiimodeli ezintathu ezivotela ukunyuka kunye nokuhla.",
        "help_rec_votes_title": "Indlela imodeli nganye evote ngayo",
        "help_rec_votes_desc": "Itheyibhile ebonisa oko imodeli nganye iqikeleleyo, utshintsho, indlela, ubunzima, kunye nempazamo yovavanyo (MAE).",
        "help_rec_metrics_title": "Amanani amathathu angezantsi kwesigqibo",
        "help_rec_metrics_desc": "Ixabiso lanamhlanje lixabiso le-SAFEX langoku. Uqikelelo oluphakathi loko iimodeli ezilindele ukuba kube kuko. Utshintsho olulindelekileyo ngumahluko.",
        "help_rec_pdf_title": "Khuphela ingcebiso njenge-PDF",
        "help_rec_pdf_desc": "Gcina ingcebiso njenge-PDF yephepha elinye onokuyiprinta uyise kwi-co-op.",
    },

    # ============================================================= ISIZULU
    "zu": {
        "app_name": "Kotulo",
        "hero_sub": "Umhlahlandlela olula wokuthi uthengise nini ummbila wakho we-SAFEX.",
        "tab_home": "Ikhaya",
        "tab_forecast": "Isibikezelo (izinhlobo ezi-3)",
        "tab_recommend": "Ngenzeni?",
        "tab_help": "Thola Usizo",
        "language_label": "Ulimi",
        "home_explainer": (
            "Siyakwamukela ku-Kotulo, lapho ubuchwepheshe buhlangana nezolimo. Inhloso yethu "
            "ukuletha ama-algorithms asetshenziswa ukufunda ngomshini kubalimi bethu basekhaya, "
            "ukuze bazi ukuthi yisiphi isikhathi esifanele sokuthengisa — ngokusekelwe "
            "eminyakeni yedatha yokuhlola yangaphambilini. Abalimi abancane bavame ukulahlekelwa "
            "yithuba lokungena emakethe ngenxa yokungabi nalo ulwazi oluthile. Sifisa ukuba "
            "yisisombululo sale nkinga, sisebenzisa ezintathu kuma-algorithms amahle kakhulu "
            "okufunda ngomshini: i-SARIMAX, i-XGBoost ne-Seq2Seq."
        ),
        "choose_how": "Khetha ukuthi ufuna ukuyisebenzisa kanjani le app",
        "demo_card_title": "🧪 Umboniso — bheka ukuthi isebenza kanjani",
        "demo_card_body": "Zama i-app ngedatha yesampula yangempela. Azikho izifayela ezidingekayo.",
        "demo_btn": "▶️ Zama uMboniso",
        "real_card_title": "📊 Thola Izibikezelo Zentengo Yangempela",
        "real_card_body": "Layisha ifayela lakho lentengo yommbila we-SAFEX kanye nezinto zangaphandle bese uthola isincomo.",
        "real_btn": "📈 Sebenzisa iDatha Yami",
        "demo_mode_banner": "🧪 Imodi yomboniso — kusetshenziswa idatha yesampula",
        "advanced_settings": "Izilungiselelo ezithuthukile",
        "train_split_label": "Ukuhlukaniswa kokuqeqesha / kokuhlola",
        "train_split_help": "I-80% yomlando iqeqesha izinhlobo; i-20% yakamuva iyazihlola.",
        "province_label": "Isifundazwe (sezindleko zokugcina)",
        "holding_cost_label": "Izindleko zokugcina (ngenyanga, %)",
        "holding_cost_help": "Okokuzenzakalela kwisifundazwe sakho.",
        "external_drivers_label": "Izinto zangaphandle",
        "your_data_glance": "Idatha yakho ngokushesha",
        "business_days_metric": "Izinsuku zebhizinisi",
        "date_range_metric": "Ububanzi bezinsuku",
        "today_price": "Intengo yanamuhla",
        "province_metric": "Isifundazwe",
        "show_data": "Bonisa idatha",
        "how_to_use": "Indlela yokuyisebenzisa",
        "step_1": "Vula <b>{forecast_tab}</b> ukuze ubone ukuthi uhlamvu ngalunye luthini.",
        "step_2": "Vula <b>{recommend_tab}</b> bese ucindezela inkinobho enkulu.",
        "pick_option": "Khetha ukukhetha okungenhla ukuqala.",
        "upload_section_title": "📊 Layisha idatha yakho",
        "upload_explainer": "Udinga amafayela amabili e-CSV. Eyokuqala iqukethe izintengo zommbila we-SAFEX.",
        "upload_prices_label": "Layisha safex_prices.csv",
        "upload_factors_label": "Layisha sa_factors.csv",
        "dayfirst_label": "Izinsuku zami zingu-dd/mm/yyyy",
        "upload_both_info": "Layisha **womabili** amafayela e-CSV ngenhla ukuqhubeka.",
        "data_loaded_success": "✓ Idatha ilayishiwe: **{n} izinsuku zebhizinisi**",
        "forecast_hero_sub": "Izinhlobo ezintathu zibheka idatha ngezindlela ezahlukene. Cindezela u-Qala kunoma iyiphi.",
        "settings_expander": "Izilungiselelo",
        "run_sarimax_btn": "Qala i-SARIMAX",
        "run_xgboost_btn": "Qala i-XGBoost",
        "run_seq2seq_btn": "Qala i-Seq2Seq",
        "sarimax_p_label": "I-AR order (p)",
        "sarimax_d_label": "Umehluko (d)",
        "sarimax_q_label": "I-MA order (q)",
        "sarimax_n_label": "Izinsuku zokubikezela",
        "xgb_horizon_label": "Umkhawulo (izinsuku)",
        "xgb_predict_label": "Bikezela",
        "xgb_price_lags_label": "Ukubambezeleka kwentengo",
        "xgb_driver_lags_label": "Ukubambezeleka kwezinto",
        "xgb_iter_label": "Ucwaningo lokuphinda",
        "s2s_enc_label": "Ubude be-encoder",
        "s2s_pred_label": "Izinsuku zokubikezela",
        "s2s_latent_label": "Usayizi ocashile",
        "s2s_epochs_label": "Inani eliphezulu lama-epoch",
        "forecast_success_day": "Isibikezelo sosuku +{n}: **R {price:,.2f} / ton**",
        "forecast_success_range": "Isibikezelo sezinsuku ezingu-{n}: **R {price:,.2f} / ton**",
        "sarimax_explainer": "<b>Okwenzayo:</b> I-SARIMAX iyimodeli yezibalo yakudala.<br><b>Indlela yokuyifunda:</b> Uma i-MAE incane, iyasiza.",
        "xgb_explainer": "<b>Okwenzayo:</b> I-XGBoost iyimodeli yokufunda ngomshini.<br><b>Indlela yokuyifunda:</b> Ngokuvamile enamandla kakhulu.",
        "s2s_explainer": "<b>Okwenzayo:</b> I-Seq2Seq ifunda izinsuku ezingama-60 bese ibhala ezingu-5 ezilandelayo.<br><b>Indlela yokuyifunda:</b> Uma ibheka phambili, ayinembi.",
        "translate_explanation": "🌍 Humusha le ncazelo",
        "recommend_hero_sub": "Cindezela inkinobho futhi i-app izoqalisa zonke izinhlobo ezintathu, izilinganise, futhi ikunikeze isincomo esisodwa.",
        "horizon_label": "Bheka phambili izinsuku ezingaki?",
        "horizon_help": "Izinsuku ezinhlanu cishe iviki elilodwa.",
        "btn_recommend": "🌟 Thola isincomo sami",
        "rec_header": "Isincomo sethu",
        "decision_hold": "GCINA",
        "decision_sell": "THENGISA",
        "confidence": "Ukuqiniseka",
        "confidence_line": "{up} kwezingu-{total} izinhlobo zibikezela ukunyuka. ({down} zibikezela ukwehla.)",
        "reason_hold": "Izinhlobo zilindele ukuthi intengo ikhuphuke cishe nge-{pct:.1f}% ezinsukwini ezingu-{days} ezizayo.",
        "reason_sell_down": "Izinhlobo zilindele ukuthi intengo yehle cishe nge-{pct:.1f}% ezinsukwini ezingu-{days} ezizayo.",
        "reason_sell_weak": "Izinhlobo aziboni ukukhuphuka okunamandla ezinsukwini ezingu-{days} ezizayo.",
        "avg_forecast": "Isibikezelo esimaphakathi",
        "expected_change": "Ushintsho olulindelekile",
        "how_each_voted": "Indlela uhlamvu ngalunye oluvote ngayo",
        "table_model": "Uhlamvu",
        "table_forecast": "Isibikezelo (R/ton)",
        "table_change": "Ushintsho",
        "table_direction": "Indlela",
        "table_weight": "Isisindo",
        "table_test_mae": "Ukuhlolwa kwe-MAE (R/ton)",
        "direction_up": "📈 phezulu",
        "direction_down": "📉 phansi",
        "direction_flat": "➖ silingene",
        "what_means": "Kusho ukuthini lokhu?",
        "what_means_hold": "<b>{hold}</b> kusho ukuthi izinhlobo zilindele ukuthi intengo ikhuphuke ngokwanele ukumboza izindleko zokugcina.",
        "what_means_sell": "<b>{sell}</b> kusho ukuthi izinhlobo aziboni ukukhuphuka okunamandla. Ukuthengisa manje kungcono.",
        "not_advice": "Asikho isibikezelo esiqinisekile. Sebenzisa lokhu kanye nolwazi lwakho ngepulazi lakho.",
        "press_button": "Cindezela inkinobho engenhla ukuthola isincomo sakho.",
        "unlock_panel": "Khetha **uMboniso** noma **Izibikezelo zangempela** kuthebhu yeKhaya.",
        "running_models_status": "Kuqalwa izinhlobo ezintathu…",
        "fitting_sarimax": "Kulungiselelwa i-SARIMAX…",
        "fitting_xgb": "Kulungiselelwa i-XGBoost…",
        "training_s2s": "Kuqeqeshwa i-Seq2Seq (~1 min)…",
        "all_finished": "Zonke izinhlobo ezintathu ziqedile ✓",
        "skipped_warning": "{model} yeqiwe: {err}",
        "pdf_download": "📄 Landa isincomo njenge-PDF",
        "pdf_unavailable": "📄 Ukulanda i-PDF akutholakali — faka i-reportlab.",
        "sarimax_failed": "I-SARIMAX yehlulekile: {err}",
        "xgb_failed": "I-XGBoost yehlulekile: {err}",
        "s2s_failed": "I-Seq2Seq yehlulekile: {err}",
        "error_could_not_read": "Ayikwazanga ukufunda amafayela akho edatha.",
        "error_common_causes": "**Izimbangela ezivamile:**\n- Izinsuku aziqondakali.\n  - 2024-01-05: **susa** ibhokisi le-dd/mm/yyyy.\n  - 05/01/2024: **yikhonkcoze**.\n- Ifayela alinayo ikholomu ye-Date.",
        "footer": "🌽 Kotulo — ithuluzi elivulekile lokubikezela ummbila we-SAFEX. Akuyona iseluleko sezezimali.",

        "help_hero_sub": "Izincazelo zolimi olulula zawo wonke amagama, izilungiselelo, nemiphumela ku-app.",
        "help_intro": "Uma ungaqiniseki ukuthi okuthile kusho ukuthini, kufune lapha. Wonke amagama asetshenziswa yi-app achazwe ngezansi.",
        "help_models_title": "Izinhlobo zethu ezi-3",
        "help_models_intro": "I-Kotulo isebenzisa izinhlobo ezintathu zamakhompiyutha. Zonke zibheka idatha ngezindlela ezahlukene.",
        "help_sarimax_title": "SARIMAX",
        "help_sarimax_desc": "Imodeli yezibalo yakudala. Ifunda iphethini yezintengo ezedlule nendlela ezihamba ngayo.",
        "help_xgboost_title": "XGBoost",
        "help_xgboost_desc": "Imodeli yokufunda ngomshini eyakhiwe ngezihlahla. Ifunda imithetho efana nokuthi \"uma i-rand iba buthaka, intengo inyuka\".",
        "help_seq2seq_title": "Seq2Seq",
        "help_seq2seq_desc": "Inethiwekhi ye-neural efunda izinsuku ezingama-60 bese ibhala ezingu-5 ezilandelayo, usuku nosuku.",
        "help_advanced_title": "Incazelo Yezilungiselelo Ezithuthukile",
        "help_advanced_intro": "Lezi zilungiselelo zishintsha indlela izinhlobo eziqeqeshwa ngayo. Uma ungaqiniseki, shiya ku-default.",
        "help_train_split_title": "Ukuhlukaniswa kokuqeqesha / kokuhlola",
        "help_train_split_desc": "Mangaki amasiko izinhlobo ezifunda kuwo, uma kuqhathaniswa nokususa eceleni ukuhlola. I-80/20 yindlela ejwayelekile.",
        "help_province_title": "Isifundazwe",
        "help_province_desc": "Isifundazwe sakho sisetha izindleko zokugcina ummbila. IFree State, North West, Mpumalanga yizona zifundazwe eziphambili zokulima ummbila.",
        "help_holding_cost_title": "Izindleko zokugcina (ngenyanga, %)",
        "help_holding_cost_desc": "Kubiza malini ukugcina ithani elilodwa lommbila inyanga eyodwa. Kuhlanganisa izindleko zesilo kanye nenzalo.",
        "help_drivers_title": "Izinto zangaphandle",
        "help_drivers_desc": "Izimpawu ezengeziwe ezisetshenziswa izinhlobo ukwenza izibikezelo. Ungavula noma uvale ngayinye.",
        "help_driver_explain_title": "Incazelo yezinto zangaphandle",
        "help_driver_explain_intro": "Izinto zangaphandle yizona ezithinta intengo yommbila. I-Kotulo isebenzisa ezintathu.",
        "help_zar_title": "ZAR/USD (izinga lokushintshisana)",
        "help_zar_desc": "Mangaki ama-rand okuthenga idola eyodwa yaseMelika. I-rand ebuthakathaka inyusa izintengo zommbila wasekhaya.",
        "help_rain_title": "Imvula (mm)",
        "help_rain_desc": "Imvula ezindaweni zokulima ummbila. Imvula eningi isho isivuno esikhulu, esinciphisa izintengo.",
        "help_fuel_title": "Intengo kaphethiloli",
        "help_fuel_desc": "Intengo kadizili nophethiloli. Uphethiloli obiza kakhulu unyusa izindleko zokulima nokuhambisa.",
        "help_settings_title": "Incazelo yezilungiselelo zohlamvu ngalunye",
        "help_settings_intro": "Uhlamvu ngalunye lunezilungiselelo zalo. Ungazishiya ku-default — sezilungisiwe.",
        "help_sarimax_settings_title": "Izilungiselelo ze-SARIMAX",
        "help_sarimax_p_desc": "I-AR order (p) — Zingaki izinsuku ezedlule uhlamvu olubheka kuzo uma lubikezela namuhla.",
        "help_sarimax_d_desc": "Umehluko (d) — Kukangaki uchungechunge lwentengo lushintshwa ngaphambi kokuba uhlamvu lubone. 1 yinto ejwayelekile.",
        "help_sarimax_q_desc": "I-MA order (q) — Zingaki amaphutha okubikezela edlule uhlamvu olulungisayo.",
        "help_sarimax_n_desc": "Izinsuku zokubikezela — Zingaki izinsuku phambili uhlamvu olubikezelayo.",
        "help_xgb_settings_title": "Izilungiselelo ze-XGBoost",
        "help_xgb_horizon_desc": "Umkhawulo (izinsuku) — Zingaki izinsuku phambili. 1 kusasa, 5 iviki elizayo.",
        "help_xgb_predict_desc": "Bikezela — Noma uhlamvu lubikezela ushintsho (diff) noma intengo ephelele (level).",
        "help_xgb_plags_desc": "Ukubambezeleka kwentengo — Zingaki izinsuku zokushintsha kwentengo uhlamvu oluzisebenzisayo.",
        "help_xgb_dlags_desc": "Ukubambezeleka kwezinto — Zingaki izinsuku zento ngayinye yangaphandle uhlamvu oluzisebenzisayo.",
        "help_xgb_iter_desc": "Ucwaningo lokuphinda — Zingaki izinhlanganisela i-tuner ezizamayo.",
        "help_s2s_settings_title": "Izilungiselelo ze-Seq2Seq",
        "help_s2s_enc_desc": "Ubude be-encoder — Zingaki izinsuku zomlando i-encoder ezifundayo ngaphambi kokubikezela.",
        "help_s2s_pred_desc": "Izinsuku zokubikezela — Zingaki izinsuku zesikhathi esizayo i-decoder ezibhalayo.",
        "help_s2s_latent_desc": "Usayizi ocashile — Usayizi wememori yangaphakathi yohlamvu.",
        "help_s2s_epochs_desc": "Inani eliphezulu lama-epoch — Inani eliphezulu lokuphasa kokuqeqesha.",
        "help_rec_title": "Incazelo ye-Thola isincomo sami",
        "help_rec_intro": "Uma ucindezela inkinobho, i-Kotulo iqalisa izinhlobo ezintathu, izilinganise, futhi ikunike isinqumo esisodwa.",
        "help_rec_hold_title": "GCINA (luhlaza)",
        "help_rec_hold_desc": "Izinhlobo zilindele ukuthi intengo ikhuphuke ngaphezu kwezindleko zokugcina ummbila. Ukulinda kuzokuzuza.",
        "help_rec_sell_title": "THENGISA (bomvu)",
        "help_rec_sell_desc": "Izinhlobo aziboni ukukhuphuka okunamandla. Ukuthengisa manje kungcono.",
        "help_rec_conf_title": "Umugqa wokuqiniseka",
        "help_rec_conf_desc": "Ibonisa ukuthi zingaki kwezingu-3 izinhlobo ezivotela ukunyuka nokwehla.",
        "help_rec_votes_title": "Indlela uhlamvu ngalunye oluvote ngayo",
        "help_rec_votes_desc": "Ithebula elibonisa isibikezelo sohlamvu ngalunye, ushintsho, indlela, isisindo, kanye nephutha lokuhlola (MAE).",
        "help_rec_metrics_title": "Izinombolo ezintathu ngaphansi kwesinqumo",
        "help_rec_metrics_desc": "Intengo yanamuhla iyintengo ye-SAFEX yamanje. Isibikezelo esimaphakathi yilokho izinhlobo ezilindele ukuthi kube yikho. Ushintsho olulindelekile ngumehluko.",
        "help_rec_pdf_title": "Landa isincomo njenge-PDF",
        "help_rec_pdf_desc": "Gcina isincomo njenge-PDF yekhasi elilodwa ongaliphrinta uyise e-co-op.",
    },

    # ============================================================= SESOTHO
    "st": {
        "app_name": "Kotulo",
        "hero_sub": "Tataiso e bonolo ea hore na u rekise neng poone ea hau ea SAFEX.",
        "tab_home": "Lapeng",
        "tab_forecast": "Ponelopele (mefuta e 3)",
        "tab_recommend": "Ke etse eng?",
        "tab_help": "Fumana Thuso",
        "language_label": "Puo",
        "home_explainer": (
            "Rea u amohela ho Kotulo, moo theknoloji e kopanang le temo. Sepheo sa rona ke "
            "ho tlisa li-algorithms tse tsamaisoang ke ho ithuta ka mochini ho lihoai tsa "
            "rona tsa lehae, e le hore li tsebe hore na ke nako efe e nepahetseng ea ho "
            "rekisa — ho ipapisitsoe le lilemo tsa data ea tlhahlobo ea nakong e fetileng. "
            "Lihoai tse nyenyane hangata li hloloheloa monyetla oa ho kena 'marakeng ka "
            "lebaka la ho hloka tlhahisoleseding e itseng. Re lakatsa ho ba tharollo ea "
            "bothata bona, re sebelisa tse tharo tsa li-algorithms tse ntle ka ho fetisisa "
            "tsa ho ithuta ka mochini: SARIMAX, XGBoost le Seq2Seq."
        ),
        "choose_how": "Khetha hore na u batla ho sebelisa app ena joang",
        "demo_card_title": "🧪 Pontšo — sheba hore na e sebetsa joang",
        "demo_card_body": "Leka app ka data ea mohlala. Ha ho na lifaele tse hlokahalang.",
        "demo_btn": "▶️ Leka Pontšo",
        "real_card_title": "📊 Fumana Ponelopele ea Theko ea Nnete",
        "real_card_body": "Kenya faele ea hau ea theko ea poone ea SAFEX le mabaka a kantle.",
        "real_btn": "📈 Sebelisa Data ea Ka",
        "demo_mode_banner": "🧪 Mokhoa oa pontšo — ho sebelisoa data ea mohlala",
        "advanced_settings": "Litlhophiso tse tsoetseng pele",
        "train_split_label": "Karohano ea koetliso / tlhahlobo",
        "train_split_help": "80% ea nalane e koetlisa mefuta; 20% ea morao-rao e ea e hlahloba.",
        "province_label": "Profense (bakeng sa litšenyehelo tsa ho boloka)",
        "holding_cost_label": "Litšenyehelo tsa ho boloka (ka khoeli, %)",
        "holding_cost_help": "E itšetlehile ka khakanyo ea profense ea hau.",
        "external_drivers_label": "Mabaka a kantle",
        "your_data_glance": "Data ea hau ka leihlo le le leng",
        "business_days_metric": "Matsatsi a khoebo",
        "date_range_metric": "Nako ea matsatsi",
        "today_price": "Theko ea kajeno",
        "province_metric": "Profense",
        "show_data": "Bontša data",
        "how_to_use": "Mokhoa oa ho e sebelisa",
        "step_1": "Bula <b>{forecast_tab}</b> ho bona seo mofuta ka mong o se bolelang.",
        "step_2": "Bula <b>{recommend_tab}</b> 'me u tobe konopo e kholo.",
        "pick_option": "Khetha khetho e ka holimo ho qala.",
        "upload_section_title": "📊 Kenya data ea hau",
        "upload_explainer": "U hloka lifaele tse peli tsa CSV. Ea pele e na le litheko tsa poone ea SAFEX.",
        "upload_prices_label": "Kenya safex_prices.csv",
        "upload_factors_label": "Kenya sa_factors.csv",
        "dayfirst_label": "Matsatsi a ka ke dd/mm/yyyy",
        "upload_both_info": "Kenya **lifaele ka bobeli** tsa CSV ka holimo ho tsoela pele.",
        "data_loaded_success": "✓ Data e kentsoe: **{n} matsatsi a khoebo**",
        "forecast_hero_sub": "Mefuta e meraro e sheba data ka litsela tse tharo tse fapaneng. Tobetsa Qala ho efe kapa efe.",
        "settings_expander": "Litlhophiso",
        "run_sarimax_btn": "Qala SARIMAX",
        "run_xgboost_btn": "Qala XGBoost",
        "run_seq2seq_btn": "Qala Seq2Seq",
        "sarimax_p_label": "AR order (p)",
        "sarimax_d_label": "Phapang (d)",
        "sarimax_q_label": "MA order (q)",
        "sarimax_n_label": "Matsatsi a ponelopele",
        "xgb_horizon_label": "Moeli (matsatsi)",
        "xgb_predict_label": "Bolela esale pele",
        "xgb_price_lags_label": "Ho lieha ha theko",
        "xgb_driver_lags_label": "Ho lieha ha mabaka",
        "xgb_iter_label": "Lipatlisiso tse pheta-phetoang",
        "s2s_enc_label": "Bolelele ba encoder",
        "s2s_pred_label": "Matsatsi a ho bolela",
        "s2s_latent_label": "Boholo bo patiloeng",
        "s2s_epochs_label": "Li-epoch tse phahameng",
        "forecast_success_day": "Ponelopele ea letsatsi +{n}: **R {price:,.2f} / ton**",
        "forecast_success_range": "Ponelopele ea matsatsi a {n}: **R {price:,.2f} / ton**",
        "sarimax_explainer": "<b>Seo e se etsang:</b> SARIMAX ke mofuta oa khale oa lipalo.<br><b>Mokhoa oa ho e bala:</b> Haeba MAE e nyane, e na le thuso.",
        "xgb_explainer": "<b>Seo e se etsang:</b> XGBoost ke mofuta oa ho ithuta ka mochini.<br><b>Mokhoa oa ho e bala:</b> Hangata e matla ka ho fetisisa.",
        "s2s_explainer": "<b>Seo e se etsang:</b> Seq2Seq e bala matsatsi a 60 a ho qetela 'me e ngola matsatsi a 5 a latelang.<br><b>Mokhoa oa ho e bala:</b> Ha e sheba hole, ha e nepahale.",
        "translate_explanation": "🌍 Fetolela tlhaloso ena",
        "recommend_hero_sub": "Tobetsa konopo 'me app e tla qala mefuta e meraro, e e lekanye, 'me e u fe keletso e le 'ngoe.",
        "horizon_label": "Sheba pele matsatsi a makae a khoebo?",
        "horizon_help": "Matsatsi a mahlano e batla e le beke e le 'ngoe.",
        "btn_recommend": "🌟 Fumana keletso ea ka",
        "rec_header": "Keletso ea rona",
        "decision_hold": "BOLOKA",
        "decision_sell": "REKISA",
        "confidence": "Tšepo",
        "confidence_line": "{up} ho {total} mefuta e bolela ho nyoloha. ({down} e bolela ho theoha.)",
        "reason_hold": "Mefuta e lebelletse hore theko e nyolohe ka {pct:.1f}% matsatsing a {days} a tlang.",
        "reason_sell_down": "Mefuta e lebelletse hore theko e theohe ka {pct:.1f}% matsatsing a {days} a tlang.",
        "reason_sell_weak": "Mefuta ha e bone ho nyoloha ho matla matsatsing a {days} a tlang.",
        "avg_forecast": "Ponelopele e mahareng",
        "expected_change": "Phetoho e lebelletsoeng",
        "how_each_voted": "Kamoo mofuta ka mong o khethileng ka teng",
        "table_model": "Mofuta",
        "table_forecast": "Ponelopele (R/ton)",
        "table_change": "Phetoho",
        "table_direction": "Tsela",
        "table_weight": "Boima",
        "table_test_mae": "Teko ea MAE (R/ton)",
        "direction_up": "📈 holimo",
        "direction_down": "📉 tlaase",
        "direction_flat": "➖ otlolohile",
        "what_means": "See se bolela eng?",
        "what_means_hold": "<b>{hold}</b> e bolela hore mefuta e lebelletse hore theko e nyolohe ho lekana ho koahela litšenyehelo tsa ho boloka.",
        "what_means_sell": "<b>{sell}</b> e bolela hore mefuta ha e bone ho nyoloha ho matla. Ho rekisa hona joale ho ka ba molemo.",
        "not_advice": "Ha ho ponelopele e tiileng. E sebelise hammoho le tsebo ea hau ea polasi.",
        "press_button": "Tobetsa konopo e ka holimo ho fumana keletso ea hau.",
        "unlock_panel": "Khetha **Pontšo** kapa **Ponelopele ea nnete** ho thebe ea Lapeng.",
        "running_models_status": "Ho qaloa mefuta e meraro…",
        "fitting_sarimax": "Ho lokisoa SARIMAX…",
        "fitting_xgb": "Ho lokisoa XGBoost…",
        "training_s2s": "Ho koetlisoa Seq2Seq (~1 min)…",
        "all_finished": "Mefuta eohle e meraro e qetile ✓",
        "skipped_warning": "{model} e tlotsoe: {err}",
        "pdf_download": "📄 Khoasolla keletso e le PDF",
        "pdf_unavailable": "📄 Ho khoasolla PDF ha ho fumanehe — kenya reportlab.",
        "sarimax_failed": "SARIMAX e hlolehile: {err}",
        "xgb_failed": "XGBoost e hlolehile: {err}",
        "s2s_failed": "Seq2Seq e hlolehile: {err}",
        "error_could_not_read": "Ha e khone ho bala lifaele tsa hau tsa data.",
        "error_common_causes": "**Mabaka a tloaelehileng:**\n- Matsatsi ha a hlalosehe.\n  - 2024-01-05: **tlosa** lebokose la dd/mm/yyyy.\n  - 05/01/2024: **e tšoaea**.\n- Faele ha e na kholomo ea Date.",
        "footer": "🌽 Kotulo — sesebelisoa se bulehileng sa ho bolela esale pele poone ea SAFEX. Ha se keletso ea lichelete.",

        "help_hero_sub": "Litlhaloso tse bonolo tsa lentsoe le leng le le leng, tlhophiso le sephetho ka app.",
        "help_intro": "Haeba u sa tsebe hore na ntho e itseng e bolela eng, e batle mona. Mantsoe 'ohle a sebelisoang ke app a hlalositsoe ka tlase.",
        "help_models_title": "Mefuta ea rona e 3",
        "help_models_intro": "Kotulo e sebelisa mefuta e meraro e fapaneng ea likhomphutha. E 'ngoe le e 'ngoe e sheba data ka tsela e fapaneng.",
        "help_sarimax_title": "SARIMAX",
        "help_sarimax_desc": "Mofuta oa khale oa lipalo. O ithuta mohlala oa litheko tse fetileng le kamoo li tsamaeang kateng ha rand, pula kapa peterole li fetoha.",
        "help_xgboost_title": "XGBoost",
        "help_xgboost_desc": "Mofuta oa ho ithuta ka mochini o hahiloeng ka lifate tsa liqeto. O ithuta melao e kang \"haeba rand e fokola, theko e nyoloha\".",
        "help_seq2seq_title": "Seq2Seq",
        "help_seq2seq_desc": "Marang-rang a neural a bala matsatsi a 60 a ho qetela 'me a ngole matsatsi a 5 a latelang, letsatsi ka leng.",
        "help_advanced_title": "Tlhaloso ea Litlhophiso tse Tsoetseng Pele",
        "help_advanced_intro": "Litlhophiso tsena li fetola kamoo mefuta e koetlisoang. Ha u sa tsebe, li siee ho default.",
        "help_train_split_title": "Karohano ea koetliso / tlhahlobo",
        "help_train_split_desc": "Bokae ba nalane mefuta e ithutang ho eona, khahlanong le bokae bo bolokiloeng ho e hlahloba. 80/20 ke mokhoa o tloaelehileng.",
        "help_province_title": "Profense",
        "help_province_desc": "Profense ea hau e beha litšenyehelo tsa ho boloka poone. Free State, North West, Mpumalanga ke libaka tse ka sehloohong tse hlahisang poone.",
        "help_holding_cost_title": "Litšenyehelo tsa ho boloka (ka khoeli, %)",
        "help_holding_cost_desc": "Ho bitsa bokae ho boloka tonne e le 'ngoe ea poone khoeli e le 'ngoe. Ho kenyelletsa litefiso tsa silo le tsoala ea chelete e tšoeroeng sejalo.",
        "help_drivers_title": "Mabaka a kantle",
        "help_drivers_desc": "Matšoao a eketsehileng ao mefuta e a sebelisang ho etsa ponelopele. U ka bula kapa u koala e 'ngoe le e 'ngoe.",
        "help_driver_explain_title": "Tlhaloso ea mabaka",
        "help_driver_explain_intro": "Mabaka a kantle ke lintho tse amang theko ea poone. Kotulo e sebelisa tse tharo.",
        "help_zar_title": "ZAR/USD (sekhahla sa phapanyetsano)",
        "help_zar_desc": "Ke li-rand tse kae ho reka dolara e le 'ngoe ea Amerika. Rand e fokolang e nyolla litheko tsa lehae.",
        "help_rain_title": "Pula (mm)",
        "help_rain_desc": "Pula libakeng tse hlahisang poone. Pula e ngata e bolela sejalo se seholo, se theolang litheko.",
        "help_fuel_title": "Theko ea peterole",
        "help_fuel_desc": "Theko ea disele le peterole. Peterole e theko e phahameng e nyolla litšenyehelo tsa ho lema le ho tsamaisa.",
        "help_settings_title": "Tlhaloso ea litlhophiso tsa mofuta ka mong",
        "help_settings_intro": "Mofuta ka mong o na le litlhophiso tsa oona. U ka li siea kaofela ho default — li se li lokisitsoe.",
        "help_sarimax_settings_title": "Litlhophiso tsa SARIMAX",
        "help_sarimax_p_desc": "AR order (p) — Matsatsi a makae a fetileng mofuta o shebang ho ona ha o bolela kajeno.",
        "help_sarimax_d_desc": "Phapang (d) — Hangata letoto la litheko le fetoloa hangata hakae pele mofuta o le bona. 1 ke e tloaelehileng.",
        "help_sarimax_q_desc": "MA order (q) — Liphupho tse kae tsa ponelopele tse fetileng tseo mofuta o li lokisang.",
        "help_sarimax_n_desc": "Matsatsi a ponelopele — Matsatsi a makae pele mofuta o bolela.",
        "help_xgb_settings_title": "Litlhophiso tsa XGBoost",
        "help_xgb_horizon_desc": "Moeli (matsatsi) — Matsatsi a makae pele. 1 hosane, 5 beke e tlang.",
        "help_xgb_predict_desc": "Bolela esale pele — Hore na mofuta o bolela phetoho (diff) kapa theko e tala (level).",
        "help_xgb_plags_desc": "Ho lieha ha theko — Matsatsi a makae a phetoho ea theko eo mofuta o e sebelisang.",
        "help_xgb_dlags_desc": "Ho lieha ha mabaka — Matsatsi a makae a lebaka le leng le le leng la kantle eo mofuta o le sebelisang.",
        "help_xgb_iter_desc": "Lipatlisiso tse pheta-phetoang — Likopano tse kae tseo tuner e li lekang.",
        "help_s2s_settings_title": "Litlhophiso tsa Seq2Seq",
        "help_s2s_enc_desc": "Bolelele ba encoder — Matsatsi a makae a nalane eo encoder e e balang pele e bolela.",
        "help_s2s_pred_desc": "Matsatsi a ho bolela — Matsatsi a makae a kamoso eo decoder e a ngolang.",
        "help_s2s_latent_desc": "Boholo bo patiloeng — Boholo ba memori ea ka hare ea mofuta.",
        "help_s2s_epochs_desc": "Li-epoch tse phahameng — Palo e phahameng ea liphase tsa koetliso.",
        "help_rec_title": "Tlhaloso ea Fumana keletso ea ka",
        "help_rec_intro": "Ha u tobetsa konopo, Kotulo e qala mefuta e meraro, e e lekanye, 'me e u fe qeto e le 'ngoe.",
        "help_rec_hold_title": "BOLOKA (tala)",
        "help_rec_hold_desc": "Mefuta e lebelletse hore theko e nyolohe ho feta litšenyehelo tsa ho boloka poone. Ho letela ho tla u tsoela molemo.",
        "help_rec_sell_title": "REKISA (khubelu)",
        "help_rec_sell_desc": "Mefuta ha e bone ho nyoloha ho matla. Ho rekisa hona joale ho ka ba molemo.",
        "help_rec_conf_title": "Mohala oa tšepo",
        "help_rec_conf_desc": "O bontša hore na ke mefuta e makae ho e meraro e khethileng ho nyoloha le ho theoha.",
        "help_rec_votes_title": "Kamoo mofuta ka mong o khethileng ka teng",
        "help_rec_votes_desc": "Tafole e bontšang seo mofuta ka mong o se boletseng, phetoho, tsela, boima, le phoso ea teko (MAE).",
        "help_rec_metrics_title": "Linomoro tse tharo ka tlase ho qeto",
        "help_rec_metrics_desc": "Theko ea kajeno ke theko ea SAFEX ea hona joale. Ponelopele e mahareng ke seo mefuta e lebelletse hore e be sona. Phetoho e lebelletsoeng ke phapang.",
        "help_rec_pdf_title": "Khoasolla keletso e le PDF",
        "help_rec_pdf_desc": "Boloka keletso e le PDF ea leqephe le le leng leo u ka le hatisang kapa ua le isa co-op.",
    },
}


# ====================================================== provincial holding ===
PROVINCE_HOLDING_COST = {
    "Free State":     1.2,
    "North West":     1.3,
    "Mpumalanga":     1.4,
}

MODEL_NAMES = ["SARIMAX", "XGBoost", "Seq2Seq"]


# ============================================================== data helpers ==
def _get_lang():
    return st.session_state.get("lang_code", "en")


def t(key, **kw):
    lang = _get_lang()
    txt = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))
    return txt.format(**kw) if kw else txt


def translate_link(english_text):
    lang = _get_lang()
    return (
        f"https://translate.google.com/?sl=en&tl={lang}"
        f"&text={quote(english_text)}&op=translate"
    )


@st.cache_data(show_spinner=False)
def load_real_sample_data():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-01", periods=600)

    base = np.linspace(3100, 4400, len(dates))
    noise = rng.normal(0, 40, len(dates)).cumsum() * 0.1
    price_series = np.round(base + noise, 2)
    prices_df = pd.DataFrame({"Date": dates, "Close": price_series})

    zar  = 15.0 + rng.normal(0, 0.10, len(dates)).cumsum() * 0.05
    rain = rng.gamma(2, 5, len(dates))
    fuel = 20.0 + rng.normal(0, 0.05, len(dates)).cumsum() * 0.08
    factors_df = pd.DataFrame({
        "Date": dates,
        "ZAR_USD": np.round(zar, 4),
        "Rainfall_mm": np.round(rain, 2),
        "Fuel_Price": np.round(fuel, 2),
    })
    return prices_df.to_csv(index=False).encode(), factors_df.to_csv(index=False).encode()


@st.cache_data(show_spinner=False)
def load_data(prices_bytes, exog_bytes, dayfirst):
    def _read(bytes_, label):
        df = pd.read_csv(io.BytesIO(bytes_))
        if "Date" not in df.columns:
            raise ValueError(
                f"{label}: no 'Date' column found. "
                f"Columns present: {list(df.columns)}"
            )
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=dayfirst, errors="coerce")
        n_bad = df["Date"].isna().sum()
        if n_bad:
            raise ValueError(
                f"{label}: {n_bad} row(s) could not be parsed as dates. "
                f"Try toggling the 'dd/mm/yyyy' checkbox. "
                f"First unparsed value: {df.loc[df['Date'].isna(), 'Date'].iloc[0]!r}"
            )
        df = df.set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    p = _read(prices_bytes, "safex_prices.csv")
    e = _read(exog_bytes, "sa_factors.csv")

    if not isinstance(e.index, pd.DatetimeIndex):
        raise TypeError("The drivers file's index is not a DatetimeIndex.")

    e = e.resample("D").last().ffill()
    data = p.join(e, how="inner")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("After the join, the index is not a DatetimeIndex.")
    data = data.resample("B").ffill()
    return data.dropna(how="all")


def score(y_true, y_pred):
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE (R/ton)": float(mean_absolute_error(a, b)),
        "RMSE (R/ton)": float(np.sqrt(mean_squared_error(a, b))),
        "MAPE %": float(np.mean(np.abs((a - b) / a)) * 100),
    }


# ============================================================ model runners ==
def run_sarimax(data, y, X, exog_cols, train_size, n_ahead,
                order=(1, 1, 1), scale=False):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    N = len(data)
    Xs = X
    if scale:
        mu, sd = X.iloc[:train_size].mean(), X.iloc[:train_size].std().replace(0, 1)
        Xs = (X - mu) / sd

    y_train = y.iloc[:train_size]
    Xtr = Xs.iloc[:train_size]
    fit = SARIMAX(y_train, exog=Xtr, order=order).fit(disp=False)

    Xte = Xs.iloc[train_size:]
    multi = pd.Series(np.asarray(fit.predict(start=train_size, end=N - 1, exog=Xte)),
                      index=y.index[train_size:])
    test_actual = y.iloc[train_size:]
    test_metrics = score(test_actual, multi)

    full = SARIMAX(y, exog=Xs, order=order).fit(disp=False)
    idx = pd.bdate_range(y.index[-1] + pd.offsets.BDay(1), periods=n_ahead)
    fut_exog = pd.DataFrame({c: Xs[c].iloc[-1] for c in exog_cols}, index=idx)
    fc = full.get_forecast(steps=n_ahead, exog=fut_exog).summary_frame(alpha=0.05)
    fc.index = idx

    return {
        "name": "SARIMAX",
        "current_price": float(y.iloc[-1]),
        "forecast_price": float(fc["mean"].iloc[-1]),
        "forecast_series": fc["mean"],
        "test_actual": test_actual, "test_pred": multi, "test_metrics": test_metrics,
    }


def run_xgboost(data, y, X, exog_cols, train_size, n_ahead,
                horizon=1, pred_mode="diff", lp=10, le=5, n_iter=30, fast=False):
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    def build_feats(dt, h):
        close = dt["Close"]; d1 = close.diff()
        f = {f"Close (t-{h})": close.shift(h)}
        for k in range(h, h + lp): f[f"dClose (t-{k})"] = d1.shift(k)
        for w in (5, 20):
            f[f"Close_minus_MA{w}"] = close.shift(h) - close.shift(h).rolling(w).mean()
            f[f"Volatility{w}"] = d1.shift(h).rolling(w).std()
        if {"High", "Low"}.issubset(dt.columns):
            for k in range(h, h + le): f[f"max-min (t-{k})"] = (dt["High"] - dt["Low"]).shift(k)
        for col in exog_cols:
            for k in range(h, h + le): f[f"{col} (t-{k})"] = dt[col].shift(k)
        f["Month"] = pd.Series(dt.index.month, index=dt.index)
        f["WeekOfYear"] = pd.Series(dt.index.isocalendar().week.astype(int).to_numpy(), index=dt.index)
        f["DayOfWeek"] = pd.Series(dt.index.dayofweek, index=dt.index)
        base = close.shift(h)
        target = (close - base) if pred_mode == "diff" else close
        return pd.DataFrame(f, index=dt.index), target, base

    Xf, yf, base = build_feats(data, horizon)
    ok = Xf.notna().all(axis=1) & yf.notna()
    Xf, yf, base = Xf[ok], yf[ok], base[ok]
    tr = Xf.index < y.index[train_size]

    if fast:
        best = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                objective="reg:squarederror", random_state=123)
        best.fit(Xf[tr], yf[tr])
    else:
        dist = {"learning_rate": [0.1, 0.05, 0.01],
                "n_estimators": list(range(50, 500, 25)),
                "max_depth": list(range(2, 11)),
                "min_child_weight": list(range(1, 6)),
                "gamma": [i / 10 for i in range(5)],
                "subsample": [0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
                "reg_alpha": [1e-5, 1e-2, 0.1, 1, 100]}
        search = RandomizedSearchCV(
            xgb.XGBRegressor(objective="reg:squarederror", n_jobs=1, random_state=123),
            dist, n_iter=int(n_iter), scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(3), n_jobs=-1, random_state=45,
        ).fit(Xf[tr], yf[tr])
        best = search.best_estimator_

    raw = best.predict(Xf[~tr])
    test_pred = pd.Series(raw + base[~tr].to_numpy() if pred_mode == "diff" else raw,
                          index=Xf.index[~tr])
    test_actual = y.loc[test_pred.index]
    test_metrics = score(test_actual, test_pred)

    ext = data.reindex(data.index.append(
        pd.bdate_range(data.index[-1] + pd.offsets.BDay(1), periods=horizon)))
    Xe, _, be = build_feats(ext, horizon)
    raw_next = float(best.predict(Xe.iloc[[-1]][list(Xf.columns)])[0])
    next_price = be.iloc[-1] + raw_next if pred_mode == "diff" else raw_next

    return {
        "name": "XGBoost",
        "current_price": float(y.iloc[-1]),
        "forecast_price": float(next_price),
        "forecast_series": pd.Series([next_price], index=ext.index[-1:]),
        "test_actual": test_actual, "test_pred": test_pred, "test_metrics": test_metrics,
    }


def run_seq2seq(data, y, X, exog_cols, train_size, n_ahead,
                enc_len=60, pred_steps=5, latent=32, epochs=50, batch=32, fast=False):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler

    tf.keras.utils.set_random_seed(42)
    N = len(data)
    lc = np.log1p(y.to_numpy(float))
    ex = StandardScaler().fit(X.iloc[:train_size]).transform(X)
    nf = 1 + len(exog_cols)

    def enc(t):
        e = lc[t - enc_len:t]; mu = e.mean()
        return np.column_stack([e - mu, ex[t - enc_len:t]]), mu, e[-1] - mu

    def build(origins):
        a, b, c, m = [], [], [], []
        for t in origins:
            f, mu, last = enc(t); tg = lc[t:t + pred_steps] - mu
            a.append(f); c.append(tg)
            b.append(np.concatenate([[last], tg[:-1]])); m.append(mu)
        return np.array(a), np.array(b)[:, :, None], np.array(c)[:, :, None], np.array(m)

    tro = range(enc_len, train_size - pred_steps + 1)
    teo = range(train_size, N - pred_steps + 1)
    etr, dtr, ytr, _ = build(tro)
    ete, _, yte, mte = build(teo)

    ei = Input(shape=(None, nf))
    _, h, c0 = LSTM(latent, dropout=0.2, return_state=True)(ei)
    di = Input(shape=(None, 1))
    dl = LSTM(latent, dropout=0.2, return_sequences=True, return_state=True)
    do, _, _ = dl(di, initial_state=[h, c0])
    dd = Dense(1)
    model = Model([ei, di], dd(do))
    model.compile(optimizer=Adam(1e-3), loss="mean_absolute_error")
    model.fit([etr, dtr], ytr, batch_size=int(batch), epochs=int(epochs),
              validation_split=0.1, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)])

    em = Model(ei, [h, c0])
    sh, sc = Input(shape=(latent,)), Input(shape=(latent,))
    si = Input(shape=(None, 1))
    so, nh, nc = dl(si, initial_state=[sh, sc])
    stepm = Model([si, sh, sc], [dd(so), nh, nc])

    def decode(x):
        s = em.predict(x, verbose=0)
        tgt = x[:, -1, 0].reshape(-1, 1, 1)
        out = np.zeros((len(x), pred_steps))
        for i in range(pred_steps):
            o, a_, b_ = stepm.predict([tgt] + list(s), verbose=0)
            out[:, i] = o[:, 0, 0]; tgt = o; s = [a_, b_]
        return out

    pp = np.expm1(decode(ete) + mte[:, None])
    ap = np.expm1(yte[:, :, 0] + mte[:, None])
    test_pred = pd.Series(pp[:, -1], index=y.index[[t + pred_steps - 1 for t in teo]])
    test_actual = y.loc[test_pred.index]
    test_metrics = score(test_actual, test_pred)

    fi, mu, _ = enc(N)
    fc = np.expm1(decode(fi[None])[0] + mu)
    idx = pd.bdate_range(y.index[-1] + pd.offsets.BDay(1), periods=pred_steps)

    return {
        "name": "Seq2Seq",
        "current_price": float(y.iloc[-1]),
        "forecast_price": float(fc[-1]),
        "forecast_series": pd.Series(fc, index=idx),
        "test_actual": test_actual, "test_pred": test_pred, "test_metrics": test_metrics,
    }


# ============================================================== recommendation =
def weighted_average_forecast(results):
    weights = {}
    for name, r in results.items():
        mae = r["test_metrics"]["MAE (R/ton)"] if r.get("test_metrics") else None
        weights[name] = 1.0 / max(mae, 1.0) if mae else 1.0
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    weights = {k: min(v, 0.45) for k, v in weights.items()}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    avg = sum(r["forecast_price"] * weights[name] for name, r in results.items())
    return float(avg), weights


def make_recommendation(results, current_price, horizon_days, hold_cost_pct):
    avg_price, weights = weighted_average_forecast(results)
    change_pct = (avg_price - current_price) / current_price * 100

    directions = [np.sign(r["forecast_price"] - current_price) for r in results.values()]
    n_up = sum(1 for d in directions if d > 0)
    n_total = len(directions)
    up_agreement = n_up / n_total

    SAFETY_MARGIN = 1.33333
    AGREEMENT_REQ = 0.6

    if change_pct > (hold_cost_pct + SAFETY_MARGIN) and up_agreement >= AGREEMENT_REQ:
        decision = "hold"
        reason = t("reason_hold", pct=change_pct, days=horizon_days)
    elif change_pct < -1.0:
        decision = "sell"
        reason = t("reason_sell_down", pct=abs(change_pct), days=horizon_days)
    else:
        decision = "sell"
        reason = t("reason_sell_weak", days=horizon_days)

    return {
        "decision": decision, "reason": reason,
        "avg_price": avg_price, "change_pct": change_pct,
        "up_agreement": up_agreement, "n_up": n_up, "n_models": n_total,
        "weights": weights,
    }


# ============================================================== PDF export ===
def build_recommendation_pdf(rec, current_price, horizon):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 60, "Kotulo — Maize Selling Recommendation")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 85, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 130, f"Decision: {rec['decision'].upper()}")

    c.setFont("Helvetica", 11)
    text = c.beginText(50, height - 160)
    for line in textwrap.wrap(rec["reason"], width=90):
        text.textLine(line)
    c.drawText(text)

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 240, f"Today's price: R {current_price:,.0f} / ton")
    c.drawString(50, height - 260, f"Forecast (day +{horizon}): R {rec['avg_price']:,.0f} / ton")
    c.drawString(50, height - 280, f"Expected change: {rec['change_pct']:+.1f}%")
    c.drawString(50, height - 300,
                 f"Model agreement: {rec['n_up']} of {rec['n_models']} models forecast a rise")

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, height - 340,
                 "This is not financial advice. Use it as one input together with your own farm knowledge.")

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================== session state =
for k, v in [("results", {}), ("rec", None), ("lang_code", "en"),
             ("sample_bytes", None), ("province", "Free State"),
             ("mode", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ==================================================== language selector (top) =
_c1, _c2 = st.columns([5, 1])
with _c2:
    _lang = st.selectbox(t("language_label"), list(LANGUAGES.keys()),
                         label_visibility="collapsed", key="lang_selector")
    st.session_state.lang_code = LANGUAGES[_lang]


# ============================================= shared defaults (before tabs) ==
data = None
y = pd.Series(dtype=float)
X = pd.DataFrame()
y_train = pd.Series(dtype=float)
y_test = pd.Series(dtype=float)
N = 0
train_size = 0
EXOG_COLS = []
TARGET = "Close"
TRAIN_FRAC = 0.8
HOLD_COST = 1.2
province = "Free State"


# ====================================================================== tabs ==
tab_home, tab_forecast, tab_rec, tab_help = st.tabs(
    [t("tab_home"), t("tab_forecast"), t("tab_recommend"), t("tab_help")]
)


# ----------------------------------------------------------------- HOME ------
with tab_home:
    st.markdown(f"""
    <div class="hero">
        <h1>🌾 {t('app_name')}</h1>
        <p>{t('hero_sub')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="explain">{t("home_explainer")}</div>', unsafe_allow_html=True)

    st.markdown(f"## {t('choose_how')}")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="card card-demo">
            <h3 style="margin-top:0;">{t('demo_card_title')}</h3>
            <p style="margin-bottom:0;">{t('demo_card_body')}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t("demo_btn"), use_container_width=True, type="primary", key="btn_demo"):
            st.session_state.mode = "demo"
            st.session_state.sample_bytes = None
            st.rerun()

    with c2:
        st.markdown(f"""
        <div class="card card-real">
            <h3 style="margin-top:0;">{t('real_card_title')}</h3>
            <p style="margin-bottom:0;">{t('real_card_body')}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t("real_btn"), use_container_width=True, key="btn_real"):
            st.session_state.mode = "real"
            st.rerun()

    if st.session_state.mode == "demo":
        st.markdown("---")
        st.success(t("demo_mode_banner"))

        if st.session_state.sample_bytes is None:
            st.session_state.sample_bytes = load_real_sample_data()

        data = load_data(st.session_state.sample_bytes[0],
                         st.session_state.sample_bytes[1], False)

        with st.expander(t("advanced_settings"), expanded=False):
            TRAIN_FRAC = st.slider(t("train_split_label"), 0.5, 0.95, 0.80, 0.05,
                                   help=t("train_split_help"))
            province = st.selectbox(t("province_label"),
                                    list(PROVINCE_HOLDING_COST.keys()),
                                    index=list(PROVINCE_HOLDING_COST).index(st.session_state.province))
            st.session_state.province = province
            HOLD_COST = st.slider(t("holding_cost_label"), 0.0, 5.0,
                                  float(PROVINCE_HOLDING_COST[province]), 0.1,
                                  help=t("holding_cost_help"))

        EXOG_COLS = st.multiselect(
            t("external_drivers_label"),
            [c for c in ["ZAR_USD", "Rainfall_mm", "Fuel_Price"] if c in data.columns],
            default=[c for c in ["ZAR_USD", "Rainfall_mm", "Fuel_Price"] if c in data.columns],
        )

        TARGET = "Close"
        N = len(data)
        train_size = int(N * TRAIN_FRAC)
        y = data[TARGET]
        X = data[EXOG_COLS]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        st.markdown(f"### {t('your_data_glance')}")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric(t("business_days_metric"), f"{N:,}")
        cc2.metric(t("date_range_metric"), f"{data.index.min().year} – {data.index.max().year}")
        cc3.metric(t("today_price"), f"R {y.iloc[-1]:,.0f} / ton")
        cc4.metric(t("province_metric"), province)
        with st.expander(t("show_data"), expanded=False):
            st.line_chart(data[[TARGET]])
            st.dataframe(data[[TARGET] + EXOG_COLS].tail(10), use_container_width=True)

        st.markdown(f"""
        ### {t('how_to_use')}
        <p><span class="step-badge">1</span>{t('step_1', forecast_tab=t('tab_forecast'))}</p>
        <p><span class="step-badge">2</span>{t('step_2', recommend_tab=t('tab_recommend'))}</p>
        """, unsafe_allow_html=True)

    elif st.session_state.mode == "real":
        st.markdown("---")
        st.markdown(f"### {t('upload_section_title')}")
        st.markdown(t("upload_explainer"))

        col_a, col_b = st.columns(2)
        pf_up = col_a.file_uploader(t("upload_prices_label"), type="csv", key="pf_upload")
        ef_up = col_b.file_uploader(t("upload_factors_label"), type="csv", key="ef_upload")
        dayfirst = st.checkbox(t("dayfirst_label"), value=False)

        if pf_up and ef_up:
            try:
                data = load_data(pf_up.getvalue(), ef_up.getvalue(), dayfirst)
            except Exception as err:
                st.error(t("error_could_not_read"))
                st.markdown(f"```\n{err}\n```\n\n{t('error_common_causes')}")
                st.stop()

            st.success(t("data_loaded_success", n=len(data)))

            with st.expander(t("advanced_settings"), expanded=False):
                TRAIN_FRAC = st.slider(t("train_split_label"), 0.5, 0.95, 0.80, 0.05,
                                       help=t("train_split_help"))
                province = st.selectbox(t("province_label"),
                                        list(PROVINCE_HOLDING_COST.keys()),
                                        index=list(PROVINCE_HOLDING_COST).index(st.session_state.province))
                st.session_state.province = province
                HOLD_COST = st.slider(t("holding_cost_label"), 0.0, 5.0,
                                      float(PROVINCE_HOLDING_COST[province]), 0.1,
                                      help=t("holding_cost_help"))

            EXOG_COLS = st.multiselect(
                t("external_drivers_label"),
                [c for c in ["ZAR_USD", "Rainfall_mm", "Fuel_Price"] if c in data.columns],
                default=[c for c in ["ZAR_USD", "Rainfall_mm", "Fuel_Price"] if c in data.columns],
            )

            TARGET = "Close"
            N = len(data)
            train_size = int(N * TRAIN_FRAC)
            y = data[TARGET]
            X = data[EXOG_COLS]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

            st.markdown(f"### {t('your_data_glance')}")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric(t("business_days_metric"), f"{N:,}")
            cc2.metric(t("date_range_metric"), f"{data.index.min().year} – {data.index.max().year}")
            cc3.metric(t("today_price"), f"R {y.iloc[-1]:,.0f} / ton")
            cc4.metric(t("province_metric"), province)
            with st.expander(t("show_data"), expanded=False):
                st.line_chart(data[[TARGET]])
                st.dataframe(data[[TARGET] + EXOG_COLS].tail(10), use_container_width=True)

            st.markdown(f"""
            ### {t('how_to_use')}
            <p><span class="step-badge">1</span>{t('step_1', forecast_tab=t('tab_forecast'))}</p>
            <p><span class="step-badge">2</span>{t('step_2', recommend_tab=t('tab_recommend'))}</p>
            """, unsafe_allow_html=True)
        else:
            st.info(t("upload_both_info"))

    else:
        st.info(t("pick_option"))


# -------------------------------------------------------------- FORECAST ------
with tab_forecast:
    if st.session_state.mode is None or data is None:
        st.info(t("unlock_panel"))
    else:
        st.markdown(f"""
        <div class="hero">
            <h1>📈 {t('tab_forecast')}</h1>
            <p>{t('forecast_hero_sub')}</p>
        </div>
        """, unsafe_allow_html=True)

        f_tabs = st.tabs(["SARIMAX", "XGBoost", "Seq2Seq"])

        with f_tabs[0]:
            with st.expander(t("settings_expander"), expanded=False):
                c = st.columns(4)
                p_ = c[0].number_input(t("sarimax_p_label"), 0, 5, 1, key="sar_p")
                d_ = c[1].number_input(t("sarimax_d_label"), 0, 2, 1, key="sar_d")
                q_ = c[2].number_input(t("sarimax_q_label"), 0, 5, 1, key="sar_q")
                n_ahead_s = c[3].number_input(t("sarimax_n_label"), 1, 60, 5, key="sar_n")
            if st.button(t("run_sarimax_btn"), key="run_sarimax"):
                with st.spinner(t("fitting_sarimax")):
                    try:
                        st.session_state.results["SARIMAX"] = run_sarimax(
                            data, y, X, EXOG_COLS, train_size, n_ahead_s, order=(p_, d_, q_))
                    except Exception as e:
                        st.error(t("sarimax_failed", err=e))
            if "SARIMAX" in st.session_state.results:
                r = st.session_state.results["SARIMAX"]
                st.success(t("forecast_success_range", n=n_ahead_s, price=r["forecast_price"]))
                if r["test_metrics"]:
                    st.dataframe(pd.DataFrame(r["test_metrics"], index=["SARIMAX"]).round(2),
                                 use_container_width=True)
                    fig, ax = plt.subplots(figsize=(11, 3.5))
                    ax.plot(y_train.iloc[-len(y_test)*2:], label="training")
                    ax.plot(r["test_actual"], "k", lw=1, label="actual")
                    ax.plot(r["test_pred"], label="SARIMAX")
                    ax.legend(fontsize=8); st.pyplot(fig)
            st.markdown(f"""
            <div class="model-note">
            {t('sarimax_explainer')}
            <br><br>
            <a href="{translate_link('SARIMAX is a classic statistical model.')}" target="_blank">{t('translate_explanation')}</a>
            </div>
            """, unsafe_allow_html=True)

        with f_tabs[1]:
            with st.expander(t("settings_expander"), expanded=False):
                c = st.columns(4)
                H_ = c[0].number_input(t("xgb_horizon_label"), 1, 20, 1, key="xg_h")
                mode_ = c[1].selectbox(t("xgb_predict_label"), ["diff", "level"], key="xg_m")
                lp_ = c[2].number_input(t("xgb_price_lags_label"), 3, 30, 10, key="xg_lp")
                le_ = c[3].number_input(t("xgb_driver_lags_label"), 1, 15, 5, key="xg_le")
                n_iter_ = st.number_input(t("xgb_iter_label"), 5, 200, 30, key="xg_ni")
            if st.button(t("run_xgboost_btn"), key="xg_run"):
                with st.spinner(t("fitting_xgb")):
                    try:
                        st.session_state.results["XGBoost"] = run_xgboost(
                            data, y, X, EXOG_COLS, train_size, H_,
                            horizon=H_, pred_mode=mode_, lp=lp_, le=le_, n_iter=n_iter_)
                    except Exception as e:
                        st.error(t("xgb_failed", err=e))
            if "XGBoost" in st.session_state.results:
                r = st.session_state.results["XGBoost"]
                st.success(t("forecast_success_day", n=H_, price=r["forecast_price"]))
                if r["test_metrics"]:
                    st.dataframe(pd.DataFrame(r["test_metrics"], index=["XGBoost"]).round(2),
                                 use_container_width=True)
            st.markdown(f"""
            <div class="model-note">
            {t('xgb_explainer')}
            <br><br>
            <a href="{translate_link('XGBoost is a tree-based model.')}" target="_blank">{t('translate_explanation')}</a>
            </div>
            """, unsafe_allow_html=True)

        with f_tabs[2]:
            with st.expander(t("settings_expander"), expanded=False):
                c = st.columns(4)
                ENC_  = c[0].number_input(t("s2s_enc_label"), 10, 250, 60, key="sq_e")
                PRED_ = c[1].number_input(t("s2s_pred_label"), 1, 20, 5, key="sq_p")
                LAT_  = c[2].number_input(t("s2s_latent_label"), 8, 128, 32, key="sq_l")
                EPS_  = c[3].number_input(t("s2s_epochs_label"), 5, 300, 50, key="sq_ep")
            if st.button(t("run_seq2seq_btn"), key="sq_run"):
                with st.spinner(t("training_s2s")):
                    try:
                        st.session_state.results["Seq2Seq"] = run_seq2seq(
                            data, y, X, EXOG_COLS, train_size, PRED_,
                            enc_len=ENC_, pred_steps=PRED_, latent=LAT_, epochs=EPS_)
                    except Exception as e:
                        st.error(t("s2s_failed", err=e))
            if "Seq2Seq" in st.session_state.results:
                r = st.session_state.results["Seq2Seq"]
                st.success(t("forecast_success_day", n=PRED_, price=r["forecast_price"]))
                if r["test_metrics"]:
                    st.dataframe(pd.DataFrame(r["test_metrics"], index=["Seq2Seq"]).round(2),
                                 use_container_width=True)
            st.markdown(f"""
            <div class="model-note">
            {t('s2s_explainer')}
            <br><br>
            <a href="{translate_link('Seq2Seq reads history and writes the next 5 days.')}" target="_blank">{t('translate_explanation')}</a>
            </div>
            """, unsafe_allow_html=True)


# --------------------------------------------------------- RECOMMENDATION -----
with tab_rec:
    if st.session_state.mode is None or data is None:
        st.info(t("unlock_panel"))
    else:
        st.markdown(f"""
        <div class="hero">
            <h1>💡 {t('tab_recommend')}</h1>
            <p>{t('recommend_hero_sub')}</p>
        </div>
        """, unsafe_allow_html=True)

        horizon = st.slider(t("horizon_label"), 1, 30, 5, help=t("horizon_help"))

        if st.button(t("btn_recommend"), type="primary", use_container_width=True):
            results = {}
            with st.status(t("running_models_status"), expanded=True) as status:
                st.write(t("fitting_sarimax"))
                try:
                    results["SARIMAX"] = run_sarimax(data, y, X, EXOG_COLS, train_size, horizon)
                except Exception as e:
                    st.warning(t("skipped_warning", model="SARIMAX", err=e))

                st.write(t("fitting_xgb"))
                try:
                    results["XGBoost"] = run_xgboost(data, y, X, EXOG_COLS, train_size, horizon,
                                                     horizon=horizon, fast=True)
                except Exception as e:
                    st.warning(t("skipped_warning", model="XGBoost", err=e))

                st.write(t("training_s2s"))
                try:
                    results["Seq2Seq"] = run_seq2seq(data, y, X, EXOG_COLS, train_size, horizon,
                                                     pred_steps=horizon, epochs=30)
                except Exception as e:
                    st.warning(t("skipped_warning", model="Seq2Seq", err=e))

                status.update(label=t("all_finished"), state="complete", expanded=False)

            if results:
                st.session_state.results.update(results)
                st.session_state.rec = make_recommendation(
                    results, float(y.iloc[-1]), horizon, hold_cost_pct=HOLD_COST)

        if st.session_state.rec:
            rec = st.session_state.rec
            is_hold = rec["decision"] == "hold"
            css_class = "card-hold" if is_hold else "card-sell"
            rec_class = "rec-hold" if is_hold else "rec-sell"
            emoji = "🟢" if is_hold else "🔴"
            word = t("decision_hold") if is_hold else t("decision_sell")

            st.markdown(f"""
            <div class="card {css_class}">
                <div style="font-size:0.95rem; color:#6B6B6B; text-transform:uppercase; letter-spacing:0.08em;">
                    {t('rec_header')}
                </div>
                <div class="rec-big {rec_class}">{emoji} {word}</div>
                <div class="rec-reason">{rec['reason']}</div>
                <div class="rec-conf">
                    {t('confidence')}: {t('confidence_line', up=rec['n_up'], total=rec['n_models'], down=rec['n_models'] - rec['n_up'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                pdf_bytes = build_recommendation_pdf(rec, float(y.iloc[-1]), horizon)
                st.download_button(
                    label=t("pdf_download"),
                    data=pdf_bytes,
                    file_name=f"kotulo_recommendation_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except ModuleNotFoundError:
                st.caption(t("pdf_unavailable"))

            c1, c2, c3 = st.columns(3)
            c1.metric(t("today_price"), f"R {y.iloc[-1]:,.0f}")
            c2.metric(f"{t('avg_forecast')} (+{horizon}d)", f"R {rec['avg_price']:,.0f}")
            c3.metric(t("expected_change"), f"{rec['change_pct']:+.1f}%")

            st.markdown(f"### {t('how_each_voted')}")
            rows = []
            for name in MODEL_NAMES:
                if name not in st.session_state.results:
                    continue
                r = st.session_state.results[name]
                change = (r["forecast_price"] - r["current_price"]) / r["current_price"] * 100
                if change > 0.5:
                    direction = t("direction_up")
                elif change < -0.5:
                    direction = t("direction_down")
                else:
                    direction = t("direction_flat")
                rows.append({
                    t("table_model"): name,
                    t("table_forecast"): round(r["forecast_price"], 2),
                    t("table_change"): f"{change:+.2f}%",
                    t("table_direction"): direction,
                    t("table_weight"): f"{rec['weights'].get(name, 0)*100:.0f}%",
                    t("table_test_mae"): round(r["test_metrics"]["MAE (R/ton)"], 2) if r.get("test_metrics") else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander(t("what_means")):
                st.markdown(
                    f"- {t('what_means_hold', hold=t('decision_hold'))}\n"
                    f"- {t('what_means_sell', sell=t('decision_sell'))}\n\n"
                    f"{t('not_advice')}"
                )
        else:
            st.info(t("press_button"))


# ----------------------------------------------------------------- HELP ------
with tab_help:
    st.markdown(f"""
    <div class="hero">
        <h1>📚 {t('tab_help')}</h1>
        <p>{t('help_hero_sub')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="explain">{t("help_intro")}</div>', unsafe_allow_html=True)

    with st.expander(f"🧠 {t('help_models_title')}", expanded=True):
        st.markdown(f'<p class="help-body">{t("help_models_intro")}</p>', unsafe_allow_html=True)
        for key in ["sarimax", "xgboost", "seq2seq"]:
            st.markdown(
                f'<div class="help-box">'
                f'<div class="help-h4">{t(f"help_{key}_title")}</div>'
                f'<p class="help-body">{t(f"help_{key}_desc")}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with st.expander(f"⚙️ {t('help_advanced_title')}", expanded=False):
        st.markdown(f'<p class="help-body">{t("help_advanced_intro")}</p>', unsafe_allow_html=True)
        for key in ["train_split", "province", "holding_cost", "drivers"]:
            st.markdown(
                f'<div class="help-box">'
                f'<div class="help-h4">{t(f"help_{key}_title")}</div>'
                f'<p class="help-body">{t(f"help_{key}_desc")}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with st.expander(f"📊 {t('help_driver_explain_title')}", expanded=False):
        st.markdown(f'<p class="help-body">{t("help_driver_explain_intro")}</p>', unsafe_allow_html=True)
        for key in ["zar", "rain", "fuel"]:
            st.markdown(
                f'<div class="help-box">'
                f'<div class="help-h4">{t(f"help_{key}_title")}</div>'
                f'<p class="help-body">{t(f"help_{key}_desc")}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with st.expander(f"🎛️ {t('help_settings_title')}", expanded=False):
        st.markdown(f'<p class="help-body">{t("help_settings_intro")}</p>', unsafe_allow_html=True)

        st.markdown(f'<div class="help-h3">{t("help_sarimax_settings_title")}</div>', unsafe_allow_html=True)
        for key in ["sarimax_p", "sarimax_d", "sarimax_q", "sarimax_n"]:
            st.markdown(f'<p class="help-body">• {t(f"help_{key}_desc")}</p>', unsafe_allow_html=True)

        st.markdown(f'<div class="help-h3">{t("help_xgb_settings_title")}</div>', unsafe_allow_html=True)
        for key in ["xgb_horizon", "xgb_predict", "xgb_plags", "xgb_dlags", "xgb_iter"]:
            st.markdown(f'<p class="help-body">• {t(f"help_{key}_desc")}</p>', unsafe_allow_html=True)

        st.markdown(f'<div class="help-h3">{t("help_s2s_settings_title")}</div>', unsafe_allow_html=True)
        for key in ["s2s_enc", "s2s_pred", "s2s_latent", "s2s_epochs"]:
            st.markdown(f'<p class="help-body">• {t(f"help_{key}_desc")}</p>', unsafe_allow_html=True)

    with st.expander(f"💡 {t('help_rec_title')}", expanded=False):
        st.markdown(f'<p class="help-body">{t("help_rec_intro")}</p>', unsafe_allow_html=True)
        for key in ["hold", "sell", "conf", "votes", "metrics", "pdf"]:
            st.markdown(
                f'<div class="help-box">'
                f'<div class="help-h4">{t(f"help_rec_{key}_title")}</div>'
                f'<p class="help-body">{t(f"help_rec_{key}_desc")}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ================================================================ footer =====
st.markdown(
    "<hr style='border:none;border-top:1px dashed #C9A961;margin-top:3rem'>"
    f"<div style='text-align:center;color:#8A8A8A;font-size:0.85rem;padding:1rem 0'>{t('footer')}</div>",
    unsafe_allow_html=True,
)