"""
CMP7005 – Programming for Data Analysis
PRAC1: Beijing Air Quality Interactive Dashboard
"""

import os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
PC   = "#3b82f6"
AC   = "#0ea5e9"
DARK = "#0f172a"
BG   = "#000000"

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], * {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}

/* ── App background ── */
.stApp {{
    background-color: {BG};
    background-image:
        radial-gradient(ellipse at 15% 0%, rgba(59,130,246,.12) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 100%, rgba(14,165,233,.08) 0%, transparent 50%);
}}

/* ── MAIN AREA TEXT — white on black ── */
[data-testid='stAppViewContainer'] h1,
[data-testid='stAppViewContainer'] h2,
[data-testid='stAppViewContainer'] h3,
[data-testid='stAppViewContainer'] h4,
[data-testid='stAppViewContainer'] h5,
[data-testid='stAppViewContainer'] h6 {{
    color: #ffffff !important;
}}
[data-testid='stAppViewContainer'] p,
[data-testid='stAppViewContainer'] li,
[data-testid='stAppViewContainer'] span:not(.badge) {{
    color: #e2e8f0;
}}
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
    color: #ffffff !important;
}}

/* Widget labels */
.stSlider label, .stSelectbox label,
.stMultiSelect label, .stDateInput label,
.stRadio > label, .stCheckbox label {{
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
}}

/* Slider value text */
.stSlider [data-testid='stTickBar'] span,
.stSlider [data-testid='stThumbValue'] {{
    color: #ffffff !important;
}}

/* Captions */
.stCaption, small {{ color: #94a3b8 !important; }}

/* ── Sidebar — white text ── */
section[data-testid='stSidebar'] {{
    background: {DARK} !important;
    border-right: 1px solid rgba(255,255,255,.06) !important;
    box-shadow: 4px 0 32px rgba(0,0,0,.3) !important;
}}
section[data-testid='stSidebar'] * {{ color: rgba(255,255,255,.82) !important; }}
section[data-testid='stSidebar'] .stRadio label {{ font-size:.87rem !important; color: rgba(255,255,255,.82) !important; }}
section[data-testid='stSidebar'] hr {{ border-color: rgba(255,255,255,.1) !important; }}

/* ── Hero — white text ── */
.hero {{
    background: linear-gradient(130deg, {DARK} 0%, #1e3a8a 55%, #1d4ed8 100%);
    padding: 34px 42px; border-radius: 24px; color: white;
    margin-bottom: 26px; box-shadow: 0 20px 60px rgba(15,23,42,.22);
    position: relative; overflow: hidden;
}}
.hero::before {{
    content:''; position:absolute; top:-70px; right:-50px;
    width:300px; height:300px;
    background: radial-gradient(circle, rgba(14,165,233,.25) 0%, transparent 70%);
    border-radius:50%;
}}
.hero h1 {{ margin:0; color:white !important; font-size:2.25rem; font-weight:800; letter-spacing:-.4px; }}
.hero .sub {{ margin:10px 0 16px; color:rgba(255,255,255,.65) !important; font-size:.97rem; max-width:560px; }}
.badge {{
    display:inline-block; background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.2); border-radius:20px;
    padding:4px 14px; font-size:.74rem; font-weight:600;
    color: white !important;
    letter-spacing:.3px; margin-right:6px; margin-top:4px;
}}

/* ── Section header ── */
.section-header {{
    color:#ffffff !important; font-size:1.1rem; font-weight:800;
    margin:4px 0 20px; padding-bottom:12px;
    border-bottom:2.5px solid {PC}; letter-spacing:-.2px;
}}

/* ── Metric cards ── */
div[data-testid='metric-container'] {{
    background:#111827; border:1px solid #1e293b;
    padding:20px 22px; border-radius:18px;
    box-shadow:0 1px 8px rgba(0,0,0,.4);
    transition:transform .2s, box-shadow .2s;
}}
div[data-testid='metric-container']:hover {{
    transform:translateY(-2px); box-shadow:0 8px 28px rgba(0,0,0,.6);
}}
div[data-testid='metric-container'] [data-testid='stMetricValue'] {{
    font-size:1.6rem !important; font-weight:800 !important; color:{PC} !important;
}}
div[data-testid='metric-container'] [data-testid='stMetricLabel'] {{
    font-size:.78rem !important; font-weight:600 !important;
    color:#94a3b8 !important; text-transform:uppercase; letter-spacing:.5px;
}}

/* ── Info / station cards ── */
.info-card {{
    background:#111827; border:1px solid #1e293b; border-left:5px solid {PC};
    padding:16px 20px; border-radius:14px;
    box-shadow:0 2px 10px rgba(0,0,0,.4); margin-bottom:14px;
}}
.station-card {{
    background:#111827; border:1px solid #1e293b;
    border-radius:16px; padding:16px 18px;
    box-shadow:0 2px 12px rgba(0,0,0,.4); height:100%;
}}

/* ── Tabs ── */
.stTabs [data-baseweb='tab-list'] {{
    background:#111827; border-radius:14px;
    padding:5px; box-shadow:0 1px 6px rgba(0,0,0,.4); gap:4px;
}}
.stTabs [data-baseweb='tab'] {{
    border-radius:10px; font-weight:600;
    font-size:.88rem; color:#94a3b8 !important; padding:8px 18px;
}}
.stTabs [aria-selected='true'] {{
    background:{PC} !important; color:white !important;
}}

/* ── Primary button ── */
button[kind='primary'] {{
    background:linear-gradient(135deg,{PC},{AC}) !important;
    border:none !important; border-radius:12px !important;
    font-weight:700 !important; letter-spacing:.3px !important;
    box-shadow:0 4px 16px rgba(29,78,216,.3) !important;
}}

/* ── Misc ── */
.stDataFrame {{ border-radius:14px; overflow:hidden; }}
.upload-area {{
    background:rgba(255,255,255,.08); border:1px dashed rgba(255,255,255,.3);
    border-radius:12px; padding:14px; font-size:.83rem; line-height:1.7; margin-bottom:10px;
}}
.footer {{
    text-align:center; padding:22px 0 10px; font-size:.76rem;
    color:#475569 !important; letter-spacing:.4px;
    border-top:1px solid #1e293b; margin-top:50px;
}}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════

# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
POLLUTANTS = ["PM2.5","PM10","SO2","NO2","CO","O3"]
METEO      = ["TEMP","PRES","DEWP","RAIN","WSPM"]

STATION_COORDS = {
    "Dongsi":    {"lat":39.9289, "lon":116.4173, "type":"Urban"},
    "Guanyuan":  {"lat":39.9290, "lon":116.3390, "type":"Urban"},
    "Changping": {"lat":40.2174, "lon":116.2310, "type":"Suburban"},
    "Huairou":   {"lat":40.3756, "lon":116.6308, "type":"Suburban"},
}
PALETTE_HEX = {
    "Dongsi":    "#ef4444",
    "Guanyuan":  "#f97316",
    "Changping": "#22c55e",
    "Huairou":   "#3b82f6",
}
SEASON_PAL = {
    "Spring":"#4ade80","Summer":"#f87171",
    "Autumn":"#fb923c","Winter":"#60a5fa",
}

DATA_DIR = "/content/CMP7005-AirQuality"
STATION_FILES = {
    "Dongsi":    "PRSA_Data_Dongsi_20130301-20170228.csv",
    "Guanyuan":  "PRSA_Data_Guanyuan_20130301-20170228.csv",
    "Changping": "PRSA_Data_Changping_20130301-20170228.csv",
    "Huairou":   "PRSA_Data_Huairou_20130301-20170228.csv",
}
REPO_OK = os.path.isdir(DATA_DIR) and all(
    os.path.exists(os.path.join(DATA_DIR, f)) for f in STATION_FILES.values()
)

# ═══════════════════════════════════════════════════════════════════

# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='padding:22px 0 6px;text-align:center'>"
        "<div style='font-size:2.6rem;line-height:1'>🌫️</div>"
        "<div style='font-size:1.2rem;font-weight:800;color:white;margin-top:8px;letter-spacing:-.3px'>"
        "Air Quality</div>"
        "<div style='font-size:.73rem;color:rgba(255,255,255,.4);margin-top:3px;"
        "letter-spacing:.6px;text-transform:uppercase'>Beijing · 2013 – 2017</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if REPO_OK:
        data_mode = st.radio(
            "📂 Data Source",
            ["Use repository sample data","Upload my own CSV files"],
        )
    else:
        data_mode = "Upload my own CSV files"
        st.info("No bundled dataset — please upload CSV files.")
    uploaded_files = None
    if data_mode == "Upload my own CSV files":
        st.markdown(
            "<div class='upload-area'>"
            "<b>📥 Upload Instructions</b><br>"
            "Upload one or more station CSV files.<br>"
            "Required: <code>year</code>, <code>month</code>, <code>day</code>, <code>hour</code><br>"
            "Pollutants: PM2.5, PM10, SO2, NO2, CO, O3"
            "</div>",
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "📎 Upload CSV file(s)", type=["csv"], accept_multiple_files=True
        )
    st.markdown("---")
    page = st.radio(
        "🧭 Navigation",
        ["🏠 Overview","🗺️ Station Map","🗂️ Dataset Explorer",
         "📊 Visualisations","🤖 Model Outputs","🔮 Predict PM2.5"],
    )
    st.markdown("---")
    st.caption("CMP7005 · PRAC1 · Air Quality Analysis")

# ═══════════════════════════════════════════════════════════════════

# HELPERS
# ═══════════════════════════════════════════════════════════════════
def aqi_category(pm25):
    if   pm25 <= 12:    return ("Good",           "✅","#16a34a")
    elif pm25 <= 35.4:  return ("Moderate",        "🟡","#ca8a04")
    elif pm25 <= 55.4:  return ("Sensitive Groups","🟠","#ea580c")
    elif pm25 <= 150.4: return ("Unhealthy",       "🔴","#dc2626")
    elif pm25 <= 250.4: return ("Very Unhealthy",  "🟣","#7c3aed")
    return ("Hazardous","⚫","#1e293b")

def aqi_label(pm25):
    a = aqi_category(pm25); return f"{a[1]} {a[0]}"

def section(title):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)

def plotly_layout(fig, h=400):
    fig.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#e2e8f0"),
        height=h, margin=dict(t=40,b=30,l=10,r=10),
    )
    return fig

def preprocess_df(df):
    df = df.copy()
    if "station" not in df.columns: df["station"] = "Uploaded Station"
    if all(c in df.columns for c in ["year","month","day","hour"]):
        df["datetime"] = pd.to_datetime(df[["year","month","day","hour"]], errors="coerce")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        raise ValueError("Need year/month/day/hour columns or a datetime column.")
    df = (df.dropna(subset=["datetime"]).sort_values("datetime")
            .reset_index(drop=True).drop_duplicates())
    nc = [c for c in POLLUTANTS+METEO if c in df.columns]
    for c in nc: df[c] = pd.to_numeric(df[c], errors="coerce")
    if nc: df[nc] = df[nc].ffill().bfill()
    if "wd" in df.columns: df["wd"] = df["wd"].ffill().bfill()
    df["year"]  = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"]   = df["datetime"].dt.day
    df["hour"]  = df["datetime"].dt.hour
    df["season"] = df["month"].map(
        {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
         6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
    )
    df["station_type"] = (
        df["station"]
        .replace({"Dongsi":"Urban","Guanyuan":"Urban","Changping":"Suburban","Huairou":"Suburban"})
        .where(lambda s: s.isin(["Urban","Suburban"]), other="Uploaded")
    )
    if "PM2.5" in df.columns:
        df["AQI_Category"] = df["PM2.5"].apply(
            lambda x: aqi_label(x) if pd.notna(x) else np.nan
        )
    return df

@st.cache_data(show_spinner="📂 Loading dataset…")

def load_repo_data():
    frames = []
    for station, fname in STATION_FILES.items():
        tmp = pd.read_csv(os.path.join(DATA_DIR, fname))
        tmp["station"] = station
        frames.append(tmp)
    return preprocess_df(pd.concat(frames, ignore_index=True))

def load_uploaded_files(uploaded_files):
    frames = []
    for file in uploaded_files:
        temp = pd.read_csv(file)
        if "station" not in temp.columns:
            inferred = file.name.replace(".csv","").replace("PRSA_Data_","")
            temp["station"] = inferred or "Uploaded Station"
        frames.append(temp)
    if not frames: raise ValueError("No uploaded files were read.")
    return preprocess_df(pd.concat(frames, ignore_index=True))

@st.cache_resource(show_spinner="🤖 Training ML models…")
def train_models(df):
    needed = ["PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN",
              "WSPM","month","hour","season","station","PM2.5"]
    model_df = df[[c for c in needed if c in df.columns]].copy()
    missing  = sorted(set(needed)-set(model_df.columns))
    if missing: raise ValueError(f"Missing columns: {missing}")
    model_df = model_df.dropna().copy()
    if len(model_df) < 50: raise ValueError("Not enough clean rows.")
    le_season = LabelEncoder(); le_station = LabelEncoder()
    model_df["season"]  = le_season.fit_transform(model_df["season"])
    model_df["station"] = le_station.fit_transform(model_df["station"])
    features = ["PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN",
                "WSPM","month","hour","season","station"]
    X, y = model_df[features], model_df["PM2.5"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train); X_te_sc = scaler.transform(X_test)
    models = {
        "📈 Linear Regression": LinearRegression(),
        "🌲 Random Forest":     RandomForestRegressor(n_estimators=150,random_state=42,n_jobs=-1),
        "🚀 Gradient Boosting": GradientBoostingRegressor(n_estimators=150,random_state=42),
    }
    results, trained = {}, {}
    for name, mdl in models.items():
        if "Linear" in name:
            mdl.fit(X_tr_sc,y_train); preds = mdl.predict(X_te_sc)
        else:
            mdl.fit(X_train,y_train); preds = mdl.predict(X_test)
        results[name] = {
            "R2":   round(r2_score(y_test,preds),4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test,preds))),3),
            "MAE":  round(float(mean_absolute_error(y_test,preds)),3),
            "preds":preds,"actuals":y_test.values,
        }
        trained[name] = mdl
    best = max(results, key=lambda x: results[x]["R2"])
    feat_imp = None
    if hasattr(trained[best],"feature_importances_"):
        feat_imp = pd.Series(trained[best].feature_importances_,index=features).sort_values(ascending=False)
    return results, trained, scaler, le_season, le_station, best, feat_imp, features

# ═══════════════════════════════════════════════════════════════════

# LOAD DATA
# ═══════════════════════════════════════════════════════════════════
try:
    if data_mode == "Use repository sample data":
        df = load_repo_data()
    else:
        if not uploaded_files:
            st.markdown(
                "<div class='hero'><h1>🌫️ Beijing Air Quality</h1>"
                "<p class='sub'>Upload your station CSV files from the sidebar to begin.</p>"
                "<span class='badge'>📊 Analytics</span>"
                "<span class='badge'>🤖 ML Models</span>"
                "<span class='badge'>🗺️ Station Map</span></div>",
                unsafe_allow_html=True,
            )
            st.info("📌 Please upload at least one CSV file to continue.")
            st.stop()
        df = load_uploaded_files(uploaded_files)
except Exception as e:
    st.error(f"❌ Data loading failed: {e}"); st.stop()

# ═══════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='hero'>"
    "<h1>🌫️ Beijing Air Quality</h1>"
    "<p class='sub'>Explore pollution trends, visualise meteorological patterns, train predictive "
    "models, and forecast PM2.5 concentrations across Beijing's monitoring network.</p>"
    "<span class='badge'>📊 Interactive Analytics</span>"
    "<span class='badge'>🤖 ML Predictions</span>"
    "<span class='badge'>🗺️ Live Station Map</span>"
    "<span class='badge'>4 Monitoring Stations</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
# GLOBAL FILTERS
# ═══════════════════════════════════════════════════════════════════
section("🎛️ Global Filters")
f1,f2,f3 = st.columns([1.2,1.2,1.6])
station_options = sorted(df["station"].dropna().unique().tolist())
season_options  = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
selected_stations = f1.multiselect("🏭 Station(s)", station_options, default=station_options)
selected_seasons  = f2.multiselect("🍂 Season(s)",  season_options,  default=season_options)
date_min   = df["datetime"].min().date()
date_max   = df["datetime"].max().date()
date_range = f3.date_input("📅 Date Range", value=(date_min,date_max),
                            min_value=date_min, max_value=date_max)
filtered_df = df[df["station"].isin(selected_stations)].copy()
if selected_seasons:
    filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
if isinstance(date_range,tuple) and len(date_range)==2:
    sd,ed = date_range
    filtered_df = filtered_df[
        (filtered_df["datetime"].dt.date>=sd)&(filtered_df["datetime"].dt.date<=ed)
    ]
if filtered_df.empty:
    st.warning("⚠️ No data after the selected filters."); st.stop()

# ═══════════════════════════════════════════════════════════════════

# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    section("🏠 Dashboard Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📄 Records",   f"{len(filtered_df):,}")
    c2.metric("🏭 Stations",  f"{filtered_df['station'].nunique()}")
    c3.metric("🌫️ Avg PM2.5",
              f"{filtered_df['PM2.5'].mean():.1f} µg/m³" if "PM2.5" in filtered_df.columns else "N/A")
    c4.metric("📈 Max PM2.5",
              f"{filtered_df['PM2.5'].max():.0f} µg/m³"  if "PM2.5" in filtered_df.columns else "N/A")
    c5.metric("📅 Years",
              f"{filtered_df['datetime'].dt.year.min()} – {filtered_df['datetime'].dt.year.max()}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns([2.4,1])
    with col1:
        st.subheader("📉 Monthly Average PM2.5 by Station")
        monthly = filtered_df.groupby(["year","month","station"])["PM2.5"].mean().reset_index()
        monthly["date"] = pd.to_datetime(monthly[["year","month"]].assign(day=1))
        fig_m = px.line(monthly, x="date", y="PM2.5", color="station",
                        color_discrete_map=PALETTE_HEX,
                        labels={"PM2.5":"PM2.5 (µg/m³)","date":"Date","station":"Station"})
        fig_m.add_hline(y=35.4, line_dash="dot", line_color="#f59e0b",
                        annotation_text="Moderate threshold", annotation_position="top left")
        fig_m.update_traces(line=dict(width=2.3))
        plotly_layout(fig_m, h=380)
        fig_m.update_xaxes(showgrid=False); fig_m.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_m, use_container_width=True)
    with col2:
        st.subheader("🩺 AQI Breakdown")
        if "AQI_Category" in filtered_df.columns:
            counts = filtered_df["AQI_Category"].value_counts().reset_index()
            counts.columns = ["Category","Count"]
            aqi_colors = {"✅ Good":"#16a34a","🟡 Moderate":"#ca8a04",
                          "🟠 Sensitive Groups":"#ea580c","🔴 Unhealthy":"#dc2626",
                          "🟣 Very Unhealthy":"#7c3aed","⚫ Hazardous":"#1e293b"}
            fig_aqi = px.bar(counts, y="Category", x="Count", orientation="h",
                             color="Category",
                             color_discrete_map={k:v for k,v in aqi_colors.items()},
                             height=380)
            fig_aqi.update_layout(showlegend=False, plot_bgcolor="#111827", paper_bgcolor="#111827",
                                  font=dict(family="Plus Jakarta Sans", color="#e2e8f0"),
                                  margin=dict(t=20,b=20,l=10,r=10))
            fig_aqi.update_xaxes(showgrid=False); fig_aqi.update_yaxes(showgrid=False)
            st.plotly_chart(fig_aqi, use_container_width=True)

    st.subheader("📋 Descriptive Statistics")
    nc = [c for c in POLLUTANTS+METEO if c in filtered_df.columns]
    if nc:
        st.dataframe(
            filtered_df[nc].describe().round(2).style.background_gradient(cmap="Blues",axis=1),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════

# PAGE: STATION MAP
# ═══════════════════════════════════════════════════════════════════
elif page == "🗺️ Station Map":
    section("🗺️ Beijing Monitoring Station Map")
    if "PM2.5" in filtered_df.columns:
        st_stats = (filtered_df.groupby("station")["PM2.5"]
                    .agg(avg="mean",mx="max",mn="min",cnt="count").reset_index())
    else:
        st_stats = pd.DataFrame({"station":list(STATION_COORDS.keys())})
        st_stats["avg"] = st_stats["mx"] = st_stats["mn"] = st_stats["cnt"] = 0

    map_rows = []
    for _,row in st_stats.iterrows():
        sn   = row["station"]
        info = STATION_COORDS.get(sn,{"lat":39.92,"lon":116.40,"type":"Unknown"})
        avg  = round(float(row["avg"]),1) if pd.notna(row.get("avg")) else 0.0
        aqi  = aqi_category(avg)
        map_rows.append({
            "Station":    sn, "lat": info["lat"], "lon": info["lon"],
            "Type":       info["type"], "Avg PM2.5": avg,
            "Max PM2.5":  round(float(row.get("mx",0)),1),
            "Records":    int(row.get("cnt",0)),
            "AQI Status": f"{aqi[1]} {aqi[0]}",
            "size":       max(20.0, avg*0.45+22),
        })
    map_df = pd.DataFrame(map_rows)

    cols = st.columns(len(map_rows))
    for i,row in enumerate(map_rows):
        aqi_c    = aqi_category(row["Avg PM2.5"])
        type_bg  = "#eff6ff" if row["Type"]=="Urban" else "#f0fdf4"
        type_col = "#1d4ed8" if row["Type"]=="Urban" else "#15803d"
        with cols[i]:
            st.markdown(
                f"<div class='station-card'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
                f"<span style='font-size:1rem;font-weight:800;color:{DARK}'>{row['Station']}</span>"
                f"<span style='background:{type_bg};color:{type_col};border-radius:8px;"
                f"padding:2px 9px;font-size:.71rem;font-weight:700'>{row['Type']}</span>"
                f"</div>"
                f"<div style='width:32px;height:4px;background:{PALETTE_HEX.get(row['Station'],PC)};"
                f"border-radius:2px;margin-bottom:10px'></div>"
                f"<div style='font-size:1.55rem;font-weight:800;"
                f"color:{PALETTE_HEX.get(row['Station'],PC)};line-height:1'>"
                f"{row['Avg PM2.5']}"
                f"<span style='font-size:.78rem;font-weight:500;color:#94a3b8'> µg/m³</span></div>"
                f"<div style='font-size:.73rem;color:#64748b;margin-top:2px'>Average PM2.5</div>"
                f"<div style='font-size:.82rem;font-weight:700;color:{aqi_c[2]};margin-top:7px'>"
                f"{aqi_c[1]} {aqi_c[0]}</div>"
                f"<div style='font-size:.71rem;color:#94a3b8;margin-top:5px'>"
                f"Max {row['Max PM2.5']} µg/m³ &nbsp;·&nbsp; {row['Records']:,} records</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

    # Folium dark street map — renders real map tiles, no token needed
    try:
        import folium
        from streamlit.components.v1 import html as st_html

        m = folium.Map(
            location=[40.05, 116.38],
            zoom_start=9,
            tiles="CartoDB dark_matter",
            width="100%", height=520,
        )

        for row in map_rows:
            col   = PALETTE_HEX.get(row["Station"], "#3b82f6")
            aqi_c = aqi_category(row["Avg PM2.5"])
            radius = max(14, int(row["Avg PM2.5"] * 0.25 + 14))

            popup_html = f"""
            <div style='font-family:sans-serif;min-width:180px;padding:4px'>
                <b style='font-size:15px;color:{col}'>{row['Station']}</b><br>
                <span style='color:#94a3b8;font-size:11px'>{row['Type']} Station</span><br><hr style='margin:6px 0;border-color:#334155'>
                <b>Avg PM2.5:</b> {row['Avg PM2.5']} µg/m³<br>
                <b>Max PM2.5:</b> {row['Max PM2.5']} µg/m³<br>
                <b>Records:</b> {row['Records']:,}<br>
                <b>AQI:</b> <span style='color:{aqi_c[2]}'>{aqi_c[1]} {aqi_c[0]}</span>
            </div>"""

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color="#000000",
                weight=1.5,
                fill=True,
                fill_color=col,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"<b>{row['Station']}</b> — {row['Avg PM2.5']} µg/m³",
            ).add_to(m)

            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(
                    html=f"<div style='font-family:sans-serif;font-size:11px;"
                         f"font-weight:700;color:white;text-shadow:0 0 4px #000;"
                         f"white-space:nowrap;margin-top:{radius+4}px;"
                         f"margin-left:-30px'>{row['Station']}</div>",
                    icon_size=(100, 20),
                ),
            ).add_to(m)

        st_html(m._repr_html_(), height=540, scrolling=False)

    except ImportError:
        st.warning("📦 Install folium to see the map: `pip install folium`")

    if "season" in filtered_df.columns and "PM2.5" in filtered_df.columns:
        st.subheader("🍂 Seasonal PM2.5 by Station")
        sea_data = filtered_df.groupby(["station","season"])["PM2.5"].mean().reset_index()
        fig_sea = px.bar(sea_data, x="station", y="PM2.5", color="season",
                         barmode="group", color_discrete_map=SEASON_PAL,
                         labels={"PM2.5":"Avg PM2.5 (µg/m³)","station":"Station","season":"Season"},
                         height=380)
        plotly_layout(fig_sea,380)
        fig_sea.update_xaxes(showgrid=False); fig_sea.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_sea, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════

# PAGE: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════
elif page == "🗂️ Dataset Explorer":
    section("🗂️ Dataset Explorer")
    tab1,tab2,tab3 = st.tabs(["👀 Data Preview","🧱 Schema & Types","🩹 Missing Values"])
    with tab1:
        n = st.slider("Rows to preview",10,500,50,10)
        st.dataframe(filtered_df.head(n), use_container_width=True, height=420)
    with tab2:
        info_df = pd.DataFrame({
            "Column":         filtered_df.columns,
            "Dtype":          filtered_df.dtypes.astype(str).values,
            "Non-Null Count": filtered_df.notna().sum().values,
            "Null %":         (filtered_df.isnull().mean()*100).round(2).values,
        })
        st.dataframe(info_df, use_container_width=True, height=420)
    with tab3:
        miss = filtered_df.isnull().sum().reset_index()
        miss.columns = ["Column","Missing Count"]
        miss["Missing %"] = (miss["Missing Count"]/len(filtered_df)*100).round(2)
        miss = miss[miss["Missing Count"]>0].sort_values("Missing %",ascending=False)
        if miss.empty:
            st.success("✅ No missing values detected.")
        else:
            fig_m = px.bar(miss, x="Column", y="Missing %",
                           color="Missing %", color_continuous_scale="Blues", height=360)
            plotly_layout(fig_m,360); fig_m.update_layout(coloraxis_showscale=False)
            fig_m.update_xaxes(showgrid=False); fig_m.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(miss, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════

# PAGE: VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 Visualisations":
    section("📊 Visualisations & Insights")
    tab1,tab2,tab3,tab4 = st.tabs(
        ["📦 Distribution","🔗 Scatter Plot","🌡️ Correlation Heatmap","⏳ Temporal Trends"]
    )
    with tab1:
        var = st.selectbox("📌 Variable",[c for c in POLLUTANTS+METEO if c in filtered_df.columns])
        c1,c2 = st.columns(2)
        with c1:
            f = px.histogram(filtered_df,x=var,nbins=50,color_discrete_sequence=[PC],
                             title=f"Distribution of {var}",height=380)
            plotly_layout(f,380); f.update_layout(bargap=0.04)
            f.update_xaxes(showgrid=False); f.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(f, use_container_width=True)
        with c2:
            f = px.box(filtered_df,x="station",y=var,color="station",
                       color_discrete_map=PALETTE_HEX,title=f"{var} by Station",height=380)
            plotly_layout(f,380); f.update_layout(showlegend=False)
            f.update_xaxes(showgrid=False); f.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(f, use_container_width=True)
    with tab2:
        ca = [c for c in POLLUTANTS+METEO if c in filtered_df.columns]
        cx,cy = st.columns(2)
        xc = cx.selectbox("↔️ X",ca,index=min(1,len(ca)-1))
        yc = cy.selectbox("↕️ Y",ca,index=0)
        smp = filtered_df.sample(min(7000,len(filtered_df)),random_state=42)
        f = px.scatter(smp,x=xc,y=yc,color="station",color_discrete_map=PALETTE_HEX,
                       opacity=0.4,height=450,trendline="ols")
        f.update_traces(marker=dict(size=4))
        plotly_layout(f,450)
        f.update_xaxes(showgrid=False); f.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(f, use_container_width=True)
    with tab3:
        cc = st.multiselect(
            "Variables for Heatmap",
            [c for c in POLLUTANTS+METEO if c in filtered_df.columns],
            default=[c for c in ["PM2.5","PM10","SO2","NO2","TEMP","WSPM"]
                     if c in filtered_df.columns],
        )
        if len(cc)>1:
            corr = filtered_df[cc].corr()
            f = px.imshow(corr,text_auto=".2f",aspect="auto",
                          color_continuous_scale="RdYlBu_r",
                          zmin=-1,zmax=1,height=500,title="Pearson Correlation Matrix")
            plotly_layout(f,500)
            st.plotly_chart(f, use_container_width=True)
        else:
            st.info("Select at least 2 variables.")
    with tab4:
        t1,t2 = st.columns([1.2,1])
        tv = t1.selectbox("📍 Variable",
                          [c for c in POLLUTANTS+METEO if c in filtered_df.columns],key="tv")
        go = t2.radio("🕒 Group By",["month","hour","season","year"],horizontal=True)
        td = filtered_df.groupby(["station",go])[tv].mean().reset_index()
        td[go] = td[go].astype(str)
        f = px.line(td,x=go,y=tv,color="station",color_discrete_map=PALETTE_HEX,
                    markers=True,height=420,labels={tv:f"Avg {tv}",go:go.title()})
        f.update_traces(line=dict(width=2.3),marker=dict(size=6))
        plotly_layout(f,420)
        f.update_xaxes(showgrid=False); f.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(f, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════

# PAGE: MODEL OUTPUTS
# ═══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Outputs":
    section("🤖 Machine Learning Model Outputs")
    try:
        results,trained,scaler,le_season,le_station,best_name,feat_imp,features = train_models(filtered_df)
        st.success(
            f"🏆 Best: **{best_name}**  ·  R² = `{results[best_name]['R2']}`  ·  "
            f"RMSE = `{results[best_name]['RMSE']} µg/m³`  ·  MAE = `{results[best_name]['MAE']} µg/m³`"
        )
        metrics_df = pd.DataFrame(
            {n:{"R²":r["R2"],"RMSE":r["RMSE"],"MAE":r["MAE"]} for n,r in results.items()}
        ).T
        st.dataframe(
            metrics_df.style.background_gradient(cmap="Blues",subset=["R²"]),
            use_container_width=True,
        )
        st.markdown("---")
        col1,col2 = st.columns(2)
        with col1:
            mn = st.selectbox("🤖 Model to inspect", list(results.keys()))
            r  = results[mn]
            rng = np.random.default_rng(42)
            idx = rng.choice(len(r["actuals"]),min(2500,len(r["actuals"])),replace=False)
            fig_av = go.Figure()
            fig_av.add_trace(go.Scatter(
                x=r["actuals"][idx],y=r["preds"][idx],mode="markers",
                marker=dict(color=PC,size=4,opacity=0.35),name="Samples"))
            lim = float(max(r["actuals"].max(),r["preds"].max()))*1.05
            fig_av.add_trace(go.Scatter(
                x=[0,lim],y=[0,lim],mode="lines",
                line=dict(color="red",dash="dash",width=1.5),name="Perfect"))
            plotly_layout(fig_av,420)
            fig_av.update_layout(title="Actual vs Predicted",
                                 xaxis_title="Actual (µg/m³)",yaxis_title="Predicted (µg/m³)")
            fig_av.update_xaxes(showgrid=False); fig_av.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig_av, use_container_width=True)
        with col2:
            resid = r["actuals"]-r["preds"]
            fig_res = px.histogram(x=resid,nbins=60,color_discrete_sequence=[AC],
                                   height=420,labels={"x":"Residual (Actual − Predicted)"},
                                   title="Residual Distribution")
            fig_res.add_vline(x=0,line_color="red",line_dash="dash",line_width=1.8)
            plotly_layout(fig_res,420); fig_res.update_layout(bargap=0.05)
            fig_res.update_xaxes(showgrid=False); fig_res.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig_res, use_container_width=True)
        if feat_imp is not None:
            st.subheader("🧠 Feature Importance — Best Model")
            fi_df = feat_imp.reset_index(); fi_df.columns = ["Feature","Importance"]
            fig_fi = px.bar(fi_df,x="Feature",y="Importance",
                            color="Importance",
                            color_continuous_scale=["#bfdbfe",PC,"#1e3a8a"],height=380)
            plotly_layout(fig_fi,380); fig_fi.update_layout(coloraxis_showscale=False)
            fig_fi.update_xaxes(showgrid=False); fig_fi.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")

# ═══════════════════════════════════════════════════════════════════

# PAGE: PREDICT PM2.5
# ═══════════════════════════════════════════════════════════════════
elif page == "🔮 Predict PM2.5":
    section("🔮 PM2.5 Prediction Tool")
    try:
        results,trained,scaler,le_season,le_station,best_name,feat_imp,features = train_models(filtered_df)
        st.markdown(
            f"<div class='info-card'>"
            f"<b>🏆 Active Model:</b> {best_name}&nbsp;·&nbsp;"
            f"<b>R²:</b> {results[best_name]['R2']}&nbsp;·&nbsp;"
            f"<b>RMSE:</b> {results[best_name]['RMSE']} µg/m³</div>",
            unsafe_allow_html=True,
        )
        col1,col2 = st.columns(2)
        with col1:
            st.markdown(f"<div style='font-weight:700;color:{DARK};font-size:.95rem;margin-bottom:10px'>🧪 Pollutant Inputs</div>",
                        unsafe_allow_html=True)
            pm10 = st.slider("🌫️ PM10 (µg/m³)",  0.0, 900.0, 100.0, 1.0)
            so2  = st.slider("🧪 SO2  (µg/m³)",  0.0, 500.0,  20.0, 1.0)
            no2  = st.slider("🟤 NO2  (µg/m³)",  0.0, 300.0,  50.0, 1.0)
            co   = st.slider("⚫ CO   (µg/m³)",  0.0,10000.0,1200.0,10.0)
            o3   = st.slider("🔵 O3   (µg/m³)",  0.0, 300.0,  60.0, 1.0)
        with col2:
            st.markdown(f"<div style='font-weight:700;color:{DARK};font-size:.95rem;margin-bottom:10px'>🌤️ Meteorological Inputs</div>",
                        unsafe_allow_html=True)
            temp  = st.slider("🌡️ Temperature (°C)",-30.0,42.0, 15.0,0.5)
            pres  = st.slider("📈 Pressure (hPa)",  980.0,1040.0,1010.0,0.5)
            dewp  = st.slider("💧 Dew Point (°C)",  -45.0,30.0,  2.0,0.5)
            rain  = st.slider("🌧️ Rainfall (mm)",    0.0,72.0,   0.0,0.1)
            wspm  = st.slider("💨 Wind Speed (m/s)", 0.0,14.0,   2.0,0.1)
            month = st.slider("📅 Month",1,12,6)
            hour  = st.slider("⏰ Hour", 0,23,12)
            sea_in = st.selectbox("🍂 Season",["Spring","Summer","Autumn","Winter"])
            sta_in = st.selectbox("🏭 Station",le_station.classes_.tolist())
        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            inp  = np.array([[pm10,so2,no2,co,o3,temp,pres,dewp,rain,wspm,
                              month,hour,
                              le_season.transform([sea_in])[0],
                              le_station.transform([sta_in])[0]]])
            mdl  = trained[best_name]
            pred = (mdl.predict(scaler.transform(inp))[0] if "Linear" in best_name
                    else mdl.predict(pd.DataFrame(inp,columns=features))[0])
            pred = max(0.0,float(pred))
            aqi  = aqi_category(pred)
            r1,r2 = st.columns(2)
            r1.metric("🌫️ Predicted PM2.5",f"{pred:.2f} µg/m³")
            r2.markdown(
                f"<div class='info-card' style='border-left-color:{aqi[2]}'>"
                f"<div style='font-size:.78rem;font-weight:600;color:#64748b;margin-bottom:4px'>"
                f"🩺 Air Quality Status</div>"
                f"<div style='font-size:1.7rem;font-weight:800;color:{aqi[2]}'>"
                f"{aqi[1]} {aqi[0]}</div></div>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"❌ Prediction tool unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='footer'>"
    "Beijing Air Quality &nbsp;·&nbsp; CMP7005 PRAC1 "
    "&nbsp;·&nbsp; Built with Streamlit · Plotly · Scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
