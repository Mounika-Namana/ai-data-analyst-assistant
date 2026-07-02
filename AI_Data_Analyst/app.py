import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Custom Module Imports
import data_loader
import data_cleaner
import eda
import visualizer
import ai_insights
import report_generator

# Page Config
st.set_page_config(
    page_title="AI Data Analyst - Executive Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CHARTS_DIR = "charts"
REPORTS_DIR = "reports"
DATASETS_DIR = "datasets"

# Create directories
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# Custom Styling Injection
def inject_custom_css():
    st.markdown("""
    <style>
    /* Executive Workspace Theme styling */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700 !important;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .cleaned-badge {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    /* Tab modifications */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Create sample dataset helper
def check_and_create_sample_dataset():
    sample_path = os.path.join(DATASETS_DIR, "corporate_sales_sample.csv")
    if not os.path.exists(sample_path):
        # Generate representative dataset with dirty values
        np.random.seed(42)
        n_rows = 250
        
        dates = pd.date_range(start="2025-01-01", periods=50, freq='D').tolist() * 5
        regions = np.random.choice(["North Region", "South Region", "East Region", "West Region"], size=n_rows)
        departments = np.random.choice(["Sales & Marketing", "Engineering", "Operations", "Finance", "HR"], size=n_rows)
        
        # Sales numeric column
        sales = np.random.normal(loc=65000, scale=18000, size=n_rows)
        # Add random outliers
        sales[12] = 250000
        sales[84] = 180000
        # Add NaNs
        sales[np.random.choice(n_rows, 18, replace=False)] = np.nan
        
        # Profit numeric column
        profit = sales * np.random.uniform(0.12, 0.28, size=n_rows)
        profit[np.random.choice(n_rows, 12, replace=False)] = np.nan
        
        # Satisfaction ratings
        sat = np.random.randint(1, 6, size=n_rows).astype(float)
        sat[np.random.choice(n_rows, 25, replace=False)] = np.nan
        
        df = pd.DataFrame({
            "Transaction Date": dates,
            "Employee ID": np.random.randint(1001, 1060, size=n_rows),
            "Department": departments,
            "Geographic Region": regions,
            "Total Sales": sales,
            "Net Profit": profit,
            "Customer Satisfaction Rating": sat
        })
        
        # Inject duplicate rows
        duplicates = df.sample(n=12, random_state=42)
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(sample_path, index=False)

check_and_create_sample_dataset()

# Initialize session state variables
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'cleaning_log' not in st.session_state:
    st.session_state.cleaning_log = None
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'loaded_filename' not in st.session_state:
    st.session_state.loaded_filename = ""

# Sidebar Configuration Layout
st.sidebar.markdown("<h2 style='text-align: center; color: white; background-color: #2c3e50; padding: 10px; border-radius: 8px;'>🤖 AI DATA ANALYST</h2>", unsafe_allow_html=True)
st.sidebar.write("")

# Section A: LLM Configuration
st.sidebar.subheader("🔑 1. AI API Configuration")
api_provider = st.sidebar.selectbox("API Provider", list(ai_insights.PROVIDERS.keys()))
api_model = st.sidebar.selectbox("Model", ai_insights.PROVIDERS[api_provider])
api_key = st.sidebar.text_input("Enter API Key", type="password", help="Providing your key allows the AI Insight Engine and Chat with Data to work.")

st.sidebar.markdown("---")

# Section B: Data Source
st.sidebar.subheader("📂 2. Load Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Option to use sample dataset
load_sample = st.sidebar.button("💡 Use Demo Corporate Dataset")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Instructions**:\n1. Load your file or click the demo button.\n2. Go through the tabs to Clean, Analyze, and Chat.\n3. Complete by compiling the Executive PDF Report.")

# Ingestion Logic
loaded_df = None
filename = ""

if uploaded_file is not None:
    filename = uploaded_file.name
    # Cache and load
    if st.session_state.loaded_filename != filename:
        # Save locally to datasets
        save_path = os.path.join(DATASETS_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            loaded_df = data_loader.load_data(save_path)
            st.session_state.df_raw = loaded_df
            st.session_state.df_cleaned = None
            st.session_state.cleaning_log = None
            st.session_state.eda_results = None
            st.session_state.ai_insights = None
            st.session_state.chat_history = []
            st.session_state.loaded_filename = filename
            st.success(f"Successfully loaded {filename}!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            
elif load_sample:
    filename = "corporate_sales_sample.csv"
    sample_path = os.path.join(DATASETS_DIR, filename)
    try:
        loaded_df = data_loader.load_data(sample_path)
        st.session_state.df_raw = loaded_df
        st.session_state.df_cleaned = None
        st.session_state.cleaning_log = None
        st.session_state.eda_results = None
        st.session_state.ai_insights = None
        st.session_state.chat_history = []
        st.session_state.loaded_filename = filename
        st.success("Loaded demo corporate dataset!")
    except Exception as e:
        st.error(f"Error loading demo dataset: {str(e)}")

# Core Workspace View
if st.session_state.df_raw is not None:
    df_active = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df_raw
    is_cleaned = st.session_state.df_cleaned is not None
    
    # Header summary
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title(f"📊 Workspace: {st.session_state.loaded_filename}")
    with col_h2:
        if is_cleaned:
            st.markdown("<span class='cleaned-badge'>✓ Sanitized & Cleaned</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='cleaned-badge' style='background-color:#fee2e2; color:#991b1b;'>⚠ Raw Uncleaned Data</span>", unsafe_allow_html=True)
            
    # TAB MENU STRUCTURE
    tab_clean, tab_eda, tab_dashboard, tab_forecast, tab_ai, tab_chat, tab_report = st.tabs([
        "📋 Data Cleaning",
        "📊 EDA Metrics",
        "📈 Interactive Dashboard",
        "🔮 Forecasting",
        "🤖 AI Insights",
        "💬 Chat with Data",
        "📄 Export Report"
    ])
    
    # ------------------ TAB 1: DATA CLEANING ------------------
    with tab_clean:
        st.header("📋 File Loader & Sanitization Pipeline")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader("Raw File Dimensions")
            st.write(f"- Rows: `{st.session_state.df_raw.shape[0]}`")
            st.write(f"- Columns: `{st.session_state.df_raw.shape[1]}`")
            
        with col_c2:
            st.write("")
            run_clean = st.button("🧼 Run Auto Data Cleaning Pipeline", type="primary")
            
        if run_clean:
            with st.spinner("Sanitizing columns, removing duplicates, and imputing missing fields..."):
                cleaned_df, log = data_cleaner.clean_data(st.session_state.df_raw)
                st.session_state.df_cleaned = cleaned_df
                st.session_state.cleaning_log = log
                # Pre-run EDA on cleaned data
                st.session_state.eda_results = eda.generate_eda(cleaned_df)
                st.success("Sanitization complete! Column names standardizing, null counts, and duplicates handled.")
                st.rerun()
                
        st.subheader("Data Preview")
        # Let user choose to view Raw or Cleaned preview
        preview_mode = st.radio("Preview Data:", ["Raw Data", "Cleaned Data"] if is_cleaned else ["Raw Data"], horizontal=True)
        
        if preview_mode == "Cleaned Data" and is_cleaned:
            st.dataframe(st.session_state.df_cleaned.head(15), use_container_width=True)
            
            # Show Cleaning Logs
            st.markdown("---")
            st.subheader("Data Cleaning Log Summary")
            log = st.session_state.cleaning_log
            
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                st.metric("Duplicates Removed", log['duplicates_removed'])
            with col_l2:
                st.metric("Row count diff", f"{log['original_shape'][0]} ➜ {log['new_shape'][0]}")
            with col_l3:
                st.metric("Total Imputed Fields", len(log['missing_numeric_filled']) + len(log['missing_categorical_filled']))
                
            # Details dropdowns
            with st.expander("Show Column Renaming Maps"):
                st.write(log['columns_renamed'])
                
            with st.expander("Imputed Numerical Values (Filled with Median)"):
                if log['missing_numeric_filled']:
                    st.write(log['missing_numeric_filled'])
                else:
                    st.write("No missing numeric values filled.")
                    
            with st.expander("Imputed Categorical Values (Filled with Mode)"):
                if log['missing_categorical_filled']:
                    st.write(log['missing_categorical_filled'])
                else:
                    st.write("No missing categorical values filled.")
        else:
            st.dataframe(st.session_state.df_raw.head(15), use_container_width=True)
            
    # ------------------ TAB 2: EXPLORATORY DATA ANALYSIS ------------------
    with tab_eda:
        st.header("📊 Exploratory Data Analysis Overview")
        if st.session_state.eda_results is None:
            # Generate EDA on active data
            st.session_state.eda_results = eda.generate_eda(df_active)
            
        eda_res = st.session_state.eda_results
        
        col_e1, col_e2 = st.columns([1, 2])
        
        with col_e1:
            st.subheader("Schema Metadata")
            schema_df = pd.DataFrame({
                "Data Type": eda_res['dtypes'],
                "Missing Values": eda_res['missing_values']
            })
            st.dataframe(schema_df, use_container_width=True)
            
        with col_e2:
            st.subheader("Statistical Characteristics (Numerical)")
            if eda_res['describe_numeric']:
                st.dataframe(pd.DataFrame(eda_res['describe_numeric']), use_container_width=True)
            else:
                st.info("No numeric columns available in the dataset.")
                
        st.markdown("---")
        
        col_e3, col_e4 = st.columns(2)
        with col_e3:
            st.subheader("Categorical Features Breakdowns")
            if eda_res['categorical_top_values']:
                cat_col = st.selectbox("Select Categorical Feature", list(eda_res['categorical_top_values'].keys()))
                dist_df = pd.DataFrame(
                    eda_res['categorical_top_values'][cat_col].items(), 
                    columns=["Category", "Occurrences"]
                )
                st.dataframe(dist_df, use_container_width=True)
            else:
                st.info("No categorical columns available.")
                
        with col_e4:
            st.subheader("Correlation Matrix")
            if eda_res['correlation_matrix']:
                st.dataframe(pd.DataFrame(eda_res['correlation_matrix']), use_container_width=True)
            else:
                st.info("Insufficient numeric columns for correlation matrix.")
                
    # ------------------ TAB 3: INTERACTIVE DASHBOARD ------------------
    with tab_dashboard:
        st.header("📈 Interactive Dashboard")
        
        # Display auto KPIs
        def auto_kpi_cards(df):
            cols = df.columns
            sales_col = next((c for c in cols if any(k in c.lower() for k in ['sales', 'revenue', 'turnover'])), None)
            profit_col = next((c for c in cols if 'profit' in c.lower()), None)
            sat_col = next((c for c in cols if any(k in c.lower() for k in ['satisfaction', 'rating', 'sat'])), None)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df):,}</div><div class='metric-label'>Total Observations</div></div>", unsafe_allow_html=True)
            with c2:
                if sales_col and pd.api.types.is_numeric_dtype(df[sales_col]):
                    val = df[sales_col].sum()
                    lbl = sales_col.replace('_', ' ').title()
                    fmt = f"${val/1e6:.2f}M" if val >= 1e6 else (f"${val/1e3:.1f}K" if val >= 1e3 else f"${val:.2f}")
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{fmt}</div><div class='metric-label'>Total {lbl}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(cols)}</div><div class='metric-label'>Total Attributes</div></div>", unsafe_allow_html=True)
            with c3:
                if profit_col and pd.api.types.is_numeric_dtype(df[profit_col]):
                    val = df[profit_col].sum()
                    lbl = profit_col.replace('_', ' ').title()
                    fmt = f"${val/1e6:.2f}M" if val >= 1e6 else (f"${val/1e3:.1f}K" if val >= 1e3 else f"${val:.2f}")
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{fmt}</div><div class='metric-label'>Total {lbl}</div></div>", unsafe_allow_html=True)
                else:
                    num_cnt = len(df.select_dtypes(include=[np.number]).columns)
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{num_cnt}</div><div class='metric-label'>Numeric Features</div></div>", unsafe_allow_html=True)
            with c4:
                if sat_col and pd.api.types.is_numeric_dtype(df[sat_col]):
                    val = df[sat_col].mean()
                    lbl = sat_col.replace('_', ' ').title()
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{val:.2f} / 5.0</div><div class='metric-label'>Avg {lbl}</div></div>", unsafe_allow_html=True)
                else:
                    cat_cnt = len(df.select_dtypes(exclude=[np.number]).columns)
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{cat_cnt}</div><div class='metric-label'>Categorical Features</div></div>", unsafe_allow_html=True)

        auto_kpi_cards(df_active)
        st.write("")
        
        # Interactive chart generator settings
        col_d1, col_d2 = st.columns([1, 3])
        
        numeric_cols = list(df_active.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(df_active.select_dtypes(exclude=[np.number]).columns)
        
        with col_d1:
            st.subheader("Chart Builder Options")
            chart_type = st.selectbox(
                "Select Visualization Type",
                ["Histogram Distribution", "Outlier Boxplot", "Correlation Heatmap", "Pie Breakdown", "Bar Comparison", "Monthly Trend"]
            )
            
            # Show options depending on chart
            fig = None
            if chart_type == "Histogram Distribution":
                if numeric_cols:
                    x_col = st.selectbox("Select Numeric Column", numeric_cols)
                    fig = visualizer.plot_plotly_histogram(df_active, x_col)
                else:
                    st.warning("No numeric columns found.")
            
            elif chart_type == "Outlier Boxplot":
                if numeric_cols:
                    y_col = st.selectbox("Select Numeric Column", numeric_cols)
                    fig = visualizer.plot_plotly_boxplot(df_active, y_col)
                else:
                    st.warning("No numeric columns found.")
                    
            elif chart_type == "Correlation Heatmap":
                if len(numeric_cols) >= 2:
                    fig = visualizer.plot_plotly_heatmap(df_active)
                else:
                    st.warning("Insufficient numeric columns for correlation matrix.")
                    
            elif chart_type == "Pie Breakdown":
                if categorical_cols:
                    cat_col = st.selectbox("Select Categorical Column", categorical_cols)
                    fig = visualizer.plot_plotly_pie(df_active, cat_col)
                else:
                    st.warning("No categorical columns found.")
                    
            elif chart_type == "Bar Comparison":
                if categorical_cols:
                    cat_col = st.selectbox("Select X-Axis Category", categorical_cols)
                    agg_type = st.selectbox("Select Aggregation", ["Frequency Count", "Sum of Value", "Mean (Average) of Value"])
                    
                    if agg_type == "Frequency Count":
                        fig = visualizer.plot_plotly_bar(df_active, cat_col)
                    else:
                        if numeric_cols:
                            num_col = st.selectbox("Select Value Column", numeric_cols)
                            agg_func = "sum" if agg_type == "Sum of Value" else "mean"
                            fig = visualizer.plot_plotly_bar(df_active, cat_col, num_col, agg_func)
                        else:
                            st.warning("No numeric column for aggregation.")
                else:
                    st.warning("No categorical columns found.")
                    
            elif chart_type == "Monthly Trend":
                date_col = visualizer.detect_datetime_column(df_active)
                if date_col and numeric_cols:
                    val_col = st.selectbox("Select Value Column", numeric_cols)
                    fig = visualizer.plot_plotly_trend(df_active, date_col, val_col)
                elif numeric_cols:
                    st.warning("No date column detected. Try Plotting a rolling sequence index.")
                    val_col = st.selectbox("Select Value Column", numeric_cols)
                    fig = px.line(df_active.head(150), y=val_col, title=f"Observation Trend: {val_col}")
                else:
                    st.warning("No numeric columns found.")
                    
        with col_d2:
            st.subheader("Live Dynamic Chart")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Configure variables to build chart.")
                
    # ------------------ TAB 4: FORECASTING ------------------
    with tab_forecast:
        st.header("🔮 Business Trend Projections")
        st.write("Leverage regression mechanics to predict subsequent period metrics based on historic patterns.")
        
        if not numeric_cols:
            st.warning("Forecasting requires numerical columns.")
        else:
            date_col = visualizer.detect_datetime_column(df_active)
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                target_col = st.selectbox("Select Forecast Target Column", numeric_cols, key="fc_target")
            with col_f2:
                periods = st.slider("Periods to Forecast Ahead", min_value=3, max_value=24, value=6, key="fc_periods")
                
            if date_col:
                st.info(f"Timeline detected: `{date_col}`. Collating transactions monthly to compute trend.")
                # Aggregate
                temp_df = df_active.copy()
                temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
                temp_df = temp_df.dropna(subset=[date_col]).sort_values(date_col)
                
                monthly = temp_df.groupby(temp_df[date_col].dt.to_period("M"))[target_col].mean().reset_index()
                monthly[date_col] = monthly[date_col].dt.to_timestamp()
                
                if len(monthly) < 4:
                    st.warning("Insufficient timeline length. Weekly or monthly observations must exceed 4 periods.")
                else:
                    # Sklearn
                    X = np.arange(len(monthly)).reshape(-1, 1)
                    y = monthly[target_col].values
                    
                    reg = LinearRegression().fit(X, y)
                    hist_trend = reg.predict(X)
                    
                    fut_idx = np.arange(len(monthly), len(monthly) + periods).reshape(-1, 1)
                    fut_trend = reg.predict(fut_idx)
                    
                    last_d = monthly[date_col].iloc[-1]
                    fut_dates = [last_d + pd.DateOffset(months=i) for i in range(1, periods + 1)]
                    
                    # Graph
                    fc_fig = go.Figure()
                    # Actual
                    fc_fig.add_trace(go.Scatter(x=monthly[date_col], y=y, mode="lines+markers", name="Historical Sales", line=dict(color="#1e293b", width=2)))
                    # Trend line
                    fc_fig.add_trace(go.Scatter(x=monthly[date_col], y=hist_trend, mode="lines", name="Historical Fit Line", line=dict(color="#18bc9c", width=1.5, dash="dash")))
                    # Future
                    fc_fig.add_trace(go.Scatter(x=fut_dates, y=fut_trend, mode="lines+markers", name="Projected Value", line=dict(color="#e74c3c", width=2)))
                    
                    fc_fig.update_layout(
                        title=f"Trend Projections: {target_col.replace('_', ' ').title()}",
                        xaxis_title="Timeline",
                        yaxis_title=target_col.replace('_', ' ').title(),
                        template="plotly_white"
                    )
                    st.plotly_chart(fc_fig, use_container_width=True)
                    
                    # Values dataframe
                    fc_df = pd.DataFrame({
                        "Forecast Timeline": [d.strftime('%Y-%B') for d in fut_dates],
                        "Projected Estimate": fut_trend
                    })
                    st.dataframe(fc_df, use_container_width=True)
            else:
                st.info("No datetime fields detected. Forecasting based on observation row sequence indexes.")
                y = df_active[target_col].values
                if len(y) < 10:
                    st.warning("Insufficient sequence length (Need at least 10 rows).")
                else:
                    y_subset = y[:150] # Take first 150 rows
                    X = np.arange(len(y_subset)).reshape(-1, 1)
                    
                    reg = LinearRegression().fit(X, y_subset)
                    hist_trend = reg.predict(X)
                    fut_idx = np.arange(len(y_subset), len(y_subset) + periods).reshape(-1, 1)
                    fut_trend = reg.predict(fut_idx)
                    
                    fc_fig = go.Figure()
                    fc_fig.add_trace(go.Scatter(x=list(range(len(y_subset))), y=y_subset, mode="lines", name="History", line=dict(color="#1e293b")))
                    fc_fig.add_trace(go.Scatter(x=list(range(len(y_subset))), y=hist_trend, mode="lines", name="Fit Line", line=dict(color="#18bc9c", dash="dash")))
                    fc_fig.add_trace(go.Scatter(x=list(range(len(y_subset), len(y_subset) + periods)), y=fut_trend, mode="lines+markers", name="Projected Forecast", line=dict(color="#e74c3c")))
                    
                    fc_fig.update_layout(title="Sequence Series Projection", template="plotly_white")
                    st.plotly_chart(fc_fig, use_container_width=True)
                    
                    fc_df = pd.DataFrame({
                        "Observation Index": list(range(len(y_subset), len(y_subset) + periods)),
                        "Projected Estimate": fut_trend
                    })
                    st.dataframe(fc_df, use_container_width=True)
                    
    # ------------------ TAB 5: AI INSIGHTS ------------------
    with tab_ai:
        st.header("🤖 AI Executive Insights Generator")
        st.write("Generates business findings, opportunities, risks, and strategic operational suggestions using OpenAI/Gemini models.")
        
        # Check API key presence
        if not api_key:
            st.warning("Please configure your API Key in the sidebar to activate the AI Insights module.")
        else:
            col_a1, col_a2 = st.columns([1, 4])
            with col_a1:
                run_ai = st.button("🧠 Generate Insights Report", type="primary")
            with col_a2:
                st.write("")
                
            if run_ai:
                with st.spinner("Processing dataset summaries and sending data descriptors to AI Engine..."):
                    try:
                        # Prepare eda summary report
                        raw_summary = eda.get_eda_summary_text(
                            eda_results=st.session_state.eda_results,
                            cleaning_log=st.session_state.cleaning_log
                        )
                        insights = ai_insights.generate_business_insights(
                            eda_summary=raw_summary,
                            provider=api_provider,
                            api_key=api_key,
                            model_name=api_model
                        )
                        st.session_state.ai_insights = insights
                        st.success("AI Insights generated successfully!")
                    except Exception as e:
                        st.error(f"API Connection Failed: {str(e)}")
                        
            if st.session_state.ai_insights:
                st.markdown("---")
                st.subheader("💡 Strategic Insights Report")
                st.markdown(st.session_state.ai_insights)
                
    # ------------------ TAB 6: CHAT WITH DATA ------------------
    with tab_chat:
        st.header("💬 Interactive Query Engine (Chat with Dataset)")
        st.write("Ask natural language questions about the dataset. The AI will translate them into Pandas logic, execute the code, and render text responses and Plotly figures.")
        
        if not api_key:
            st.warning("Please configure your API Key in the sidebar to activate the Chat interface.")
        else:
            # Display chat messages
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "fig" in message and message["fig"] is not None:
                        st.plotly_chart(message["fig"], use_container_width=True)
                    if "code" in message and message["code"]:
                        with st.expander("Show Executed python code"):
                            st.code(message["code"], language="python")
                            
            # Chat Input
            user_input = st.chat_input("Ask: 'What is the correlation between sales and profit?' or 'Show me a bar chart of sales by department'")
            
            if user_input:
                # Add to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing question, generating pandas code, executing..."):
                        res = ai_insights.chat_with_dataset(
                            df=df_active,
                            user_query=user_input,
                            provider=api_provider,
                            api_key=api_key,
                            model_name=api_model
                        )
                        
                        if res["success"]:
                            st.write(res["result_text"])
                            if res["result_fig"] is not None:
                                st.plotly_chart(res["result_fig"], use_container_width=True)
                            with st.expander("Show Executed python code"):
                                st.code(res["code"], language="python")
                                
                            # Save to state
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": res["result_text"],
                                "fig": res["result_fig"],
                                "code": res["code"]
                            })
                        else:
                            err_msg = f"Sorry, there was an execution error with the code compiled for this query:\n\n`{res['error']}`"
                            st.error(err_msg)
                            if "code" in res and res["code"]:
                                with st.expander("Show Compiled Code"):
                                    st.code(res["code"], language="python")
                                    
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": err_msg,
                                "code": res.get("code", "")
                            })
                            
    # ------------------ TAB 7: EXPORT REPORT ------------------
    with tab_report:
        st.header("📄 Export Executive PDF Report")
        st.write("Compile data load metrics, cleaning logs, descriptive statistics, static visualizations, and generated AI Insights into a high-end corporate PDF document.")
        
        if st.session_state.ai_insights is None:
            st.warning("Please generate AI Insights first (in the 'AI Insights' tab) before exporting the report. The PDF requires the executive brief to compile.")
        else:
            st.info("Click the button below to assemble the PDF report. This will compile all statistics and save static PNG visualizations for the report story.")
            compile_report = st.button("🔨 Compile Executive Report", type="primary")
            
            if compile_report:
                with st.spinner("Generating static charts & assembling PDF elements..."):
                    try:
                        # 1. Save static charts to disk
                        chart_paths = visualizer.save_static_charts(df_active, CHARTS_DIR)
                        
                        # 2. Assemble PDF path
                        pdf_path = os.path.join(REPORTS_DIR, "executive_report.pdf")
                        
                        # 3. Build report
                        report_generator.build_pdf_report(
                            output_pdf_path=pdf_path,
                            dataset_name=st.session_state.loaded_filename,
                            cleaning_log=st.session_state.cleaning_log if st.session_state.cleaning_log else {},
                            eda_results=st.session_state.eda_results,
                            ai_insights=st.session_state.ai_insights,
                            chart_paths=chart_paths
                        )
                        
                        st.success("Executive PDF Report compiled successfully!")
                        
                        # Provide download link
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download Executive Report (PDF)",
                                data=pdf_file,
                                file_name=f"Executive_Report_{st.session_state.loaded_filename.split('.')[0]}.pdf",
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.error(f"Failed to generate report: {str(e)}")
else:
    # Landing / Welcome Page
    st.markdown("<div style='text-align: center; margin-top: 50px;'>", unsafe_allow_html=True)
    st.markdown("## Welcome to the AI Executive Data Analyst Workspace 🤖")
    st.write("Upload a CSV or Excel spreadsheet in the sidebar or load the demo dataset to begin your analysis.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Showcase features
    st.write("")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.info("🧼 **Sanitization Pipeline**\nAutomatically maps features to standard casing, removes duplicated records, and safely fills empty variables with median or mode averages.")
    with col_w2:
        st.info("📈 **Dynamic Dashboards**\nInspect data trends using interactive Plotly interfaces, run statistical tests, and forecast metrics without C++ compiler errors.")
    with col_w3:
        st.info("🧠 **AI Business Brain**\nSend structured summaries to OpenAI/Gemini models, chat with your dataframe interactively, and build executive PDF reports.")
