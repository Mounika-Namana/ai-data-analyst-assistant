import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shutil
from typing import Dict, Optional, List

# Configure Matplotlib styles for clean look
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def clean_charts_directory(charts_dir: str):
    """
    Safely purges any files in the charts directory to prepare for a new run.
    """
    if os.path.exists(charts_dir):
        try:
            shutil.rmtree(charts_dir)
        except Exception:
            pass
    os.makedirs(charts_dir, exist_ok=True)

def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Analyze columns and search for potential date/time fields.
    """
    # 1. Check actual datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
            
    # 2. Heuristically check columns with datetime keywords and try conversion
    date_keywords = ['date', 'time', 'timestamp', 'year', 'month', 'day']
    for col in df.columns:
        if any(kw in col.lower() for kw in date_keywords):
            sample = df[col].dropna().head(10)
            if not sample.empty:
                try:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.isnull().sum() / len(parsed) < 0.3:
                        return col
                except Exception:
                    pass
    return None

def save_static_charts(df: pd.DataFrame, charts_dir: str) -> Dict[str, str]:
    """
    Generates and saves static PNG charts for ReportLab PDF integration.
    Returns:
    - Dict[str, str]: A dictionary mapping chart names to their file paths.
    """
    clean_charts_directory(charts_dir)
    generated_charts = {}
    
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    
    # Use clean, professional color palettes
    sns.set_theme(style="whitegrid")
    
    # 1. Histogram (Data Distribution)
    if numeric_cols:
        col = numeric_cols[0]
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, color='#2c3e50', bins=30)
        plt.title(f"Distribution of {col.replace('_', ' ').title()}", pad=15)
        plt.xlabel(col.replace('_', ' ').title())
        plt.ylabel("Frequency")
        plt.tight_layout()
        path = os.path.join(charts_dir, "histogram.png")
        plt.savefig(path, dpi=150)
        plt.close()
        generated_charts["histogram"] = path
        
    # 2. Box Plot (Outlier Analysis)
    if numeric_cols:
        col = numeric_cols[0]
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col], color='#18bc9c')
        plt.title(f"Outlier Box Plot: {col.replace('_', ' ').title()}", pad=15)
        plt.xlabel(col.replace('_', ' ').title())
        plt.tight_layout()
        path = os.path.join(charts_dir, "boxplot.png")
        plt.savefig(path, dpi=150)
        plt.close()
        generated_charts["boxplot"] = path
        
    # 3. Correlation Heatmap (Feature Relationships)
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
        plt.title("Correlation Matrix Heatmap", pad=15)
        plt.tight_layout()
        path = os.path.join(charts_dir, "heatmap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        generated_charts["heatmap"] = path
        
    # 4. Pie Chart (Categorical Distribution)
    if categorical_cols:
        col = categorical_cols[0]
        val_counts = df[col].value_counts().head(8)
        if not val_counts.empty:
            plt.figure(figsize=(6, 5))
            colors = sns.color_palette("pastel", len(val_counts))
            plt.pie(val_counts, labels=val_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
            plt.title(f"Category breakdown: {col.replace('_', ' ').title()}", pad=15)
            plt.tight_layout()
            path = os.path.join(charts_dir, "pie_chart.png")
            plt.savefig(path, dpi=150)
            plt.close()
            generated_charts["pie_chart"] = path
            
    # 5. Bar Chart (Category Comparisons)
    if categorical_cols:
        cat_col = categorical_cols[0]
        plt.figure(figsize=(8, 4.5))
        if numeric_cols:
            num_col = numeric_cols[0]
            grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
            sns.barplot(x=grouped.values, y=grouped.index, palette='viridis', hue=grouped.index, legend=False)
            plt.title(f"Average {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}", pad=15)
            plt.xlabel(f"Mean {num_col.replace('_', ' ').title()}")
        else:
            grouped = df[cat_col].value_counts().head(10)
            sns.barplot(x=grouped.values, y=grouped.index, palette='magma', hue=grouped.index, legend=False)
            plt.title(f"Frequency Count of {cat_col.replace('_', ' ').title()} (Top 10)", pad=15)
            plt.xlabel("Occurrences")
        plt.ylabel(cat_col.replace('_', ' ').title())
        plt.tight_layout()
        path = os.path.join(charts_dir, "bar_chart.png")
        plt.savefig(path, dpi=150)
        plt.close()
        generated_charts["bar_chart"] = path
        
    # 6. Trend Line Chart (Time Series or Sequential data)
    date_col = detect_datetime_column(df)
    if date_col and numeric_cols:
        num_col = numeric_cols[0]
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
        temp_df = temp_df.dropna(subset=[date_col]).sort_values(date_col)
        
        # Monthly aggregates
        grouped = temp_df.groupby(temp_df[date_col].dt.to_period("M"))[num_col].mean()
        if len(grouped) > 1:
            plt.figure(figsize=(8.5, 4))
            grouped.plot(kind='line', marker='o', color='#e74c3c', linewidth=2)
            plt.title(f"Trend Analysis: Mean {num_col.replace('_', ' ').title()} Over Time", pad=15)
            plt.xlabel("Timeline (Monthly)")
            plt.ylabel(num_col.replace('_', ' ').title())
            plt.xticks(rotation=30)
            plt.tight_layout()
            path = os.path.join(charts_dir, "trend_chart.png")
            plt.savefig(path, dpi=150)
            plt.close()
            generated_charts["trend_chart"] = path
            
    # Fallback to Index Trend Chart if no date column exists
    if "trend_chart" not in generated_charts and numeric_cols:
        num_col = numeric_cols[0]
        plt.figure(figsize=(8.5, 4))
        subset = df.head(100) # Plot first 100 observations to keep it uncluttered
        plt.plot(subset.index, subset[num_col], color='#3498db', alpha=0.5, label='Actual Value')
        if len(subset) > 10:
            rolling_w = min(10, len(subset) // 5)
            rolling = subset[num_col].rolling(window=rolling_w, min_periods=1).mean()
            plt.plot(subset.index, rolling, color='#e74c3c', linewidth=2, label='Smoothed Trend')
        plt.title(f"Observation Sequence Trend: {num_col.replace('_', ' ').title()} (First 100 Rows)", pad=15)
        plt.xlabel("Row Index")
        plt.ylabel(num_col.replace('_', ' ').title())
        plt.legend(frameon=True)
        plt.tight_layout()
        path = os.path.join(charts_dir, "trend_chart.png")
        plt.savefig(path, dpi=150)
        plt.close()
        generated_charts["trend_chart"] = path
        
    return generated_charts

# Plotly functions for Streamlit GUI Renderings
def plot_plotly_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(
        df, x=col, marginal="box", 
        title=f"Distribution of {col.replace('_', ' ').title()}",
        color_discrete_sequence=['#2c3e50'],
        template="plotly_white"
    )
    fig.update_layout(bargap=0.03, hovermode='x')
    return fig

def plot_plotly_boxplot(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.box(
        df, y=col, points="outliers",
        title=f"Box Plot and Outliers for {col.replace('_', ' ').title()}",
        color_discrete_sequence=['#18bc9c'],
        template="plotly_white"
    )
    return fig

def plot_plotly_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        raise ValueError("No numerical columns found for correlation heatmap.")
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Heatmap Matrix",
        template="plotly_white"
    )
    return fig

def plot_plotly_pie(df: pd.DataFrame, col: str) -> go.Figure:
    val_counts = df[col].value_counts().reset_index()
    val_counts.columns = [col, 'count']
    if len(val_counts) > 10:
        other_sum = val_counts.iloc[9:]['count'].sum()
        val_counts = val_counts.head(9)
        other_row = pd.DataFrame([{col: 'Other Categories', 'count': other_sum}])
        val_counts = pd.concat([val_counts, other_row], ignore_index=True)
        
    fig = px.pie(
        val_counts, names=col, values='count',
        title=f"Category Distribution: {col.replace('_', ' ').title()}",
        color_discrete_sequence=px.colors.qualitative.Safe,
        template="plotly_white"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_plotly_bar(df: pd.DataFrame, cat_col: str, num_col: Optional[str] = None, agg_func: str = "count") -> go.Figure:
    if num_col is None or agg_func == "count":
        grouped = df[cat_col].value_counts().reset_index()
        grouped.columns = [cat_col, 'count']
        grouped = grouped.sort_values(by='count', ascending=False).head(15)
        fig = px.bar(
            grouped, x=cat_col, y='count',
            title=f"Frequency Count of {cat_col.replace('_', ' ').title()} (Top 15)",
            color=cat_col, color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white"
        )
        fig.update_layout(showlegend=False)
    else:
        if agg_func == "mean":
            grouped = df.groupby(cat_col)[num_col].mean().reset_index()
            title = f"Average {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}"
        elif agg_func == "sum":
            grouped = df.groupby(cat_col)[num_col].sum().reset_index()
            title = f"Total {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}"
        else:
            grouped = df.groupby(cat_col)[num_col].mean().reset_index()
            title = f"Mean {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}"
            
        grouped = grouped.sort_values(by=num_col, ascending=False).head(15)
        fig = px.bar(
            grouped, x=cat_col, y=num_col,
            title=title, color=cat_col,
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white"
        )
        fig.update_layout(showlegend=False)
    return fig

def plot_plotly_trend(df: pd.DataFrame, date_col: str, val_col: str) -> go.Figure:
    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
    temp_df = temp_df.dropna(subset=[date_col]).sort_values(date_col)
    
    # Group monthly to make a clean trend line
    grouped = temp_df.groupby(temp_df[date_col].dt.to_period("M"))[val_col].mean().reset_index()
    grouped[date_col] = grouped[date_col].astype(str)
    
    fig = px.line(
        grouped, x=date_col, y=val_col, markers=True,
        title=f"Monthly Trend: Average {val_col.replace('_', ' ').title()} Over Time",
        color_discrete_sequence=['#e74c3c'],
        template="plotly_white"
    )
    fig.update_xaxes(title="Month Timeline")
    fig.update_yaxes(title=val_col.replace('_', ' ').title())
    return fig
