import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def generate_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate Exploratory Data Analysis (EDA) metrics for a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The cleaned DataFrame.
    
    Returns:
    - Dict[str, Any]: A dictionary containing statistical characteristics of the dataset.
    """
    shape = df.shape
    columns = list(df.columns)
    
    # Column datatypes as strings
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Missing values
    missing_values = df.isnull().sum().to_dict()
    
    # Segregate columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Numeric description
    describe_numeric = {}
    if not numeric_cols.empty:
        describe_numeric = df[numeric_cols].describe().to_dict()
        
    # Categorical description
    describe_categorical = {}
    if not categorical_cols.empty:
        describe_categorical = df[categorical_cols].describe().to_dict()
        
    # Correlation Matrix
    correlation_matrix = {}
    if len(numeric_cols) >= 2:
        correlation_matrix = df[numeric_cols].corr().round(4).to_dict()
        
    # Unique value distributions for categorical columns
    categorical_top_values = {}
    for col in categorical_cols:
        val_counts = df[col].value_counts().head(5).to_dict()
        categorical_top_values[col] = val_counts
        
    return {
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing_values,
        "describe_numeric": describe_numeric,
        "describe_categorical": describe_categorical,
        "correlation_matrix": correlation_matrix,
        "categorical_top_values": categorical_top_values
    }

def format_dict_to_markdown_table(data_dict: Dict[str, Dict[str, Any]]) -> str:
    """
    Safely converts a nested statistics dictionary (e.g. describe()) 
    into a Markdown table without requiring external packages like tabulate.
    """
    if not data_dict:
        return "*No data available.*"
        
    cols = list(data_dict.keys())
    # Identify unique row indices (e.g., 'count', 'mean', 'std', 'min', etc.)
    row_keys = list(data_dict[cols[0]].keys())
    
    # Header
    header = "| Metric | " + " | ".join(cols) + " |"
    divider = "| :--- | " + " | ".join([":---:"] * len(cols)) + " |"
    
    # Rows
    rows = []
    for key in row_keys:
        row_cells = []
        for col in cols:
            val = data_dict[col].get(key, np.nan)
            if isinstance(val, float):
                row_cells.append(f"{val:.3f}")
            elif isinstance(val, (int, np.integer)):
                row_cells.append(str(val))
            else:
                row_cells.append(str(val))
        rows.append(f"| {key} | " + " | ".join(row_cells) + " |")
        
    return "\n".join([header, divider] + rows)

def get_eda_summary_text(eda_results: Dict[str, Any], cleaning_log: Optional[Dict[str, Any]] = None) -> str:
    """
    Convert the EDA results and data cleaning logs into a structured text report.
    This serves as the raw material for the AI Insight Engine.
    
    Parameters:
    - eda_results (Dict[str, Any]): Outputs from generate_eda.
    - cleaning_log (Optional[Dict[str, Any]]): Log of cleaning operations.
    
    Returns:
    - str: A clean, markdown-formatted text summary.
    """
    summary = []
    summary.append("# DATASET SUMMARY REPORT\n")
    
    summary.append("## 1. File Metadata")
    summary.append(f"- **Number of Rows**: {eda_results['shape'][0]}")
    summary.append(f"- **Number of Columns**: {eda_results['shape'][1]}")
    
    if cleaning_log:
        summary.append("\n## 2. Data Cleaning Log")
        summary.append(f"- **Duplicate Rows Removed**: {cleaning_log.get('duplicates_removed', 0)}")
        
        filled_num = cleaning_log.get('missing_numeric_filled', {})
        filled_cat = cleaning_log.get('missing_categorical_filled', {})
        
        if filled_num or filled_cat:
            summary.append("- **Missing Values Handled**:")
            for col, val in filled_num.items():
                summary.append(f"  - Filled numerical column `{col}` with median: `{val}`")
            for col, val in filled_cat.items():
                summary.append(f"  - Filled categorical column `{col}` with mode: `{val}`")
        else:
            summary.append("- **Missing Values**: None detected or already filled.")
            
    summary.append("\n## 3. Data Schema & Columns")
    for col, dtype in eda_results['dtypes'].items():
        missing = eda_results['missing_values'].get(col, 0)
        summary.append(f"- `{col}` (type: {dtype}) | Missing count: {missing}")
        
    if eda_results['describe_numeric']:
        summary.append("\n## 4. Descriptive Statistics (Numerical Columns)")
        summary.append(format_dict_to_markdown_table(eda_results['describe_numeric']))
        
    if eda_results['categorical_top_values']:
        summary.append("\n## 5. Categorical Columns Distributions (Top 5 Value Counts)")
        for col, counts in eda_results['categorical_top_values'].items():
            summary.append(f"\n### Column: `{col}`")
            for val, count in counts.items():
                summary.append(f"- `{val}`: {count} occurrences")
                
    if eda_results['correlation_matrix']:
        summary.append("\n## 6. Correlation Matrix (Numerical Columns)")
        summary.append(format_dict_to_markdown_table(eda_results['correlation_matrix']))
        
    return "\n".join(summary)
