import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, Any

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the dataset:
    1. Standardize column names (lowercase, replace spaces/hyphens with underscores, remove non-alphanumeric).
    2. Detect and fill missing values (median for numeric, mode for categorical).
    3. Remove duplicate rows.
    
    Parameters:
    - df (pd.DataFrame): Input raw DataFrame.
    
    Returns:
    - Tuple[pd.DataFrame, Dict[str, Any]]: The cleaned DataFrame and a logging summary dictionary.
    """
    cleaned_df = df.copy()
    original_shape = df.shape
    
    # 1. Standardize column names
    columns_renamed = {}
    new_cols = []
    for col in cleaned_df.columns:
        # Convert to string, lowercase, strip whitespace
        new_col = str(col).strip().lower()
        # Replace spaces, hyphens, slashes with a single underscore
        new_col = re.sub(r'[\s\-\/]+', '_', new_col)
        # Remove non-alphanumeric characters except underscores
        new_col = re.sub(r'[^\w]', '', new_col)
        # Replace consecutive underscores
        new_col = re.sub(r'_+', '_', new_col)
        # Strip leading/trailing underscores
        new_col = new_col.strip('_')
        
        # If blank, name it 'column'
        if not new_col:
            new_col = "column"
            
        # Avoid duplicate column name collisions
        base_col = new_col
        counter = 1
        while new_col in new_cols:
            new_col = f"{base_col}_{counter}"
            counter += 1
            
        columns_renamed[col] = new_col
        new_cols.append(new_col)
        
    cleaned_df.columns = new_cols
    
    # Map raw missing value counts to standardized column names for reference
    missing_before = df.isnull().sum().to_dict()
    missing_before_std = {columns_renamed[k]: v for k, v in missing_before.items()}
    
    # 2. Fill missing values
    missing_numeric_filled = {}
    missing_categorical_filled = {}
    
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
    
    # Fill numeric columns with median
    for col in numeric_cols:
        null_count = cleaned_df[col].isnull().sum()
        if null_count > 0:
            median_val = cleaned_df[col].median()
            # Fallback if column is entirely NaN
            if pd.isna(median_val):
                median_val = 0.0
            
            # Keep as regular python type (float/int) for JSON/summary readability
            if isinstance(median_val, (np.integer, np.floating)):
                median_val = median_val.item()
                
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            missing_numeric_filled[col] = median_val
            
    # Fill categorical columns with mode
    for col in categorical_cols:
        null_count = cleaned_df[col].isnull().sum()
        if null_count > 0:
            mode_series = cleaned_df[col].mode()
            if not mode_series.empty:
                mode_val = mode_series.iloc[0]
            else:
                mode_val = "Unknown"
                
            if isinstance(mode_val, (np.integer, np.floating)):
                mode_val = mode_val.item()
            else:
                mode_val = str(mode_val)
                
            cleaned_df[col] = cleaned_df[col].fillna(mode_val)
            missing_categorical_filled[col] = mode_val
            
    # 3. Remove duplicate rows
    duplicates_count = cleaned_df.duplicated().sum()
    if duplicates_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        
    new_shape = cleaned_df.shape
    
    log = {
        "original_shape": original_shape,
        "new_shape": new_shape,
        "columns_renamed": columns_renamed,
        "missing_before": missing_before_std,
        "missing_numeric_filled": missing_numeric_filled,
        "missing_categorical_filled": missing_categorical_filled,
        "duplicates_removed": int(duplicates_count)
    }
    
    return cleaned_df, log
