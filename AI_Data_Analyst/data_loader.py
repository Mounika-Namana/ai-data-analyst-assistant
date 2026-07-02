import pandas as pd
import os
from typing import Union

def load_data(file_path: str, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
    """
    Load data from a CSV or Excel file.
    
    Parameters:
    - file_path (str): Path to the file.
    - sheet_name (Union[str, int]): Sheet name or index (for Excel files only).
    
    Returns:
    - pd.DataFrame: Loaded dataset.
    
    Raises:
    - FileNotFoundError: If the file does not exist.
    - ValueError: For unsupported formats or corrupted/empty files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    filename, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        if ext == '.csv':
            # Try to read with utf-8 first, fallback to latin1 for encoding flexibility
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
            return df
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Only CSV and Excel (.xlsx, .xls) are supported.")
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file contains no data.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse the CSV file. Please check file structure: {str(e)}")
    except Exception as e:
        raise ValueError(f"An error occurred while loading the dataset: {str(e)}")
