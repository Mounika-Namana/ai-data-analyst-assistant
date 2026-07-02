import pandas as pd
import numpy as np
import io
import sys
import traceback
from typing import Dict, Any, Optional, Tuple, List

# Providers and models selection helpers
PROVIDERS = {
    "Google Gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
}

def call_llm(
    prompt: str,
    system_prompt: str,
    provider: str,
    api_key: str,
    model_name: str
) -> str:
    """
    Helper function to dispatch API requests to Gemini or OpenAI.
    """
    if not api_key:
        raise ValueError("API Key is required to generate insights. Please enter your API key in the sidebar.")
        
    if provider == "Google Gemini":
        import google.generativeai as genai
        try:
            genai.configure(api_key=api_key)
            # Use safety settings to prevent blocks on financial data or custom charts
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Google Gemini API error: {str(e)}")
            
    elif provider == "OpenAI":
        from openai import OpenAI
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    else:
        raise ValueError(f"Unknown API provider: {provider}")

def generate_business_insights(
    eda_summary: str,
    provider: str,
    api_key: str,
    model_name: str
) -> str:
    """
    Generate strategic business analysis from the EDA summary report.
    """
    system_prompt = (
        "You are a Senior Business Analyst and Executive Advisor. "
        "Your goal is to extract deep, actionable, data-driven business insights from statistical summaries. "
        "Be professional, precise, and strategic."
    )
    
    user_prompt = f"""
Analyze the following dataset summary and provide a comprehensive corporate analysis report.

Provide details under these EXACT sections:
1. Executive Summary (Brief 2-3 sentence overview of the dataset context)
2. Key Findings (Identify trends, anomalies, outliers, or significant correlations)
3. Risks & Warnings (Data gaps, outliers, risk points, negative correlations)
4. Business Opportunities (Revenue growth, cost-saving, process improvement suggestions)
5. Actionable Recommendations (Concrete, prioritized next steps)

Write in clean markdown without meta-chatter. Keep the style executive-ready.

---
Dataset Summary:
{eda_summary}
"""
    return call_llm(user_prompt, system_prompt, provider, api_key, model_name)

def generate_query_code(
    df: pd.DataFrame,
    user_query: str,
    provider: str,
    api_key: str,
    model_name: str
) -> str:
    """
    Ask the LLM to write a Python pandas snippet to answer the user query.
    """
    # Sample the data to show format
    sample_data = df.head(3).to_string()
    
    system_prompt = (
        "You are an expert Python Data Analyst. Your job is to write code using pandas, numpy, and plotly "
        "to answer user data questions. Respond ONLY with executable python code inside a markdown block. "
        "Do not write any introductory or concluding text."
    )
    
    user_prompt = f"""
We have a pandas DataFrame named `df` loaded in memory.
Write python code to answer the following user query: "{user_query}"

---
## Dataset Metadata:
- Shape: {df.shape}
- Columns and Datatypes: { {col: str(dtype) for col, dtype in df.dtypes.items()} }
- First 3 rows:
{sample_data}

---
## Code Requirements:
1. The DataFrame is named `df`. Do NOT load, create, or modify files from disk.
2. Store your final textual/table answer in a string variable named `result_text`. Use formatting like newlines or tabulate style for tables.
3. If the user's question asks for a chart/visualization (or implies one is needed), create a Plotly figure and store it in the variable `result_fig` (e.g. `result_fig = px.bar(...)`). Import `plotly.express as px` or `plotly.graph_objects as go` if needed. Do NOT call `fig.show()`.
4. Ensure the python code is wrapped in a standard markdown block:
```python
# your code here
```
5. Handle NaN values or exceptions gracefully inside the code if necessary.
"""
    return call_llm(user_prompt, system_prompt, provider, api_key, model_name)

def execute_generated_code(df: pd.DataFrame, code_markdown: str) -> Dict[str, Any]:
    """
    Cleans and executes LLM-generated code on the dataframe, returning stdout,
    generated figures, and execution logs.
    """
    # Clean the markdown formatting
    code_lines = code_markdown.strip().split("\n")
    if code_lines[0].startswith("```"):
        code_lines = code_lines[1:]
    if code_lines and code_lines[-1].startswith("```"):
        code_lines = code_lines[:-1]
    cleaned_code = "\n".join(code_lines)
    
    # Execution context
    exec_globals = {
        "pd": pd,
        "np": np,
        "df": df,
        "px": None,  # Will be lazy-imported inside, but make accessible
        "go": None
    }
    exec_locals = {}
    
    # Capture stdout
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    
    try:
        # Import plotly within context for safety
        import plotly.express as px
        import plotly.graph_objects as go
        exec_globals["px"] = px
        exec_globals["go"] = go
        
        # Execute the code
        exec(cleaned_code, exec_globals, exec_locals)
        sys.stdout = old_stdout
        
        # Extract results
        stdout_output = stdout_capture.getvalue()
        result_text = exec_locals.get("result_text", None)
        result_fig = exec_locals.get("result_fig", None)
        
        # Fallback if result_text is not assigned but stdout has print outputs
        if not result_text and stdout_output.strip():
            result_text = stdout_output
        elif not result_text:
            result_text = "Query completed successfully, but no text output was set in `result_text` or printed."
            
        return {
            "success": True,
            "result_text": str(result_text),
            "result_fig": result_fig,
            "code": cleaned_code
        }
        
    except Exception as e:
        sys.stdout = old_stdout
        tb = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": tb,
            "code": cleaned_code
        }

def chat_with_dataset(
    df: pd.DataFrame,
    user_query: str,
    provider: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Wrapper that generates Python code to answer a query, executes it, and handles the output.
    """
    try:
        code_response = generate_query_code(df, user_query, provider, api_key, model_name)
        execution_result = execute_generated_code(df, code_response)
        return execution_result
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse query and execute code: {str(e)}",
            "code": ""
        }
