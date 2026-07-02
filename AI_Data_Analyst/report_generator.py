import os
import re
import datetime
from typing import Dict, Any, List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def add_header_footer(canvas, doc):
    """
    Draws headers and footers on pages.
    """
    canvas.saveState()
    # Suppress header on cover page (page 1)
    if doc.page > 1:
        canvas.setFont('Helvetica-Bold', 8)
        canvas.setFillColor(colors.HexColor('#2c3e50'))
        canvas.drawString(54, 745, "AI DATA ANALYST - EXECUTIVE REPORT")
        canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
        canvas.setLineWidth(0.5)
        canvas.line(54, 737, 558, 737)
        
    # Footer on all pages
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(54, 36, "Confidential - Automated Business Intelligence Report")
    canvas.drawRightString(558, 36, f"Page {doc.page}")
    canvas.restoreState()

def markdown_to_flowables(md_text: str, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    """
    Parse a basic markdown string and return ReportLab Flowable objects.
    """
    flowables = []
    if not md_text:
        return flowables
        
    lines = md_text.split("\n")
    for line in lines:
        line_str = line.strip()
        if not line_str:
            flowables.append(Spacer(1, 4))
            continue
            
        # Parse titles and headers
        if line_str.startswith("# "):
            flowables.append(Spacer(1, 10))
            flowables.append(Paragraph(line_str[2:], styles["CustomH1"]))
            flowables.append(Spacer(1, 5))
        elif line_str.startswith("## "):
            flowables.append(Spacer(1, 8))
            flowables.append(Paragraph(line_str[3:], styles["CustomH2"]))
            flowables.append(Spacer(1, 4))
        elif line_str.startswith("### "):
            flowables.append(Spacer(1, 6))
            flowables.append(Paragraph(line_str[4:], styles["CustomH3"]))
            flowables.append(Spacer(1, 3))
        # Bullet list items
        elif line_str.startswith("- ") or line_str.startswith("* "):
            content = line_str[2:]
            # Replace markdown bold/italic tags
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
            content = re.sub(r'\*(.*?)\*', r'<i>\1</i>', content)
            flowables.append(Paragraph(content, styles["CustomBullet"]))
        else:
            # Standard paragraph body
            content = line_str
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
            content = re.sub(r'\*(.*?)\*', r'<i>\1</i>', content)
            flowables.append(Paragraph(content, styles["CustomBody"]))
            flowables.append(Spacer(1, 4))
            
    return flowables

def build_pdf_report(
    output_pdf_path: str,
    dataset_name: str,
    cleaning_log: Dict[str, Any],
    eda_results: Dict[str, Any],
    ai_insights: str,
    chart_paths: Dict[str, str]
) -> str:
    """
    Compiles everything into a clean, executive-ready PDF report.
    
    Returns:
    - str: Path to the generated PDF.
    """
    # Create reports directory if it doesn't exist
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    # Page settings: Margins 54pt (0.75in). Printable width = 612 - 108 = 504.
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    # Initialize stylesheets
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor('#2c3e50')
    c_secondary = colors.HexColor('#18bc9c')
    c_text = colors.HexColor('#334155')
    
    # Custom Paragraph Styles
    custom_styles = {
        "CoverTitle": ParagraphStyle(
            'CoverTitle', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=26, leading=32,
            textColor=c_primary, alignment=TA_CENTER
        ),
        "CoverSubtitle": ParagraphStyle(
            'CoverSubtitle', parent=styles['Normal'],
            fontName='Helvetica', fontSize=12, leading=16,
            textColor=c_secondary, alignment=TA_CENTER
        ),
        "CoverMeta": ParagraphStyle(
            'CoverMeta', parent=styles['Normal'],
            fontName='Helvetica', fontSize=9, leading=14,
            textColor=colors.HexColor('#64748b'), alignment=TA_CENTER
        ),
        "CustomH1": ParagraphStyle(
            'CustomH1', parent=styles['Heading1'],
            fontName='Helvetica-Bold', fontSize=18, leading=22,
            textColor=c_primary, keepWithNext=True
        ),
        "CustomH2": ParagraphStyle(
            'CustomH2', parent=styles['Heading2'],
            fontName='Helvetica-Bold', fontSize=13, leading=17,
            textColor=c_secondary, keepWithNext=True
        ),
        "CustomH3": ParagraphStyle(
            'CustomH3', parent=styles['Heading3'],
            fontName='Helvetica-Bold', fontSize=11, leading=15,
            textColor=c_primary, keepWithNext=True
        ),
        "CustomBody": ParagraphStyle(
            'CustomBody', parent=styles['BodyText'],
            fontName='Helvetica', fontSize=9.5, leading=13.5,
            textColor=c_text
        ),
        "CustomBullet": ParagraphStyle(
            'CustomBullet', parent=styles['Normal'],
            fontName='Helvetica', fontSize=9.5, leading=13.5,
            textColor=c_text, leftIndent=15, firstLineIndent=-10
        ),
        "TableHeader": ParagraphStyle(
            'TableHeader', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=9.5, leading=12,
            textColor=colors.white
        ),
        "TableCell": ParagraphStyle(
            'TableCell', parent=styles['Normal'],
            fontName='Helvetica', fontSize=8.5, leading=11,
            textColor=c_text
        )
    }
    
    story = []
    
    # ================= PAGE 1: COVER PAGE =================
    story.append(Spacer(1, 100))
    story.append(Paragraph("AI DATA ANALYST REPORT", custom_styles["CoverTitle"]))
    story.append(Spacer(1, 15))
    story.append(Paragraph("Automated Exploratory Data Analysis & Strategic AI Insights", custom_styles["CoverSubtitle"]))
    story.append(Spacer(1, 250))
    
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    meta_text = f"""
    <b>Target Dataset:</b> {dataset_name}<br/>
    <b>Report Date:</b> {current_date}<br/>
    <b>Prepared For:</b> Executive Decision Support<br/>
    <b>Engine Version:</b> Production v1.0.0
    """
    story.append(Paragraph(meta_text, custom_styles["CoverMeta"]))
    story.append(PageBreak())
    
    # ================= PAGE 2: EXEC SUMMARY & CLEANING =================
    story.append(Paragraph("1. Executive Summary & Data Cleaning", custom_styles["CustomH1"]))
    story.append(Spacer(1, 10))
    
    exec_summary_text = (
        f"This report provides an automated end-to-end audit and analytical overview of the "
        f"dataset <b>{dataset_name}</b>. The pipeline completed file loading, structured standardizations, "
        f"missing value imputations, duplicate filtering, and statistical computations. "
        f"Following the sanitization phase, an AI Business Insight Engine compiled strategic opportunities "
        f"and key risks to guide business decision-making processes."
    )
    story.append(Paragraph(exec_summary_text, custom_styles["CustomBody"]))
    story.append(Spacer(1, 15))
    
    # Cleaning Summary Table
    story.append(Paragraph("Data Cleaning Operations Log", custom_styles["CustomH2"]))
    story.append(Spacer(1, 6))
    
    orig_shape = cleaning_log.get('original_shape', (0, 0))
    new_shape = cleaning_log.get('new_shape', (0, 0))
    dups_removed = cleaning_log.get('duplicates_removed', 0)
    num_imputed = len(cleaning_log.get('missing_numeric_filled', {}))
    cat_imputed = len(cleaning_log.get('missing_categorical_filled', {}))
    
    cleaning_table_data = [
        [Paragraph("Metric", custom_styles["TableHeader"]), Paragraph("Before Pipeline", custom_styles["TableHeader"]), Paragraph("After Pipeline", custom_styles["TableHeader"])],
        [Paragraph("Dataset Shape (Rows, Cols)", custom_styles["TableCell"]), Paragraph(str(orig_shape), custom_styles["TableCell"]), Paragraph(str(new_shape), custom_styles["TableCell"])],
        [Paragraph("Duplicate Rows Removed", custom_styles["TableCell"]), Paragraph("-", custom_styles["TableCell"]), Paragraph(str(dups_removed), custom_styles["TableCell"])],
        [Paragraph("Imputed Numerical Columns", custom_styles["TableCell"]), Paragraph("-", custom_styles["TableCell"]), Paragraph(str(num_imputed), custom_styles["TableCell"])],
        [Paragraph("Imputed Categorical Columns", custom_styles["TableCell"]), Paragraph("-", custom_styles["TableCell"]), Paragraph(str(cat_imputed), custom_styles["TableCell"])]
    ]
    
    t_clean = Table(cleaning_table_data, colWidths=[204, 150, 150])
    t_clean.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
    ]))
    story.append(t_clean)
    
    # Detailed imputation summaries if any
    imputations = []
    if cleaning_log.get('missing_numeric_filled'):
        for col, val in cleaning_log['missing_numeric_filled'].items():
            imputations.append(f"Imputed numerical column <b>{col}</b> with median value: <b>{val}</b>")
    if cleaning_log.get('missing_categorical_filled'):
        for col, val in cleaning_log['missing_categorical_filled'].items():
            imputations.append(f"Imputed categorical column <b>{col}</b> with mode value: <b>{val}</b>")
            
    if imputations:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Imputation Detail:", custom_styles["CustomH3"]))
        for imp in imputations[:6]: # Limit to 6 items to avoid overflow
            story.append(Paragraph(f"- {imp}", custom_styles["CustomBullet"]))
        if len(imputations) > 6:
            story.append(Paragraph(f"- ... and {len(imputations) - 6} other column imputations.", custom_styles["CustomBullet"]))
            
    story.append(PageBreak())
    
    # ================= PAGE 3: DATA SCHEMA & DESCRIPTIVE STATS =================
    story.append(Paragraph("2. Schema & Descriptive Statistics", custom_styles["CustomH1"]))
    story.append(Spacer(1, 8))
    
    # Column Schema Table (Max 15 columns shown for PDF safety)
    story.append(Paragraph("Dataset Column Schema (First 15 Columns)", custom_styles["CustomH2"]))
    story.append(Spacer(1, 6))
    
    schema_headers = [
        Paragraph("Column Name", custom_styles["TableHeader"]),
        Paragraph("Data Type", custom_styles["TableHeader"]),
        Paragraph("Missing Count", custom_styles["TableHeader"])
    ]
    schema_table_data = [schema_headers]
    
    cols_to_print = eda_results['columns'][:15]
    for col in cols_to_print:
        dtype = eda_results['dtypes'].get(col, 'unknown')
        missing = eda_results['missing_values'].get(col, 0)
        schema_table_data.append([
            Paragraph(col, custom_styles["TableCell"]),
            Paragraph(dtype, custom_styles["TableCell"]),
            Paragraph(str(missing), custom_styles["TableCell"])
        ])
        
    t_schema = Table(schema_table_data, colWidths=[220, 144, 140])
    t_schema.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('TOPPADDING', (0, 0), (-1, 0), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]))
    story.append(t_schema)
    
    if len(eda_results['columns']) > 15:
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<i>* Note: {len(eda_results['columns']) - 15} additional columns omitted for presentation clarity.</i>", custom_styles["CustomBody"]))
        
    story.append(Spacer(1, 15))
    
    # Descriptive Statistics (Numerical Transposed Table)
    if eda_results.get('describe_numeric'):
        story.append(Paragraph("Numerical Statistics Summary (Transposed)", custom_styles["CustomH2"]))
        story.append(Spacer(1, 6))
        
        stat_headers = ["Column", "Count", "Mean", "Std Dev", "Min", "50% (Median)", "Max"]
        stat_table_data = [[Paragraph(h, custom_styles["TableHeader"]) for h in stat_headers]]
        
        # Display first 8 numerical columns to prevent height overflow
        num_cols_to_print = list(eda_results['describe_numeric'].keys())[:8]
        for col in num_cols_to_print:
            stats = eda_results['describe_numeric'][col]
            stat_table_data.append([
                Paragraph(col, custom_styles["TableCell"]),
                Paragraph(f"{stats.get('count', 0):.0f}", custom_styles["TableCell"]),
                Paragraph(f"{stats.get('mean', 0):.2f}", custom_styles["TableCell"]),
                Paragraph(f"{stats.get('std', 0):.2f}", custom_styles["TableCell"]),
                Paragraph(f"{stats.get('min', 0):.2f}", custom_styles["TableCell"]),
                Paragraph(f"{stats.get('50%', 0):.2f}", custom_styles["TableCell"]),
                Paragraph(f"{stats.get('max', 0):.2f}", custom_styles["TableCell"])
            ])
            
        t_stats = Table(stat_table_data, colWidths=[120, 64, 64, 64, 64, 64, 64])
        t_stats.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_primary),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('TOPPADDING', (0, 0), (-1, 0), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ]))
        story.append(t_stats)
        if len(eda_results['describe_numeric']) > 8:
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<i>* Note: {len(eda_results['describe_numeric']) - 8} additional numerical columns omitted for presentation clarity.</i>", custom_styles["CustomBody"]))
            
    story.append(PageBreak())
    
    # ================= PAGE 4: VISUALIZATIONS =================
    story.append(Paragraph("3. Exploratory Data Visualizations", custom_styles["CustomH1"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "The following charts reflect distributions, trends, and relationships identified "
        "across the key features of the cleaned dataset.", custom_styles["CustomBody"]
    ))
    story.append(Spacer(1, 10))
    
    # Add charts in pairs or sequentially using KeepTogether to keep them on page cleanly
    chart_list = []
    
    # Define order of display
    ordered_charts = ["histogram", "boxplot", "heatmap", "pie_chart", "bar_chart", "trend_chart"]
    for chart_name in ordered_charts:
        if chart_name in chart_paths and os.path.exists(chart_paths[chart_name]):
            c_path = chart_paths[chart_name]
            chart_title = chart_name.replace("_", " ").title()
            
            # Subheading
            subhead = Paragraph(f"Figure: {chart_title}", custom_styles["CustomH2"])
            # ReportLab image (Width = 400, Height = 200, scaled down to look crisp)
            img = Image(c_path, width=400, height=200)
            
            # Pack each chart + title together so they don't break across pages
            chart_container = KeepTogether([
                subhead,
                Spacer(1, 4),
                img,
                Spacer(1, 15)
            ])
            chart_list.append(chart_container)
            
    # Add charts to story, inserting a PageBreak after every 2 charts to space them beautifully
    for idx, chart_item in enumerate(chart_list):
        story.append(chart_item)
        # Add PageBreak after every 2 charts, except for the last one
        if (idx + 1) % 2 == 0 and (idx + 1) < len(chart_list):
            story.append(PageBreak())
            
    story.append(PageBreak())
    
    # ================= PAGE 5: AI INSIGHTS =================
    story.append(Paragraph("4. AI Insights & Strategic Recommendations", custom_styles["CustomH1"]))
    story.append(Spacer(1, 10))
    
    if ai_insights:
        insight_flowables = markdown_to_flowables(ai_insights, custom_styles)
        story.extend(insight_flowables)
    else:
        story.append(Paragraph("No AI insights generated. Please check LLM credentials and prompt pipeline.", custom_styles["CustomBody"]))
        
    # Build document, attaching headers/footers
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    return output_pdf_path
