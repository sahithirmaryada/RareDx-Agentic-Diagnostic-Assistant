"""
Report generation utilities for RareDx diagnostic results.
Supports PDF, Excel, and JSON export formats.
"""

import io
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_pdf_report(
    patient_id: str,
    patient_age: int,
    patient_gender: str,
    clinical_note: str,
    selected_symptoms: List[str],
    selected_genes: List[str],
    candidates: List[Dict[str, Any]],
    traceability_map: Dict[str, Any],
    audit_trail: List[Dict[str, Any]],
    summary: str,
) -> bytes:
    """
    Generate a professional PDF report of the diagnostic results.
    Returns PDF as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#003d82"),
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#003d82"),
        spaceAfter=6,
        spaceBefore=12,
    )
    normal_style = styles["Normal"]
    normal_style.fontSize = 10

    elements = []

    # --- Title ---
    elements.append(Paragraph("🩺 RareDx Clinical Diagnostic Report", title_style))
    elements.append(
        Paragraph(
            f"<i>Agentic Rare-Disease-Diagnostic Assistant | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 0.3 * inch))

    # --- Patient Demographics ---
    elements.append(Paragraph("Patient Demographics", heading_style))
    demo_data = [
        ["Patient ID", patient_id],
        ["Age", str(patient_age)],
        ["Gender", patient_gender],
    ]
    demo_table = Table(demo_data, colWidths=[1.5 * inch, 4 * inch])
    demo_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0f8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ]
        )
    )
    elements.append(demo_table)
    elements.append(Spacer(1, 0.2 * inch))

    # --- Clinical Input Summary ---
    elements.append(Paragraph("Clinical Input Summary", heading_style))
    if clinical_note:
        elements.append(Paragraph(f"<b>Clinical Notes:</b> {clinical_note[:300]}...", normal_style))
    if selected_symptoms:
        elements.append(
            Paragraph(f"<b>Symptoms (HPO):</b> {', '.join(selected_symptoms[:5])}", normal_style)
        )
    if selected_genes:
        elements.append(Paragraph(f"<b>Genetic Markers:</b> {', '.join(selected_genes[:5])}", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # --- Clinical Summary ---
    elements.append(Paragraph("Diagnostic Summary", heading_style))
    elements.append(Paragraph(summary, normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # --- Ranked Candidates ---
    elements.append(Paragraph("Ranked Diagnostic Candidates", heading_style))
    if candidates:
        candidate_data = [["Rank", "Disease Name", "Confidence", "ICD-10", "Evidence"]]
        for idx, can in enumerate(candidates[:10], 1):
            evidence_str = ", ".join(can.get("evidence_badges", []))
            candidate_data.append(
                [
                    str(idx),
                    can.get("disease", "Unknown")[:30],
                    f"{int(can.get('score', 0) * 100)}%",
                    can.get("icd10", "N/A")[:15],
                    evidence_str[:40],
                ]
            )
        candidate_table = Table(candidate_data, colWidths=[0.6 * inch, 2 * inch, 1 * inch, 1 * inch, 1.4 * inch])
        candidate_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003d82")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
                ]
            )
        )
        elements.append(candidate_table)
    else:
        elements.append(Paragraph("No verified candidates found.", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # --- Evidence Grounding ---
    elements.append(Paragraph("Evidence Grounding", heading_style))
    if traceability_map.get("graph_paths"):
        elements.append(Paragraph(f"<b>Neo4j Paths:</b>", normal_style))
        for path in traceability_map.get("graph_paths", [])[:3]:
            elements.append(Preformatted(path, styles["Normal"]))
    if traceability_map.get("citations"):
        elements.append(
            Paragraph(
                f"<b>PubMed Citations:</b> {', '.join(traceability_map.get('citations', [])[:5])}",
                normal_style,
            )
        )
    elements.append(Spacer(1, 0.2 * inch))

    # --- Page Break before Audit Trail ---
    elements.append(PageBreak())

    # --- Audit Trail ---
    elements.append(Paragraph("Audit Trail (Diagnostic Process Log)", heading_style))
    if audit_trail:
        audit_data = [["Timestamp", "Node", "Action", "Details"]]
        for log in audit_trail[:10]:
            audit_data.append(
                [
                    log.get("timestamp", "N/A")[-8:],  # Show HH:MM:SS
                    log.get("node", "N/A")[:15],
                    log.get("action", "N/A")[:20],
                    log.get("details", "N/A")[:35],
                ]
            )
        audit_table = Table(audit_data, colWidths=[1 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        audit_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003d82")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
                ]
            )
        )
        elements.append(audit_table)

    # --- Footer ---
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(
        Paragraph(
            "<i>RareDx only returns candidates supported by the graph or the literature. This report is for clinical support only and does not replace clinical judgment.</i>",
            styles["Normal"],
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_excel_report(
    patient_id: str,
    patient_age: int,
    patient_gender: str,
    clinical_note: str,
    selected_symptoms: List[str],
    selected_genes: List[str],
    candidates: List[Dict[str, Any]],
    traceability_map: Dict[str, Any],
    audit_trail: List[Dict[str, Any]],
) -> bytes:
    """
    Generate an Excel workbook with diagnostic results.
    Returns Excel file as bytes.
    """
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # --- Sheet 1: Patient & Clinical Summary ---
        patient_data = {
            "Field": ["Patient ID", "Age", "Gender", "Clinical Notes", "Symptoms", "Genetic Markers"],
            "Value": [
                patient_id,
                patient_age,
                patient_gender,
                clinical_note[:200] if clinical_note else "N/A",
                "; ".join(selected_symptoms) if selected_symptoms else "None",
                "; ".join(selected_genes) if selected_genes else "None",
            ],
        }
        pd.DataFrame(patient_data).to_excel(writer, sheet_name="Patient Summary", index=False)

        # --- Sheet 2: Ranked Candidates ---
        if candidates:
            candidate_records = []
            for idx, can in enumerate(candidates, 1):
                candidate_records.append(
                    {
                        "Rank": idx,
                        "Disease Name": can.get("disease", "Unknown"),
                        "Confidence Score": f"{int(can.get('score', 0) * 100)}%",
                        "Orphanet Code": can.get("orphacode", "N/A"),
                        "ICD-10 Code": can.get("icd10", "N/A"),
                        "Matched Symptoms": "; ".join(can.get("matched_symptoms", [])),
                        "Matched Genes": "; ".join(can.get("matched_genes", [])),
                        "Evidence Badges": "; ".join(can.get("evidence_badges", [])),
                    }
                )
            pd.DataFrame(candidate_records).to_excel(writer, sheet_name="Candidates", index=False)

        # --- Sheet 3: Evidence Paths ---
        evidence_records = []
        for path in traceability_map.get("graph_paths", []):
            evidence_records.append({"Graph Path": path, "Type": "Neo4j"})
        for pmid in traceability_map.get("citations", []):
            evidence_records.append({"Graph Path": f"PMID: {pmid}", "Type": "PubMed"})
        if evidence_records:
            pd.DataFrame(evidence_records).to_excel(writer, sheet_name="Evidence", index=False)

        # --- Sheet 4: Audit Trail ---
        if audit_trail:
            audit_df = pd.DataFrame(audit_trail)
            audit_df.to_excel(writer, sheet_name="Audit Trail", index=False)

    buffer.seek(0)
    return buffer.getvalue()


def generate_json_report(
    patient_id: str,
    patient_age: int,
    patient_gender: str,
    clinical_note: str,
    selected_symptoms: List[str],
    selected_genes: List[str],
    candidates: List[Dict[str, Any]],
    traceability_map: Dict[str, Any],
    audit_trail: List[Dict[str, Any]],
    summary: str,
) -> str:
    """
    Generate a JSON export of all diagnostic data.
    Returns JSON as string.
    """
    report_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "application": "RareDx",
        },
        "patient": {
            "id": patient_id,
            "age": patient_age,
            "gender": patient_gender,
        },
        "clinical_input": {
            "clinical_note": clinical_note,
            "selected_symptoms": selected_symptoms,
            "selected_genes": selected_genes,
        },
        "diagnostic_summary": summary,
        "ranked_candidates": candidates,
        "evidence_grounding": traceability_map,
        "audit_trail": audit_trail,
    }
    return json.dumps(report_data, indent=2)
