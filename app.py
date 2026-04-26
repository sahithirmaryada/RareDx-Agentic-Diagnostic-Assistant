import logging
import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
from pydantic import BaseModel, ValidationError, constr, conint
from core.agent import rare_dx_agent
from core.retriever import InfrastructureError, get_infrastructure_status
from scripts.db_utils import get_all_symptoms, get_all_genes
from scripts.report_generator import generate_pdf_report, generate_excel_report
from typing import Annotated
from pydantic import BaseModel, Field, StringConstraints, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Input validation models
class PatientInput(BaseModel):
    patient_id: Annotated[str, StringConstraints(min_length=1, max_length=50)]
    patient_age: Annotated[int, Field(ge=1, le=120)]
    patient_gender: str
    clinical_note: Annotated[str, StringConstraints(max_length=5000)] = ""
    selected_symptoms: list[str]
    selected_genes: list[str]

    class Config:
        validate_assignment = True

# --- 1. UI Configuration & Branding ---
st.set_page_config(page_title="RareDx | Clinical Decision Support", layout="wide", page_icon="🩺")
st.title("🩺 RareDx: Agentic Diagnostic Assistant ")
st.markdown("""
    *Empowering clinicians with traceable, evidence-based "Rare Disease" identification.*
    ***
""")

# --- 2. Startup: Dynamic Data Synchronization ---
health_status = get_infrastructure_status()
catalog_error = None

if "hpo_options" not in st.session_state:
    st.session_state.hpo_options = []
if "gene_options" not in st.session_state:
    st.session_state.gene_options = []

if health_status["neo4j"]["ok"]:
    needs_catalog_refresh = (
        not st.session_state.hpo_options or not st.session_state.gene_options
    )
    if needs_catalog_refresh:
        with st.spinner("🔄 Synchronizing with Neo4j Knowledge Graph..."):
            try:
                st.session_state.hpo_options = get_all_symptoms()
                st.session_state.gene_options = get_all_genes()
            except InfrastructureError as exc:
                st.session_state.hpo_options = []
                st.session_state.gene_options = []
                catalog_error = str(exc)
            except Exception as exc:
                st.session_state.hpo_options = []
                st.session_state.gene_options = []
                catalog_error = "Unexpected error while loading structured inputs."
                st.error(f"Unexpected catalog error: {exc}")
else:
    st.session_state.hpo_options = []
    st.session_state.gene_options = []

if "diagnostic_results" not in st.session_state:
    st.session_state.diagnostic_results = None

# --- 3. Sidebar: Patient Profile ---

with st.sidebar:
    st.subheader("System Status")
    if health_status["neo4j"]["ok"] and health_status["chroma"]["ok"]:
        st.success("Neo4j: Online  \nChromaDB: Online")
    else:
        st.error("Neo4j: Offline")
        st.caption(health_status["neo4j"]["detail"])
        st.error("ChromaDB: Offline")
        st.caption(health_status["chroma"]["detail"])

    if not health_status["ready"]:
        st.error(
            "Infrastructure Offline: Grounded diagnostic services are currently unavailable. Please check backend connectivity."
        )
    elif catalog_error:
        st.error(f"Structured input catalog unavailable: {catalog_error}")

    # st.divider()
    st.header("👤  Patient Profile")
    patient_id = st.text_input("Patient ID", value="PID-test-xx", key="patient_id")
    patient_age = st.number_input("Age", 1, 100, 30, key="patient_age")
    patient_gender = st.selectbox("Gender", ["Female", "Male", "Other"], key="patient_gender")
    st.divider()
    st.info("No evidence, no diagnosis. RareDx strictly enforces evidence-based outputs.")

# Custom CSS for "The Glass Box" Aesthetic
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    .evidence-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .icd-badge {
        background-color: #e7f3ff;
        color: #007bff;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .trust-pill {
        background-color: #d4edda;
        color: #155724;
        padding: 2px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        border: 1px solid #c3e6cb;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. Main Multi-Tab Interface ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✚ Clinical Input", 
    "✔️ Diagnostic Results", 
    "🔎 Evidence Explorer", 
    "👀 Audit Trail",
    "📥 Download Report"
])

# --- TAB 1: Clinical Input ---
with tab1:
    with st.container(border=True):
        st.subheader("Clinical Data Entry")
        note_col, param_col = st.columns([2, 1])
        
        with note_col:
            clinical_notes = st.text_area("Unstructured Clinical Notes", placeholder="e.g., 30yo female with butterfly rash and joint pain...", height=250)
        
        with param_col:
            st.session_state.hpo_options = get_all_symptoms() if health_status["neo4j"]["ok"] else []
            st.session_state.gene_options = get_all_genes() if health_status["neo4j"]["ok"] else []
            
            sel_symptoms = st.multiselect("Standardized HPO Symptoms", options=st.session_state.hpo_options)
            sel_genes = st.multiselect("Genomic Markers", options=st.session_state.gene_options)
            st.file_uploader("Upload DICOM/FHIR (Optional)")

    if st.button("🔬 Run RareDx Diagnosis", disabled=not health_status["ready"], use_container_width=True, type="primary"):
        # Validate inputs
        try:
            patient_input = PatientInput(
                patient_id=patient_id,
                patient_age=patient_age,
                patient_gender=patient_gender,
                clinical_note= clinical_notes,
                selected_symptoms= sel_symptoms,
                selected_genes= sel_genes
            )
        except ValidationError as e:
            st.error(f"Input validation failed: {e}")
            st.stop()

        if not patient_input.selected_symptoms and not patient_input.selected_genes:
            st.warning("Select at least one structured symptom or genetic marker.")
        elif not health_status["ready"]:
            st.error(
                "Infrastructure Offline: Grounded diagnostic services are currently unavailable. Please check backend connectivity."
            )
        else:
            with st.spinner("Agentic Brain is querying Graph and literature..."):
                initial_state = {
                    "clinical_note": patient_input.clinical_note,
                    "selected_symptoms": patient_input.selected_symptoms,
                    "selected_genes": patient_input.selected_genes,
                    "patient_id": patient_input.patient_id,
                    "patient_age": patient_input.patient_age,
                    "patient_gender": patient_input.patient_gender,
                    "graph_results": [],
                    "semantic_results": [],
                    "candidates": [],
                    "validated_evidence": [],
                    "final_report": {},
                    "audit_trail": []
                }
                
                try:
                    final_state = rare_dx_agent.invoke(initial_state)
                    st.session_state.diagnostic_results = final_state
                    st.success("Diagnostic processing complete.")
                except InfrastructureError as exc:
                    st.session_state.diagnostic_results = None
                    st.error(str(exc))
                except Exception as e:
                    st.session_state.diagnostic_results = None
                    st.error(f"Agent Execution Error: {e}")

# --- TAB 2: Diagnostic Results ---
with tab2:
    if st.session_state.diagnostic_results:
        report = st.session_state.diagnostic_results.get("final_report", {})
        st.subheader("Ranked Diagnostic Candidates")
        st.markdown(f"> **Clinical Summary:** {report.get('summary', 'Synthesis complete.')}")
        
        candidates = st.session_state.diagnostic_results.get("candidates", [])
        if candidates:
            for can in candidates:
                with st.container():
                    c1, c2, c3 = st.columns([3, 1, 1])
                    c1.markdown(f"### {can.get('disease', 'Unknown')}")
                    c2.metric("Confidence Score", f"{int(can.get('score', 0)*100)}%")
                    icd10_code = can.get("icd10") or "Unavailable"
                    c3.markdown(f"**ICD-10 Code:** :blue-background[{icd10_code}]")
                    badge_text = ", ".join(can.get("evidence_badges", []))
                    if badge_text:
                        st.caption(f"Evidence: {badge_text}")
                    st.divider()
        else:
            st.error("No verified candidates found. Try relaxing the search criteria.")
    else:
        st.info("Awaiting input analysis...")

# --- TAB 3: Evidence Explorer ---
with tab3:
    if st.session_state.diagnostic_results:
        t_map = st.session_state.diagnostic_results.get("final_report", {}).get("traceability_map", {})
        st.subheader("🛠️ Evidence Grounding")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🕸️ Neo4j Graph Paths")
            for path in t_map.get("graph_paths", ["No paths found."]):
                st.code(path, language="cypher")
        
        with col_b:
            st.markdown("#### 📚 PubMed Evidence")
            for pmid in t_map.get("citations", []):
                st.link_button(f"PMID: {pmid}", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        
        st.markdown("#### ✅ Clinical Validation (MCP)")
        for val in t_map.get("mcp_validation", ["No additional grounded metadata available."]):
            st.success(val)
    else:
        st.info("Evidence paths will appear here after diagnosis.")

# --- TAB 4: Audit Trail ---
with tab4:
    if st.session_state.diagnostic_results:
        st.subheader("🕵️ Agent Decision Log (Audit)")
        audit_data = st.session_state.diagnostic_results.get("audit_trail", [])
        
        # Display as a searchable table
        df = pd.DataFrame(audit_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("#### 📄 Final Traceability Map (JSON)")
        st.json(st.session_state.diagnostic_results.get("final_report", {}).get("traceability_map", {}))
    else:
        st.info("Full audit trail will be available upon execution.")


# --- TAB 5: Download Report ---
with tab5:
    if st.session_state.diagnostic_results:
        st.subheader("📥 Download Report")
        st.markdown("Download your diagnostic results in your preferred format.")
        
        report = st.session_state.diagnostic_results.get("final_report", {})
        candidates = st.session_state.diagnostic_results.get("candidates", [])
        
        col_pdf, col_excel = st.columns(2)
        
        with col_pdf:
            st.markdown("### 📕 PDF Report")
            st.markdown("Professional clinical report suitable for printing and filing.")
            pdf_data = generate_pdf_report(
                patient_id=st.session_state.diagnostic_results.get("patient_id", "Unknown"),
                patient_age=st.session_state.diagnostic_results.get("patient_age", 0),
                patient_gender=st.session_state.diagnostic_results.get("patient_gender", "Unknown"),
                clinical_note=st.session_state.diagnostic_results.get("clinical_note", ""),
                selected_symptoms=st.session_state.diagnostic_results.get("selected_symptoms", []),
                selected_genes=st.session_state.diagnostic_results.get("selected_genes", []),
                candidates=candidates,
                traceability_map=report.get("traceability_map", {}),
                audit_trail=st.session_state.diagnostic_results.get("audit_trail", []),
                summary=report.get("summary", ""),
            )
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_data,
                file_name=f"RareDx_Report_{st.session_state.diagnostic_results.get('patient_id', 'Unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        
        with col_excel:
            st.markdown("### 📊 Excel Report")
            st.markdown("Structured data export with multiple sheets for analysis.")
            excel_data = generate_excel_report(
                patient_id=st.session_state.diagnostic_results.get("patient_id", "Unknown"),
                patient_age=st.session_state.diagnostic_results.get("patient_age", 0),
                patient_gender=st.session_state.diagnostic_results.get("patient_gender", "Unknown"),
                clinical_note=st.session_state.diagnostic_results.get("clinical_note", ""),
                selected_symptoms=st.session_state.diagnostic_results.get("selected_symptoms", []),
                selected_genes=st.session_state.diagnostic_results.get("selected_genes", []),
                candidates=candidates,
                traceability_map=report.get("traceability_map", {}),
                audit_trail=st.session_state.diagnostic_results.get("audit_trail", []),
            )
            st.download_button(
                label="⬇️ Download Excel",
                data=excel_data,
                file_name=f"RareDx_Report_{st.session_state.diagnostic_results.get('patient_id', 'Unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    else:
        st.info("Generate diagnostic results first to enable report downloads.")