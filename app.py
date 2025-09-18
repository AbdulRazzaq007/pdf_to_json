import streamlit as st
import json
import tempfile
import os

# Import our extractor
from pdf_to_json import pdf_to_json

st.set_page_config(page_title="PDF → JSON Extractor", layout="wide")

st.title("📄 PDF to Structured JSON Extractor")

st.markdown(
    """
    Upload a PDF file and this app will extract:
    - **Paragraphs**  
    - **Tables**  
    - **Charts (OCR)**  

    into a **well-structured JSON** preserving hierarchy.
    """
)

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_path = tmp_pdf.name

    st.info("✅ File uploaded. Parsing... This may take a while for large PDFs.")

    # Run parser
    try:
        output_path = os.path.join(tempfile.gettempdir(), "output.json")
        result = pdf_to_json(tmp_path, output_path)

        st.success("Extraction complete!")

        # Show JSON in app
        st.subheader("📑 Extracted JSON (preview)")
        st.json(result)

        # Download button
        with open(output_path, "r", encoding="utf-8") as f:
            json_data = f.read()
        st.download_button(
            label="💾 Download JSON",
            data=json_data,
            file_name="output.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        # cleanup
        os.unlink(tmp_path)
