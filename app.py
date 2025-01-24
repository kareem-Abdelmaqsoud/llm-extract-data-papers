import streamlit as st
from gmft.auto import CroppedTable, AutoTableDetector, AutoTableFormatter
from gmft.pdf_bindings import PyPDFium2Document
import google.generativeai as genai
import base64
import io
import json
import pandas as pd

# Initialize the GMFT components
detector = AutoTableDetector()
formatter = AutoTableFormatter()

# Configure the generative model (replace with your API key environment variable)
import os
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Function to ingest a PDF and extract tables
def ingest_pdf(pdf_path):
    doc = PyPDFium2Document(pdf_path)
    tables = []
    for page in doc:
        tables += detector.extract(page)
    return tables, doc

# Function to extract data from a table image
def extract_table_data(image):
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    # Define the prompt
    prompt = ('Extract all the data in this table in a tabular form.'
              'Make sure all the elemental compositions add up to 100')

    # Generate the response using Gemini
    response = genai.GenerativeModel(model_name="gemini-1.5-pro").generate_content(
        [
            {'mime_type': 'image/png', 'data': image_data},
            prompt
        ],
        generation_config={"temperature": 0}
    )
    return response.text

# Function to convert text to JSON and DataFrame
def text_to_dataframe(response_text):
    prompt = f'''Convert the following table into a JSON object. The JSON should have two keys:
    1. "headers": A list containing the column names from the table.
    2. "data": A list of rows, where each row is a list of values corresponding to the headers.

    Make sure:
    - Missing values in the table are represented as `null` in the JSON.
    - Numeric values are properly formatted as numbers, not strings.
    - The JSON should not contain any additional text or explanations, only the structured JSON object.

    Here is the table:
    {response_text}
    '''
    response = genai.GenerativeModel(model_name="gemini-1.5-pro").generate_content(
        prompt, generation_config={"temperature": 0}
    )
    raw_response = response.text.strip("```json").strip("```").strip()

    # Convert JSON string to Python dictionary
    table_data = json.loads(raw_response)

    # Replace 'null' with None for proper handling by pandas
    for i, row in enumerate(table_data['data']):
        table_data['data'][i] = [None if x is None else x for x in row]

    # Convert to DataFrame
    return pd.DataFrame(table_data['data'], columns=table_data['headers'])

# Streamlit UI
st.title("PDF Table Extractor with GMFT and Gemini")
st.subheader("Upload a PDF to extract tables and process data")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file to disk
    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract tables from the PDF
    st.write("Processing the PDF to detect tables...")
    tables, doc = ingest_pdf(pdf_path)
    st.write(f"Number of tables detected: {len(tables)}")

    # Iterate through detected tables
    for i, table in enumerate(tables):
        st.write(f"Table {i+1}")
        ft = formatter.extract(table)
        image = ft.image(dpi=256)
        st.image(image, caption=f"Table {i+1}", use_container_width=True)

        if st.button(f"Process Table {i+1}", key=f"table_{i+1}"):
            st.write("Processing table with Gemini...")
            response_text = extract_table_data(image)

            st.write("Extracting structured data...")
            df = text_to_dataframe(response_text)

            st.dataframe(df)

            # Save the DataFrame to an in-memory buffer as an Excel file
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)

            # Rewind the buffer to the beginning
            buffer.seek(0)

            # Add a download button
            st.download_button(
                label="Download Table as Excel",
                data=buffer,
                file_name=f"table_{i+1}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )