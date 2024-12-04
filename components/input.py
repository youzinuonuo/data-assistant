import streamlit as st

def get_excel_file():
    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx", "xls"],
        help="Upload your Excel file to analyze"
    )
    return uploaded_file