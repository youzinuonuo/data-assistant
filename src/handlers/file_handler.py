import pandas as pd
import streamlit as st
from typing import List

class ExcelFileHandler:
    def __init__(self, file) -> None:
        self.file = file

    def read(self) -> List[pd.DataFrame]:
        try:
            xls = pd.ExcelFile(self.file)
            sheet_names = xls.sheet_names
            
            selected_sheet = st.selectbox(
                "Select a sheet to analyze",
                options=sheet_names,
                index=0
            )
            
            if not selected_sheet:
                st.warning("Please select at least one sheet")
                return None
                
            dfs = []
            # for sheet in selected_sheets:
            #     df = pd.read_excel(xls, sheet_name=sheet)
            df = pd.read_excel(xls, sheet_name=selected_sheet)
            dfs.append(df)
            return dfs
            
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None