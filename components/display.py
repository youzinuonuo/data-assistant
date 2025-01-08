import streamlit as st
from typing import List
import pandas as pd
import pathlib
import uuid
import matplotlib.pyplot as plt
import io

def display_dataframes(dfs: List[pd.DataFrame]):
    for i, df in enumerate(dfs, 1):
        st.subheader(f"Sheet {i}")
        st.dataframe(df)

def display_results(result) -> None:
    if result is not None:
        print(f"Result type: {type(result)}")
        
        # Handle string result
        if isinstance(result, str):
            st.code(result, language='text')
            
        # Handle numeric result
        elif isinstance(result, (int, float)):
            st.code(str(result), language='text')
            
        # Handle DataFrame result
        elif isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        
        # Handle Matplotlib figure
        elif isinstance(result, plt.Figure):
            # Save figure to memory buffer
            buf = io.BytesIO()
            result.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Display the figure
            st.image(buf, use_column_width=True)
            
            # Add download button
            key = f"download_{str(uuid.uuid4())}"
            st.download_button("Download Plot", buf, "plot.png", "image/png", key=key)
            
            # Cleanup resources
            plt.close(result)
            buf.close()
        
        else:
            st.warning(f"Unsupported result type: {type(result)}")
    
    else:
        st.error('No result available. Please check your question and try again.')