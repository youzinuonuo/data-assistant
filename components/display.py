import streamlit as st
from typing import List
import pandas as pd
from pandasai import Agent
import os,pathlib
import uuid

def display_dataframes(dfs: List[pd.DataFrame]):
    for i, df in enumerate(dfs, 1):
        st.subheader(f"Sheet {i}")
        st.dataframe(df)

def display_results(result) -> None:
    if result is not None:
        print(result)
        print(type(result))
        if isinstance(result, str):
            if result.endswith('.png'):
                current_dir = pathlib.Path(__file__).parent.parent
                chart_path = current_dir.joinpath(result)
                if chart_path.exists():
                    st.image(str(chart_path), use_column_width=True)
                    with open(chart_path, "rb") as image_file:
                        key = f"download_{str(uuid.uuid4())}"
                        st.download_button("download", image_file, result.split('/')[-1], "image/png", key=key)
                else:
                    st.warning(f"can't find: {chart_path}")
            else:
                st.code(result, language='text')
        elif isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        elif isinstance(result, (int, float)):
            st.code(str(result), language='text')
        # elif isinstance(result, pathlib.Path):
        #     if result.exists():
        #         current_dir = pathlib.Path(__file__).parent.parent
        #         chart_path = current_dir.joinpath(result)
        #         st.image(str(chart_path), use_column_width=True)
        #         with open(chart_path, "rb") as image_file:
        #             st.download_button("download", image_file, "chart.png", "image/png")

    else:
        st.error('We are unable to retrieve any result. Please check your question and try again.')