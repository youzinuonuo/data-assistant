from typing import List
import pandas as pd
from pandasai import SmartDatalake, Agent
from pandasai.responses import StreamlitResponse
from pandasai.llm import BambooLLM
import os
def chat_with_data(dfs: List[pd.DataFrame], query: str):
    try:
        lake = SmartDatalake(dfs)
        # config={'llm':llm,
        #         'response_parser':StreamlitResponse,
        #         }
        # agent = Agent(lake, config=config)
        # res = agent.chat(query)
        # return res;
    except Exception as e:
        return f"Error processing query: {str(e)}"
    

def chat_lake(dfs: List[pd.DataFrame], query: str) -> None:
    llm = BambooLLM(api_key="")
    # os.environ["PANDASAI_API_KEY"] = ""
    config={
        "llm": llm,
        "save_charts": True,
        "save_charts_path": "./exports/charts",
    }
    lake = SmartDatalake(dfs, config=config)
    response = lake.chat(query)
    print(response)
    return response;
