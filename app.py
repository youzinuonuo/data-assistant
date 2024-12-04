import streamlit as st
import pathlib
from components.display import display_dataframes, display_results
from components.input import get_excel_file
from src.handlers.file_handler import ExcelFileHandler
from src.handlers.chat_handler import chat_with_data, chat_lake
import os

st.set_page_config(
    page_title="Data Chat Assistant",
    layout="wide"
)

def main():
    st.title("Data Chat Assistant")
    
    # 初始化session state
    if 'dfs' not in st.session_state:
        st.session_state['dfs'] = None
    # 初始化聊天记录
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # 创建三列，中间列作为间隔
    left_col, gap, right_col = st.columns([2, 0.2, 5])
    
    with left_col:
        st.header("Data Input")
        # 文件上传部分
        uploaded_file = get_excel_file()
        
        if uploaded_file:
            handler = ExcelFileHandler(uploaded_file)
            st.session_state['dfs'] = handler.read()
    
    # 中间列作为间隔
    with gap:
        st.markdown('<div class="gap-column"></div>', unsafe_allow_html=True)
    
    with right_col:
        st.markdown('<div class="right-column">', unsafe_allow_html=True)
        
        # 显示数据预览
        if st.session_state['dfs']:
            with st.expander("Data Preview", expanded=False):
                display_dataframes(st.session_state['dfs'])
        
        st.header("Chat Interface")
        if st.session_state['dfs'] is not None:
            # 聊天输入框固定在上方
            user_input = st.chat_input("Ask a question about your data")
            
            if user_input:
                # 显示当前对话
                st.write("Current Conversation:")
                with st.chat_message("user"):
                    st.write(user_input)
                
                with st.spinner("Analyzing..."):
                    response = chat_lake(st.session_state['dfs'], user_input)
                    with st.chat_message("assistant"):
                        with st.expander("Result", expanded=True):
                            display_results(response)
                    # 将对话添加到历史记录
                    st.session_state['chat_history'].append({
                        "question": user_input,
                        "response": response
                    })
            
            # 历史记录折叠显示
            if len(st.session_state['chat_history']) > 0:
                with st.expander("Chat History", expanded=False):
                    for chat in st.session_state['chat_history']:
                        with st.chat_message("user"):
                            st.write(chat["question"])
                        with st.chat_message("assistant"):
                            display_results(chat["response"])
        else:
            st.info("Please upload an Excel file to start chatting.")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("clear history"):
        st.session_state['chat_history'] = []
        st.rerun()

    # 限制保存最近的10条对话
    MAX_HISTORY = 10
    if len(st.session_state['chat_history']) > MAX_HISTORY:
        st.session_state['chat_history'] = st.session_state['chat_history'][-MAX_HISTORY:]

if __name__ == "__main__":
    main()