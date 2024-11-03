import os
import pandas as pd
import streamlit as st
from langchain_ollama import ChatOllama
from pandasai import SmartDataframe
import time
from datetime import datetime, timedelta
import gc
import requests
from io import StringIO
from openai import OpenAI
from pandasai.llm.openai import OpenAI as PandasAIOpenAI
import plotly.express as px
import plotly.graph_objects as go
import json
import re

# Enable garbage collection
gc.enable()

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'current_analysis' not in st.session_state:
    st.session_state['current_analysis'] = None
if 'data_quality_report' not in st.session_state:
    st.session_state['data_quality_report'] = None
if 'loading' not in st.session_state:
    st.session_state['loading'] = False
if 'new_input' not in st.session_state:
    st.session_state['new_input'] = False

# Modify handle_user_input to update session state with the user's input or quick action
def handle_user_input():
    if st.session_state.get('current_analysis'):
        start_time = datetime.now()
        st.session_state['chat_history'].append({
            "role": "user",
            "content": st.session_state['current_analysis'],
            "timestamp": start_time.strftime("%H:%M:%S")
        })
        st.session_state['user_input'] = ''  # Clear user input
        st.session_state['loading'] = True
        st.session_state['new_input'] = False  # Reset the flag after handling

# Function to handle user input or quick actions once
def handle_user_input_once():
    if st.session_state.get("new_input"):
        handle_user_input()
        st.session_state["new_input"] = False  # Reset after handling

# Input field with flagging new input
st.text_input("Ask a question about the data:", key="user_input")
if st.session_state['user_input']:
    st.session_state['current_analysis'] = st.session_state['user_input']
    st.session_state['new_input'] = True  # Set flag when there's new input

# Quick Actions
st.subheader("⚡ Quick Actions")
if st.button("📊 Generate Summary"):
    st.session_state['current_analysis'] = "Generate a comprehensive summary of this dataset"
    st.session_state['new_input'] = True  # Trigger input processing

if st.button("🔍 Find Correlations"):
    st.session_state['current_analysis'] = "Find and explain the most significant correlations in this dataset"
    st.session_state['new_input'] = True

if st.button("📈 Trend Analysis"):
    st.session_state['current_analysis'] = "Analyze and describe the major trends present in this dataset"
    st.session_state['new_input'] = True

# Call function to handle new input if flagged
handle_user_input_once()

# Function to clear chat history
def clear_chat():
    st.session_state['chat_history'] = []

def process_output(response, data):
    """
    Process the API response and determine the appropriate output format
    """
    if isinstance(response, pd.DataFrame):
        return "table", response

    try:
        if isinstance(response, str):
            json_data = json.loads(response)
            if isinstance(json_data, dict) or isinstance(json_data, list):
                return "json", json_data
    except json.JSONDecodeError:
        pass

    code_blocks = re.findall(r'```.*?\n(.*?)```', str(response), re.DOTALL)
    if code_blocks:
        try:
            for block in code_blocks:
                if 'pandas' in block or 'df' in block:
                    local_dict = {'df': data, 'pd': pd}
                    exec(block, globals(), local_dict)
                    if 'result' in local_dict and isinstance(local_dict['result'], pd.DataFrame):
                        return "table", local_dict['result']
        except Exception as e:
            st.warning(f"Could not execute code block: {str(e)}")

    return "text", response

def render_output(output_type, content):
    """
    Render different types of output in the appropriate format
    """
    if output_type == "table":
        st.dataframe(
            content,
            use_container_width=True,
            hide_index=False,
        )

        csv = content.to_csv(index=False)
        st.download_button(
            label="Download table as CSV",
            data=csv,
            file_name="table_export.csv",
            mime="text/csv"
        )

        # Automatic visualization suggestion
        if len(content.columns) >= 2:
            numeric_cols = content.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) >= 2:
                fig = px.scatter(content, x=numeric_cols[0], y=numeric_cols[1])
                st.plotly_chart(fig)

    elif output_type == "json":
        st.json(content)

    elif output_type == "text":
        if content.endswith('temp_chart.png'):
            st.image(content, caption='Chart')
        else:
            st.write(content)
    
            numbers = re.findall(r'\d+\.?\d*', str(content))
            if len(numbers) > 3:
                try:
                    fig = go.Figure(data=[
                        go.Bar(x=list(range(len(numbers))), y=[float(n) for n in numbers])
                    ])
                    fig.update_layout(title="Numerical Insights Visualization")
                    st.plotly_chart(fig)
                except Exception:
                    pass

def optimize_dataframe(df):
    """
    Optimize DataFrame memory usage
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
    return df

@st.cache_data
def load_nt_dataset(url):
    """
    Load and optimize dataset
    """
    try:
        data = pd.read_csv(url)
        data = optimize_dataframe(data)

        quality_report = {
            "missing_values": data.isnull().sum().to_dict(),
            "duplicates": len(data) - len(data.drop_duplicates()),
            "memory_usage": data.memory_usage(deep=True).sum() / 1024**2
        }

        return data, quality_report
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None

def create_chat_interface(data):
    """
    Create and manage the chat interface
    """
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_output(message.get("output_type", "text"), message["content"])
            else:
                st.markdown(message["content"])

    input_container = st.container()
    with input_container:
        if st.session_state['loading'] == False:
            col1, col2 = st.columns([14, 1], vertical_alignment="bottom")
            with col1:
                st.text_input(
                    "Ask a question about the data:",
                    key="user_input",
                    on_change=handle_user_input
                )
            with col2:
                if st.button("Send 🚀"):
                    handle_user_input()
    if st.session_state['loading'] == True:
        if st.session_state['current_analysis']:
            try:
                with st.spinner("🧙‍♂️ Analyzing data..."):
                    llm = PandasAIOpenAI(api_token=API, model="gpt-4")
                    df = SmartDataframe(
                        data,
                        config={
                            "llm": llm,
                            "max_tokens": 200,
                            "temperature": 0.5,
                            "analysis_depth": "Intermediate",
                            "show_visualizations": True,
                            "language": "English",
                        },
                    )

                    response = df.chat(st.session_state['current_analysis'])
                    output_type, processed_response = process_output(response, data)

                    st.session_state['chat_history'].append({
                        "role": "assistant",
                        "content": processed_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "output_type": output_type
                    })

                st.session_state['current_analysis'] = None
                st.session_state['loading'] = False
                st.rerun()

            except Exception as e:
                st.error(f"🔥 Analysis error: {str(e)}")
                st.session_state['chat_history'].append({
                    "role": "assistant",
                    "content": f"🔥 Analysis error: {str(e)}",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "output_type": "text"
                })
                st.session_state['current_analysis'] = None
