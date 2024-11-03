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

# API key
API = st.secrets["API_KEY"]

# Dataset mapping
NT_DATASETS = {
    "🏫 Education Statistics": {
        "url": "data/School_List_Public_2024_11_02_06_12_00.csv",
        "description": "List of Schools across Northern Territory",
        "category": "Education",
        "last_updated": "2021-06-23",
    },
    "🚓 NT Crime Statistics": {
        "url": "data/nt_crime_statistics_full_data.csv",
        "description": "Crime Statistics and Safty Metrics across Northern Territory",
        "category": "Crime",
        "last_updated": "2024-10-03",
    },
    "💼 NT Employment": {
        "url": "data/erecruit-open-data-portal.csv",
        "description": "Employment and Workfoce statistics across Northern Territory",
        "category": "Employment",
        "last_updated": "2021-06-23",
    },
}
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

def handle_user_input():
    """
    Handle user input and update session state
    """
    if st.session_state.get('user_input') and st.session_state.get('user_input').strip():
        start_time = datetime.now()
                
       
        st.session_state['current_analysis'] = st.session_state.get('user_input')
        st.session_state['user_input'] = ''
        st.session_state['loading'] = True
        st.session_state['chat_history'].append({
            "role": "user",
            "content": st.session_state['current_analysis'],
            "timestamp": start_time.strftime("%H:%M:%S")
        })

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
                # start_time = datetime.now()

                with st.spinner("🧙‍♂️ Analyzing data..."):
                    # llm = PandasAIOpenAI(api_token=API, model="gpt-4o")
                    llm = PandasAIOpenAI(api_token=API, model="gpt-4")

                    df = SmartDataframe(
                        data,
                        config={
                            "llm": llm,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "analysis_depth": analysis_depth.lower(),
                            "show_visualizations": show_visualizations,
                            "language": language.lower(),
                        },
                    )

                    response = df.chat(st.session_state['current_analysis'])
                    output_type, processed_response = process_output(response, data)

                    # st.session_state['chat_history'].append({
                    #     "role": "user",
                    #     "content": st.session_state['current_analysis'],
                    #     "timestamp": start_time.strftime("%H:%M:%S")
                    # })

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

# Page configuration
st.set_page_config(
    page_title="AskNT",
    page_icon="logo\\favicon.jpeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS with Dark Mode and Animations
st.markdown(
    """
    <style>
        
    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideFromLeft {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideFromRight {
        from { transform: translateX(100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes floatingEffect {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Designs CSS */
    
    .dataset-card {
        background: #2d2d2d;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-in;
        transition: transform 0.3s ease;
    }
    
    .dataset-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        transition: transform 0.3s ease;
        animation: floatingEffect 3s ease-in-out infinite;
        /* Add border and shadow */
        border: 1px solid #cccccc; /* Light gray border */
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
    }
    

    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #2d2d2d;
    }
    ::-webkit-scrollbar-thumb {
        background: #4527a0;
        border-radius: 5px;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("🌟 Data Explorer")

    # Dataset Categories
    st.subheader("📑 Categories")
    categories = list(set(dataset["category"] for dataset in NT_DATASETS.values()))
    selected_category = st.selectbox("Filter by category", ["All"] + categories)

    # Dataset Selection
    st.subheader("🔍 Select Dataset")
    if selected_category == "All":
        available_datasets = NT_DATASETS.keys()
    else:
        available_datasets = [
            name
            for name, info in NT_DATASETS.items()
            if info["category"] == selected_category
        ]

    selected_dataset = st.selectbox("Choose a dataset", available_datasets, on_change=clear_chat)

    # Analysis Settings
    with st.expander("⚙️ Analysis Settings"):
        # Temperature slider
        temperature = st.slider(
            "🌡️ Temperature",
            0.0,
            1.0,
            0.5,
            0.1,
            help="Higher values make the output more random, lower values more deterministic.",
        )

        # Max tokens slider
        max_tokens = st.slider(
            "📏 Max Tokens",
            50,
            500,
            200,
            50,
            help="Maximum number of tokens in the response.",
        )

        # Analysis depth
        analysis_depth = st.select_slider(
            "🔬 Analysis Depth",
            options=["Basic", "Intermediate", "Advanced"],
            value="Intermediate",
            help="Choose the depth of analysis. Advanced may take longer but provides more detailed insights.",
        )

        # Visualization toggle
        show_visualizations = st.checkbox(
            "📊 Show Visualizations",
            value=True,
            help="Toggle to include data visualizations in the analysis",
        )

        # Language selection
        language = st.selectbox(
            "🌐 Output Language",
            ["English", "Spanish", "French", "German", "Chinese"],
            index=0,
            help="Select the language for the analysis output",
        )

    # Quick Actions
    st.subheader("⚡ Quick Actions")
    if st.button("📊 Generate Summary"):
        st.session_state['current_analysis'] = "Generate a comprehensive summary of this dataset"
    if st.button("🔍 Find Correlations"):
        st.session_state['current_analysis'] = "Find and explain the most significant correlations in this dataset"
    if st.button("📈 Trend Analysis"):
        st.session_state['current_analysis'] = "Analyze and describe the major trends present in this dataset"

# Main Content
if selected_dataset:
    dataset_info = NT_DATASETS[selected_dataset]

    # Animated Header
    st.markdown(
        f"""
        <h1 style='text-align: center; color: #bb86fc; animation: fadeIn 1s ease-in;'>
            🌟 {selected_dataset} Explorer 🌟
        </h1>
    """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("🚀 Loading your data magic..."):
            data, quality_report = load_nt_dataset(dataset_info["url"])
            if data is not None:
                # Interactive Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            📊 Rows<br>
                            <h2>{len(data):,}</h2>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    # st.metric("📊 Rows", f"{len(data):,}")
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            🧮 Columns<br>
                            <h2>{len(data.columns)}</h2>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    # st.metric("🧮 Columns", len(data.columns))
                with col3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            🕒 Last Updated<br>
                            <h2>{dataset_info["last_updated"]}</h2>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    # st.metric("🕒 Last Updated", dataset_info["last_updated"])

                # Data Preview Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Preview", "ℹ️ Info", "📊 Statistics", "📋 Data Quality"])
                with tab1:
                    st.dataframe(data.head(10), use_container_width=True)
                with tab2:
                    st.json(data.dtypes.astype(str).to_dict())
                with tab3:
                    st.dataframe(data.describe(), use_container_width=True)
                with tab4:
                    st.json(quality_report)

                # Chat Interface
                st.markdown("### 🤖 Chat with Your Data 💬")
                create_chat_interface(data)

    except Exception as e:
        st.error(f"📛 Error loading dataset: {str(e)}")
else:
    st.info("🎭 Choose your dataset adventure from the sidebar!")
