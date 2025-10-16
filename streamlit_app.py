"""
Streamlit UI for Hybrid RAG + Time Series Chatbot Agent
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import plotly.graph_objects as go
from typing import Dict, Any
import json

# Import our agent
from ts_agent import HybridChatAgent

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .file-upload-section {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .plot-container {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []  # list of {role, content}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'selected_csv' not in st.session_state:
        st.session_state.selected_csv = None

def initialize_agent():
    """Initialize the hybrid chat agent"""
    if st.session_state.agent is None:
        with st.spinner("Initializing AI agent..."):
            try:
                st.session_state.agent = HybridChatAgent()
                st.success("✅ Agent initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")
                st.info("Please check your GROQ_API_KEY in environment variables")

def save_uploaded_file(uploaded_file, file_type):
    """Save uploaded file to db directory"""
    db_dir = "db"
    os.makedirs(db_dir, exist_ok=True)
    
    file_path = os.path.join(db_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def reindex_kb_if_needed():
    """Recreate the agent to pick up new files in db/ for RAG indexing."""
    # Simple approach: reinstantiate agent so _index_db runs
    try:
        st.session_state.agent = HybridChatAgent()
    except Exception as e:
        st.warning(f"Reindex skipped: {e}")

def display_chat_message(role: str, content: str):
    """Display a chat message with proper styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def display_analysis_results(results: Dict[str, Any]):
    """Display time series analysis results"""
    if not results:
        return
    
    st.subheader("📊 Analysis Results")
    
    # Display summary
    if 'summary' in results:
        st.markdown("### Summary")
        st.text(results['summary'])
    
    # Display outputs
    if 'outputs' in results:
        outputs = results['outputs']
        
        # Show HTML plots
        if 'prophet_anomalies_html' in outputs:
            st.markdown("### 🔍 Anomaly Detection Plot")
            try:
                with open(outputs['prophet_anomalies_html'], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600)
            except Exception as e:
                st.error(f"Could not display anomaly plot: {e}")
        
        # Show other output files
        st.markdown("### 📁 Generated Files")
        for key, path in outputs.items():
            if key != 'prophet_anomalies_html' and isinstance(path, str) and os.path.exists(path):
                st.write(f"**{key}:** {path}")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">📊 Time Series Analysis Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Initialize agent button
        if st.button("🚀 Initialize Agent", type="primary"):
            initialize_agent()
        
        # API Key status
        if os.getenv("GROQ_API_KEY"):
            st.success("✅ GROQ API Key found")
        else:
            st.error("❌ GROQ API Key not found")
            st.info("Please set GROQ_API_KEY in your environment")
        
        st.divider()
        
        # File upload section
        st.header("📁 Upload Files")
        
        uploaded_files = st.file_uploader(
            "Upload CSV or PDF files for analysis",
            type=['csv', 'pdf'],
            accept_multiple_files=True,
            help="Upload files to add them to the knowledge base"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                    file_path = save_uploaded_file(uploaded_file, uploaded_file.name.split('.')[-1])
                    st.session_state.uploaded_files.append({
                        'name': uploaded_file.name,
                        'path': file_path,
                        'type': uploaded_file.name.split('.')[-1]
                    })
                    st.success(f"✅ Uploaded: {uploaded_file.name}")
            # Reindex KB so RAG can see new files
            reindex_kb_if_needed()
            # Update selected CSV default
            csvs = [f for f in st.session_state.uploaded_files if f['type'].lower() == 'csv']
            if csvs and not st.session_state.selected_csv:
                st.session_state.selected_csv = csvs[0]['path']
        
        # Show uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📋 Uploaded Files")
            for file_info in st.session_state.uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {file_info['name']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{file_info['name']}"):
                        try:
                            os.remove(file_info['path'])
                            st.session_state.uploaded_files.remove(file_info)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
            # CSV picker for TS analysis
            csv_options = [f['path'] for f in st.session_state.uploaded_files if f['type'].lower() == 'csv']
            if csv_options:
                st.session_state.selected_csv = st.selectbox("Select CSV for analysis", options=csv_options, index=0 if st.session_state.selected_csv not in csv_options else csv_options.index(st.session_state.selected_csv))
        
        st.divider()
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.analysis_results = {}
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        # Render chat messages (chatbot style)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input (appears at bottom, one-line)
        chat_prompt = st.chat_input("Ask a question or request time series analysis…")
        if chat_prompt is not None:
            if st.session_state.agent is None:
                st.error("Please initialize the agent first!")
            else:
                st.session_state.messages.append({"role": "user", "content": chat_prompt})
                with st.chat_message("user"):
                    st.markdown(chat_prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            response = st.session_state.agent.chat(chat_prompt)
                        except Exception as e:
                            response = f"Error: {e}"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("📈 Analysis Dashboard")
        
        # Display analysis results
        if st.session_state.analysis_results:
            display_analysis_results(st.session_state.analysis_results)
        else:
            st.info("No analysis results yet. Try asking for time series analysis!")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        colq1, colq2 = st.columns(2)
        with colq1:
            if st.button("📊 Analyze selected CSV"):
                if st.session_state.agent and st.session_state.selected_csv and os.path.exists(st.session_state.selected_csv):
                    with st.spinner("Analyzing…"):
                        try:
                            analysis = st.session_state.agent._analyze_csv_path(st.session_state.selected_csv)
                            st.session_state.analysis_results = analysis
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                else:
                    st.error("No CSV selected or agent not initialized!")
        with colq2:
            if st.button("📊 Analyze commit_history.csv"):
                if st.session_state.agent and os.path.exists("commit_history.csv"):
                    with st.spinner("Analyzing…"):
                        try:
                            analysis = st.session_state.agent._analyze_csv_path("commit_history.csv")
                            st.session_state.analysis_results = analysis
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                else:
                    st.error("Agent not initialized or file not found!")
        
        if st.button("🔄 Refresh Agent"):
            st.session_state.agent = None
            initialize_agent()
            st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🤖 Hybrid RAG + Time Series Analysis Chatbot | Powered by Groq AI & LangGraph</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
