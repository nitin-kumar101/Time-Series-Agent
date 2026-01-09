"""
Time Series Analysis Chatbot - Streamlit UI
A comprehensive web interface for time series analysis using MCP server tools
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

# MCP imports
from mcp import ClientSession
from mcp.client.sse import sse_client

# Local imports
from mcp_server import (
    perform_comprehensive_ts_analysis,
    forecast_time_series,
    detect_anomalies,
    generate_analysis_report,
    analyze_csv_file,
    upload_pdf,
    search_documents,
    list_documents,
    get_rag_stats,
    generate_rag_answer
)


class TimeSeriesChatbot:
    """Main chatbot class for time series analysis"""

    def __init__(self):
        self.server_url = "http://localhost:8000/sse"
        self.conversation_history = []
        self.current_analysis_results = None
        self.uploaded_file = None

    def add_message(self, role: str, content: str, message_type: str = "text"):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "type": message_type,
            "timestamp": datetime.now()
        })

    def get_available_tools(self) -> Dict[str, str]:
        """Get available MCP tools"""
        return {
            # Time Series Tools
            "analyze_csv": "Analyze basic CSV file structure and data types",
            "comprehensive_analysis": "Perform comprehensive time series analysis (trend, seasonality, stationarity)",
            "forecast": "Generate time series forecasts using ARIMA or Exponential Smoothing",
            "detect_anomalies": "Detect anomalies in time series data using Prophet",
            "generate_report": "Generate detailed HTML and JSON analysis reports",
            # RAG Tools
            "upload_pdf": "Upload and process PDF documents for RAG system",
            "search_documents": "Search documents with AI-powered answer generation",
            "generate_rag_answer": "Generate comprehensive answers from document context",
            "list_documents": "List all documents available in the RAG system",
            "rag_stats": "Get statistics about the RAG system"
        }

    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server tool"""
        try:
            async with sse_client(url=self.server_url) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()

                    result = await session.call_tool(tool_name, arguments=arguments)
                    response = json.loads(result.content[0].text)
                    return response
        except Exception as e:
            return {"error": f"Failed to call MCP tool: {str(e)}"}

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and determine appropriate action"""
        user_input = user_input.lower().strip()

        # RAG Commands
        if any(keyword in user_input for keyword in ["upload", "pdf", "document"]):
            return {"action": "upload_pdf_help", "message": "Please use the PDF upload section in the sidebar to upload documents."}

        elif any(keyword in user_input for keyword in ["search", "find", "query"]) and not self.uploaded_file:
            # Extract search query
            query = user_input
            generate = False

            # Check if user wants generation
            if any(gen_word in user_input.lower() for gen_word in ["answer", "explain", "tell me", "summarize"]):
                generate = True

            for prefix in ["search for", "find", "query", "look for", "answer about", "tell me about", "explain"]:
                if prefix in user_input:
                    query = user_input.split(prefix, 1)[1].strip()
                    break

            return {"action": "search_documents", "query": query, "generate_answer": generate}

        elif any(keyword in user_input for keyword in ["list", "show", "documents", "files"]):
            return {"action": "list_documents"}

        elif any(keyword in user_input for keyword in ["stats", "statistics", "rag stats"]):
            return {"action": "rag_stats"}

        # Time Series Commands
        elif self.uploaded_file and any(keyword in user_input for keyword in ["analyze", "analysis", "comprehensive"]):
            if "comprehensive" in user_input:
                return {"action": "comprehensive_analysis", "file_path": str(self.uploaded_file)}
            else:
                return {"action": "analyze_csv", "file_path": str(self.uploaded_file)}

        # Check for forecasting
        elif self.uploaded_file and any(keyword in user_input for keyword in ["forecast", "predict", "future"]):
            periods = 12  # default
            method = "arima"  # default

            if "exponential" in user_input:
                method = "exponential_smoothing"

            # Try to extract number of periods
            import re
            period_match = re.search(r'(\d+)\s*(period|month|year)', user_input)
            if period_match:
                periods = int(period_match.group(1))
                if "year" in period_match.group(2):
                    periods *= 12

            return {
                "action": "forecast",
                "file_path": str(self.uploaded_file),
                "method": method,
                "periods": periods
            }

        # Check for anomaly detection
        elif self.uploaded_file and any(keyword in user_input for keyword in ["anomal", "outlier", "unusual"]):
            return {"action": "detect_anomalies", "file_path": str(self.uploaded_file)}

        # Check for report generation
        elif self.uploaded_file and any(keyword in user_input for keyword in ["report", "summary", "html"]):
            return {"action": "generate_report", "file_path": str(self.uploaded_file)}

        # Default response
        return {"action": "help", "message": "Please upload a CSV file for time series analysis or a PDF for document analysis, then ask me questions!"}


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Time Series Analysis Chatbot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 AI Analysis Chatbot")
    st.markdown("*Time Series Analysis & Document Q&A powered by MCP Server Tools*")

    # Check for Azure OpenAI credentials
    import os
    if not (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")):
        st.warning("⚠️ **Azure OpenAI credentials not set**: AI-powered answer generation for documents will be disabled. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT as environment variables to enable this feature.")
        st.info("💡 Configure your Azure OpenAI resource in the Azure portal and set the environment variables.")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = TimeSeriesChatbot()

    chatbot = st.session_state.chatbot

    # Sidebar for file upload and tools
    with st.sidebar:
        st.header("📁 File Upload")

        # CSV Upload
        st.subheader("📊 Time Series Data (CSV)")
        csv_file = st.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")

        if csv_file is not None:
            # Save uploaded file temporarily
            temp_path = Path("temp_upload.csv")
            with open(temp_path, "wb") as f:
                f.write(csv_file.getvalue())

            chatbot.uploaded_file = temp_path
            st.success(f"✅ CSV uploaded: {csv_file.name}")

            # Display basic file info
            df = pd.read_csv(temp_path)
            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write("**Column names:**", ", ".join(df.columns.tolist()[:5]))
            if len(df.columns) > 5:
                st.write(f"... and {len(df.columns) - 5} more")

        st.markdown("---")

        # PDF Upload for RAG
        st.subheader("📄 Documents (PDF)")
        pdf_file = st.file_uploader("Upload PDF for RAG", type=['pdf'], key="pdf_uploader")

        if pdf_file is not None:
            # Save uploaded PDF temporarily
            temp_pdf_path = Path("temp_upload.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())

            st.success(f"✅ PDF uploaded: {pdf_file.name}")

            # Upload to RAG system
            if st.button("Process PDF for RAG", key="process_pdf"):
                with st.spinner("Processing PDF and adding to knowledge base..."):
                    try:
                        result = upload_pdf(str(temp_pdf_path), pdf_file.name)
                        if "error" in result:
                            st.error(f"❌ Error: {result['error']}")
                        else:
                            st.success("✅ PDF processed successfully!")
                            st.write(f"**Document ID:** {result.get('document_id', 'N/A')}")
                            st.write(f"**Chunks created:** {result.get('chunks_created', 0)}")
                            # Clean up temp file
                            temp_pdf_path.unlink(missing_ok=True)
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")

        # RAG System Stats
        st.markdown("---")
        st.subheader("📊 RAG System")
        if st.button("Show RAG Stats", key="rag_stats_btn"):
            with st.spinner("Getting RAG system statistics..."):
                try:
                    result = get_rag_stats()
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        stats = result.get("statistics", {})
                        st.write(f"**Documents:** {stats.get('total_documents', 0)}")
                        st.write(f"**Chunks:** {stats.get('total_chunks', 0)}")
                        st.write(f"**Storage:** {stats.get('storage_size_mb', 0):.2f} MB")
                except Exception as e:
                    st.error(f"❌ Error getting stats: {str(e)}")

        if st.button("List Documents", key="list_docs_btn"):
            with st.spinner("Getting document list..."):
                try:
                    result = list_documents()
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        docs = result.get("documents", [])
                        if not docs:
                            st.info("No documents uploaded yet.")
                        else:
                            st.write(f"**Total Documents:** {result.get('total_documents', 0)}")
                            for doc in docs[:5]:  # Show first 5
                                st.write(f"• **{doc['name']}** ({doc['chunk_count']} chunks)")
                            if len(docs) > 5:
                                st.write(f"... and {len(docs) - 5} more")
                except Exception as e:
                    st.error(f"❌ Error listing documents: {str(e)}")

        # Quick search
        st.markdown("---")
        st.subheader("🔍 Quick Search")
        search_query = st.text_input("Search documents", key="quick_search", placeholder="Ask a question about your documents...")
        generate_answer = st.checkbox("Generate AI answer", value=False, key="generate_checkbox",
                                    help="Use AI to generate a comprehensive answer based on search results (requires Azure OpenAI credentials)")

        if st.button("Search", key="quick_search_btn") and search_query:
            with st.spinner(f"Searching for: '{search_query}'{' with AI generation' if generate_answer else ''}"):
                try:
                    result = search_documents(search_query, top_k=5, generate_answer=generate_answer)
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.success(f"Found {result.get('total_results', 0)} results")

                        # Display generated answer if available
                        if generate_answer and "generated_answer" in result:
                            st.markdown("### 🤖 AI Generated Answer")
                            st.info(result["generated_answer"])

                            with st.expander("View Sources"):
                                for source in result.get("generation_sources", []):
                                    st.write(f"**Source {source['source_id']}:** {source['document_name']} (relevance: {source['relevance_score']})")
                                    st.write(f"*{source['text_preview']}*")
                                    st.write("---")

                        # Display raw search results
                        if result.get("results"):
                            with st.expander("View Raw Search Results", expanded=not generate_answer):
                                for i, res in enumerate(result.get("results", [])[:3], 1):
                                    st.write(f"**Result {i}** - {res.get('document_name', 'Unknown')} (Score: {res.get('score', 0):.3f})")
                                    st.write(res.get('text', '')[:200] + "..." if len(res.get('text', '')) > 200 else res.get('text', ''))
                                    st.write("---")

                        # Show generation error if any
                        if generate_answer and "generation_error" in result:
                            st.warning(f"⚠️ Answer generation failed: {result['generation_error']}")

                except Exception as e:
                    st.error(f"❌ Error searching: {str(e)}")

        st.header("🛠️ Available Tools")

        # Group tools by category
        tools = chatbot.get_available_tools()

        st.subheader("📊 Time Series Analysis")
        ts_tools = {k: v for k, v in tools.items() if k in ["analyze_csv", "comprehensive_analysis", "forecast", "detect_anomalies", "generate_report"]}
        for tool_name, description in ts_tools.items():
            st.write(f"**{tool_name.replace('_', ' ').title()}:**")
            st.write(f"*{description}*")
            st.write("---")

        st.subheader("📄 Document Analysis (RAG)")
        rag_tools = {k: v for k, v in tools.items() if k in ["upload_pdf", "search_documents", "list_documents", "rag_stats"]}
        for tool_name, description in rag_tools.items():
            st.write(f"**{tool_name.replace('_', ' ').title()}:**")
            st.write(f"*{description}*")
            st.write("---")

    # Main chat interface
    st.header("💬 Chat Interface")

    # Display conversation history
    chat_container = st.container()

    with chat_container:
        for message in chatbot.conversation_history[-10:]:  # Show last 10 messages
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                if message["type"] == "analysis_result":
                    display_analysis_result(message["content"])
                elif message["type"] == "report":
                    display_report(message["content"])
                else:
                    st.markdown(f"**Assistant:** {message['content']}")

    # User input
    user_input = st.text_input("Ask me about your time series data:", key="user_input")

    if st.button("Send", key="send_button") and user_input:
        # Add user message
        chatbot.add_message("user", user_input)

        # Process user input
        action = chatbot.process_user_input(user_input)

        # Execute action
        if action["action"] == "help":
            response = action["message"]
            chatbot.add_message("assistant", response)

        elif action["action"] == "upload_pdf_help":
            response = action["message"]
            chatbot.add_message("assistant", response)

        elif action["action"] == "search_documents":
            generate = action.get("generate_answer", True)  # Default to generating answers
            spinner_text = f"Searching documents for: '{action['query']}'{' with AI generation' if generate else ''}"

            with st.spinner(spinner_text):
                try:
                    result = search_documents(action["query"], top_k=5, generate_answer=generate)
                    if "error" in result:
                        chatbot.add_message("assistant", f"❌ Error: {result['error']}")
                    else:
                        total_results = result.get('total_results', 0)

                        if generate and "generated_answer" in result:
                            # Show generated answer
                            answer = result["generated_answer"]
                            chatbot.add_message("assistant", f"🤖 **AI Answer for '{action['query']}':**\n\n{answer}", "rag_answer")
                        elif total_results > 0:
                            # Show raw search results
                            chatbot.add_message("assistant", f"✅ Found {total_results} relevant results for '{action['query']}'", "search_results")
                        else:
                            chatbot.add_message("assistant", f"📄 No relevant documents found for '{action['query']}'. Try rephrasing or upload more documents.")

                        chatbot.current_analysis_results = result
                except Exception as e:
                    chatbot.add_message("assistant", f"❌ Error searching documents: {str(e)}")

        elif action["action"] == "list_documents":
            with st.spinner("Getting document list..."):
                try:
                    result = list_documents()
                    if "error" in result:
                        chatbot.add_message("assistant", f"❌ Error: {result['error']}")
                    else:
                        docs = result.get("documents", [])
                        if not docs:
                            chatbot.add_message("assistant", "📄 No documents uploaded yet. Please upload a PDF first.")
                        else:
                            chatbot.add_message("assistant", f"📄 Found {result.get('total_documents', 0)} documents in the system", "document_list")
                            chatbot.current_analysis_results = result
                except Exception as e:
                    chatbot.add_message("assistant", f"❌ Error listing documents: {str(e)}")

        elif action["action"] == "rag_stats":
            with st.spinner("Getting RAG system statistics..."):
                try:
                    result = get_rag_stats()
                    if "error" in result:
                        chatbot.add_message("assistant", f"❌ Error: {result['error']}")
                    else:
                        stats = result.get("statistics", {})
                        response = f"📊 RAG System Stats:\n"
                        response += f"• Documents: {stats.get('total_documents', 0)}\n"
                        response += f"• Chunks: {stats.get('total_chunks', 0)}\n"
                        response += f"• Storage: {stats.get('storage_size_mb', 0):.2f} MB"
                        chatbot.add_message("assistant", response, "rag_stats")
                        chatbot.current_analysis_results = result
                except Exception as e:
                    chatbot.add_message("assistant", f"❌ Error getting RAG stats: {str(e)}")

        elif action["action"] == "analyze_csv":
            with st.spinner("Analyzing CSV file..."):
                # Call MCP tool directly (synchronous version)
                try:
                    result = analyze_csv_file(action["file_path"])
                    if "error" in result:
                        chatbot.add_message("assistant", f"❌ Error: {result['error']}")
                    else:
                        chatbot.add_message("assistant", "✅ CSV analysis completed!", "analysis_result")
                        chatbot.current_analysis_results = result
                except Exception as e:
                    chatbot.add_message("assistant", f"❌ Error calling analysis tool: {str(e)}")

        elif action["action"] == "comprehensive_analysis":
            with st.spinner("Performing comprehensive time series analysis..."):
                try:
                    result = perform_comprehensive_ts_analysis(action["file_path"])
                    if "error" in result:
                        chatbot.add_message("assistant", f"❌ Error: {result['error']}")
                    else:
                        chatbot.add_message("assistant", "✅ Comprehensive analysis completed!", "analysis_result")
                        chatbot.current_analysis_results = result
                except Exception as e:
                    chatbot.add_message("assistant", f"❌ Error calling comprehensive analysis: {str(e)}")

        elif action["action"] == "forecast":
            with st.spinner(f"Generating {action['periods']}-period forecast using {action['method']}..."):
                try:
                    result = forecast_time_series(
                        action["file_path"],
                        periods=action["periods"],
                        method=action["method"]
                    )
                    if "error" in result:
                        chatbot.add_message("assistant", f"❌ Error: {result['error']}")
                    else:
                        chatbot.add_message("assistant", f"✅ Forecast generated using {action['method'].upper()}!", "analysis_result")
                        chatbot.current_analysis_results = result
                except Exception as e:
                    chatbot.add_message("assistant", f"❌ Error calling forecast tool: {str(e)}")

        elif action["action"] == "detect_anomalies":
            with st.spinner("Detecting anomalies..."):
                try:
                    result = detect_anomalies(action["file_path"], method="prophet")
                    if "error" in result:
                        chatbot.add_message("assistant", f"❌ Error: {result['error']}")
                    else:
                        chatbot.add_message("assistant", "✅ Anomaly detection completed!", "analysis_result")
                        chatbot.current_analysis_results = result
                except Exception as e:
                    chatbot.add_message("assistant", f"❌ Error calling anomaly detection: {str(e)}")

        elif action["action"] == "generate_report":
            with st.spinner("Generating analysis report..."):
                try:
                    result = generate_analysis_report(action["file_path"], "reports")
                    if "error" in result:
                        chatbot.add_message("assistant", f"❌ Error: {result['error']}")
                    else:
                        chatbot.add_message("assistant", "✅ Report generated successfully!", "report")
                        chatbot.current_analysis_results = result
                except Exception as e:
                    chatbot.add_message("assistant", f"❌ Error generating report: {str(e)}")

        # Clear input
        st.rerun()

    # Analysis Results Display
    if chatbot.current_analysis_results:
        st.header("📊 Results")

        result = chatbot.current_analysis_results

        # Handle RAG generated answers
        if "generated_answer" in result:
            st.subheader("🤖 AI Generated Answer")
            st.success(result["generated_answer"])

            if "generation_sources" in result:
                with st.expander("📚 Sources Used", expanded=False):
                    sources = result["generation_sources"]
                    for source in sources:
                        st.write(f"**Source {source['source_id']}:** {source['document_name']}")
                        st.write(f"*Relevance: {source['relevance_score']}*")
                        st.write(f"Preview: {source['text_preview']}")
                        st.write("---")

            if "generation_error" in result:
                st.warning(f"⚠️ Generation note: {result['generation_error']}")

        # Handle RAG search results (raw)
        if "results" in result and isinstance(result.get("results"), list):
            st.subheader("🔍 Search Results")
            search_results = result["results"]
            if not search_results:
                st.info("No relevant documents found for your query.")
            else:
                st.write(f"**Total Results:** {result.get('total_results', 0)}")

                # Show summary
                if len(search_results) > 0:
                    top_result = search_results[0]
                    st.info(f"📄 Top result from **{top_result.get('document_name', 'Unknown')}** (relevance: {top_result.get('score', 0):.3f})")

                # Detailed results
                for i, res in enumerate(search_results[:10], 1):  # Show top 10
                    with st.expander(f"Result {i}: {res.get('document_name', 'Unknown')} (Score: {res.get('score', 0):.3f})"):
                        st.write(f"**Document:** {res.get('document_name', 'Unknown')}")
                        st.write(f"**Score:** {res.get('score', 0):.3f}")
                        st.write(f"**Text:** {res.get('text', '')[:500]}...")
                        if len(res.get('text', '')) > 500:
                            st.write("... (truncated)")

        # Handle document list
        elif "documents" in result and isinstance(result.get("documents"), list):
            st.subheader("📄 Available Documents")
            docs = result["documents"]
            if not docs:
                st.info("No documents uploaded yet.")
            else:
                st.write(f"**Total Documents:** {result.get('total_documents', 0)}")

                # Display as a table
                doc_data = []
                for doc in docs:
                    doc_data.append({
                        "Name": doc.get("name", "Unknown"),
                        "Chunks": doc.get("chunk_count", 0),
                        "Uploaded": doc.get("created_at", "Unknown")[:10] if doc.get("created_at") else "Unknown"
                    })

                if doc_data:
                    st.dataframe(pd.DataFrame(doc_data))

        # Handle RAG stats
        elif "statistics" in result:
            st.subheader("📊 RAG System Statistics")
            stats = result["statistics"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", stats.get("total_documents", 0))
            with col2:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
            with col3:
                st.metric("Storage Size", f"{stats.get('storage_size_mb', 0):.2f} MB")

        # Handle Time Series analysis results
        elif "analysis_results" in result:
            analysis = result["analysis_results"]

            # Create tabs for different analysis components
            tab_names = []
            tab_contents = []

            if "trend" in analysis:
                tab_names.append("Trend Analysis")
                tab_contents.append(display_trend_analysis(analysis["trend"]))

            if "seasonality" in analysis:
                tab_names.append("Seasonality")
                tab_contents.append(display_seasonality_analysis(analysis["seasonality"]))

            if "stationarity" in analysis:
                tab_names.append("Stationarity")
                tab_contents.append(display_stationarity_analysis(analysis["stationarity"]))

            if tab_names:
                tabs = st.tabs(tab_names)
                for i, (tab, content) in enumerate(zip(tabs, tab_contents)):
                    with tab:
                        st.markdown(content)

        # Forecast results
        if "forecast" in result:
            st.subheader("🔮 Forecast Results")
            forecast_data = result["forecast"]

            if isinstance(forecast_data, dict) and "forecast" in forecast_data:
                forecast_series = forecast_data["forecast"]
                if isinstance(forecast_series, dict) and "values" in forecast_series:
                    # Create forecast plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=forecast_series["index"],
                        y=forecast_series["values"],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title="Time Series Forecast",
                        xaxis_title="Date",
                        yaxis_title="Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Anomaly results
        if "anomalies" in result and isinstance(result["anomalies"], dict):
            anomalies_data = result["anomalies"]
            if "anomalies" in anomalies_data:
                st.subheader("🔍 Anomalies Detected")
                anomaly_info = anomalies_data["anomalies"]

                if isinstance(anomaly_info, dict) and "values" in anomaly_info:
                    anomaly_count = sum(1 for record in anomaly_info["values"] if record.get("is_anomaly", False))
                    st.write(f"**Total anomalies detected:** {anomaly_count}")

                    if anomaly_count > 0:
                        # Display anomalies table
                        anomalies_df = pd.DataFrame(anomaly_info["values"])
                        anomaly_records = anomalies_df[anomalies_df["is_anomaly"] == True]
                        if not anomaly_records.empty:
                            st.dataframe(anomaly_records.head(10))

    # Footer
    st.markdown("---")
    st.markdown("*Time Series Analysis Chatbot - Built with Streamlit and MCP Server*")


def display_analysis_result(content: Dict[str, Any]):
    """Display analysis results in a formatted way"""
    # For simple results, show as formatted text
    if "success" in content and len(content) <= 3:
        if content.get("success"):
            st.success("✅ Operation completed successfully!")
        else:
            st.error("❌ Operation failed")
    else:
        # For complex results, show as expandable JSON
        with st.expander("View Raw Results"):
            st.json(content)


def display_report(content: Dict[str, Any]):
    """Display report information"""
    if "report_files" in content:
        st.success("Report generated successfully!")
        for file_path in content["report_files"]:
            st.write(f"📄 {Path(file_path).name}")

        # Try to display HTML report inline
        html_file = next((f for f in content["report_files"] if f.endswith('.html')), None)
        if html_file and Path(html_file).exists():
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)


def display_trend_analysis(trend_data: Dict[str, Any]) -> str:
    """Display trend analysis results"""
    content = "### Trend Analysis\n\n"

    if "trend_direction" in trend_data:
        direction = trend_data["trend_direction"]
        strength = trend_data.get("trend_strength", 0)
        slope = trend_data.get("trend_slope", 0)

        content += f"**Direction:** {direction.title()}\n\n"
        content += f"**Strength:** {strength:.3f}\n\n"
        content += f"**Slope:** {slope:.6f}\n\n"

        # Add interpretation
        if strength > 0.7:
            content += "📈 **Strong trend detected**\n"
        elif strength > 0.3:
            content += "📊 **Moderate trend detected**\n"
        else:
            content += "📉 **Weak or no significant trend**\n"

    return content


def display_seasonality_analysis(seasonality_data: Dict[str, Any]) -> str:
    """Display seasonality analysis results"""
    content = "### Seasonality Analysis\n\n"

    detected = seasonality_data.get("seasonality_detected", False)
    period = seasonality_data.get("period")
    strength = seasonality_data.get("seasonal_strength", 0)

    content += f"**Seasonality Detected:** {'Yes' if detected else 'No'}\n\n"

    if period:
        content += f"**Period:** {period} time units\n\n"

    content += f"**Strength:** {strength:.3f}\n\n"

    if detected and strength > 0.5:
        content += "🔄 **Strong seasonal pattern detected**\n"
    elif detected:
        content += "🔄 **Seasonal pattern detected**\n"
    else:
        content += "➡️ **No significant seasonality**\n"

    return content


def display_stationarity_analysis(stationarity_data: Dict[str, Any]) -> str:
    """Display stationarity analysis results"""
    content = "### Stationarity Analysis\n\n"

    is_stationary = stationarity_data.get("is_stationary", False)

    content += f"**Stationary:** {'Yes' if is_stationary else 'No'}\n\n"

    # ADF test results
    adf_data = stationarity_data.get("adf_test", {})
    if adf_data:
        p_value = adf_data.get("p_value", 1)
        content += f"**ADF Test p-value:** {p_value:.4f}\n\n"
        content += f"**ADF Test Result:** {'Stationary' if p_value < 0.05 else 'Non-stationary'}\n\n"

    # KPSS test results
    kpss_data = stationarity_data.get("kpss_test", {})
    if kpss_data:
        p_value = kpss_data.get("p_value", 1)
        content += f"**KPSS Test p-value:** {p_value:.4f}\n\n"
        content += f"**KPSS Test Result:** {'Non-stationary' if p_value < 0.05 else 'Stationary'}\n\n"

    if is_stationary:
        content += "✅ **Time series is stationary - good for forecasting**\n"
    else:
        content += "⚠️ **Time series is non-stationary - may need differencing**\n"

    return content


if __name__ == "__main__":
    main()