#!/usr/bin/env python3
"""
MCP UI - Streamlit interface for Time Series Analysis
Provides a beautiful web interface that uses the time series analysis tools directly
"""

import streamlit as st
import os
import sys
from typing import Dict, Any, List, Optional
import pandas as pd
import tempfile
import time

# Import our analysis modules directly
sys.path.insert(0, os.path.dirname(__file__))
from ts_agent import HybridChatAgent
from data_detector import DataAnalyzer
from time_series_tools import TimeSeriesAnalyzer, TimeSeriesVisualizer
from reporting import ReportGenerator, ExportManager


class DirectAnalysisClient:
    """Direct client that uses time series analysis tools without MCP protocol"""

    def __init__(self):
        self.connected = False
        self.tools = []
        self.agent = None
        self.data_analyzer = None
        self.ts_analyzer = None

    def connect(self) -> bool:
        """Initialize the analysis tools"""
        try:
            # Set dummy API key if not present (for demo purposes)
            if not os.environ.get('GROQ_API_KEY'):
                os.environ['GROQ_API_KEY'] = ''

            self.agent = HybridChatAgent()
            self.data_analyzer = DataAnalyzer()
            self.ts_analyzer = TimeSeriesAnalyzer()
            self.connected = True

            # Define available tools
            self.tools = [
                {
                    "name": "analyze_csv_file",
                    "description": "Analyze a CSV file for time series patterns, trends, seasonality, and anomalies",
                    "inputSchema": {
                        "properties": {
                            "csv_path": {
                                "type": "string",
                                "description": "Path to the CSV file to analyze"
                            },
                            "output_dir": {
                                "type": "string",
                                "description": "Directory to save analysis outputs (optional)",
                                "default": "output/ts"
                            }
                        },
                        "required": ["csv_path"]
                    }
                },
                {
                    "name": "detect_data_types",
                    "description": "Detect and analyze data types in a CSV file",
                    "inputSchema": {
                        "properties": {
                            "csv_path": {
                                "type": "string",
                                "description": "Path to the CSV file to analyze"
                            }
                        },
                        "required": ["csv_path"]
                    }
                },
                {
                    "name": "analyze_time_series",
                    "description": "Perform comprehensive time series analysis on a prepared dataset",
                    "inputSchema": {
                        "properties": {
                            "csv_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            },
                            "column": {
                                "type": "string",
                                "description": "Column name to analyze (optional, will auto-detect if not provided)"
                            },
                            "analysis_type": {
                                "type": "string",
                                "description": "Type of analysis: 'trend', 'seasonality', 'forecast', 'anomaly', or 'comprehensive'",
                                "default": "comprehensive",
                                "enum": ["trend", "seasonality", "forecast", "anomaly", "comprehensive"]
                            }
                        },
                        "required": ["csv_path"]
                    }
                },
                {
                    "name": "forecast_time_series",
                    "description": "Generate forecasts using ARIMA or Exponential Smoothing",
                    "inputSchema": {
                        "properties": {
                            "csv_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            },
                            "column": {
                                "type": "string",
                                "description": "Column name to forecast (optional)"
                            },
                            "periods": {
                                "type": "integer",
                                "description": "Number of periods to forecast",
                                "default": 30
                            },
                            "method": {
                                "type": "string",
                                "description": "Forecasting method: 'arima' or 'exponential'",
                                "default": "arima",
                                "enum": ["arima", "exponential"]
                            }
                        },
                        "required": ["csv_path"]
                    }
                },
                {
                    "name": "detect_anomalies",
                    "description": "Detect anomalies in time series data using Prophet",
                    "inputSchema": {
                        "properties": {
                            "csv_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            },
                            "column": {
                                "type": "string",
                                "description": "Column name to analyze (optional)"
                            },
                            "interval_width": {
                                "type": "number",
                                "description": "Confidence interval width (0.95-0.99)",
                                "default": 0.99,
                                "minimum": 0.8,
                                "maximum": 0.99
                            }
                        },
                        "required": ["csv_path"]
                    }
                },
                {
                    "name": "query_documents",
                    "description": "Query documents in the knowledge base using RAG",
                    "inputSchema": {
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to ask about the documents"
                            }
                        },
                        "required": ["question"]
                    }
                },
                {
                    "name": "list_available_data",
                    "description": "List available CSV files and their basic information",
                    "inputSchema": {
                        "properties": {}
                    }
                }
            ]

            return True
        except Exception as e:
            st.error(f"Failed to initialize analysis tools: {e}")
            return False

    def connect_sync(self) -> bool:
        """Connect to analysis tools directly"""
        return self.connect()

    def disconnect(self):
        """Disconnect from analysis tools"""
        self.connected = False
        self.agent = None
        self.data_analyzer = None
        self.ts_analyzer = None

    def stop_server_sync(self):
        """Stop the MCP server process synchronously"""
        if self.server_process:
            try:
                self.server_process.terminate()
                # Give it a moment to terminate gracefully
                import time
                time.sleep(1)
                if self.server_process.poll() is None:
                    self.server_process.kill()
            except Exception:
                pass  # Ignore errors in cleanup

    async def stop_server(self):
        """Stop the MCP server process"""
        self.stop_server_sync()

    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the server"""
        if not self.writer:
            raise RuntimeError("Server not connected")

        # Send request
        request_json = json.dumps(request) + "\n"
        self.writer.write(request_json.encode())
        await self.writer.drain()

        # Read response with timeout
        try:
            response_line = await asyncio.wait_for(
                self.reader.readline(), timeout=30.0
            )
            if not response_line:
                raise RuntimeError("Server closed connection")

            response = json.loads(response_line.decode().strip())
            return response
        except asyncio.TimeoutError:
            raise RuntimeError("Request timed out")
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    async def initialize(self) -> bool:
        """Initialize the MCP connection"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "mcp-ui-client",
                    "version": "1.0.0"
                }
            }
        }

        try:
            response = await self.send_request(request)
            return "result" in response
        except Exception:
            return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        try:
            response = await self.send_request(request)
            if "result" in response:
                return response["result"].get("tools", [])
            return []
        except Exception:
            return []

    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool directly using the analysis functions"""
        try:
            if tool_name == "analyze_csv_file":
                return self._analyze_csv_file(**arguments)
            elif tool_name == "detect_data_types":
                return self._detect_data_types(**arguments)
            elif tool_name == "analyze_time_series":
                return self._analyze_time_series(**arguments)
            elif tool_name == "forecast_time_series":
                return self._forecast_time_series(**arguments)
            elif tool_name == "detect_anomalies":
                return self._detect_anomalies(**arguments)
            elif tool_name == "query_documents":
                return self._query_documents(**arguments)
            elif tool_name == "list_available_data":
                return self._list_available_data(**arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def _analyze_csv_file(self, csv_path: str, output_dir: str = "output/ts") -> Dict[str, Any]:
        """Analyze a CSV file comprehensively"""
        if not os.path.exists(csv_path):
            return {"error": f"CSV file not found: {csv_path}"}

        try:
            # Use the agent's analysis method
            analysis = self.agent._analyze_csv_path(csv_path)
            summary = analysis['summary']

            # Add output file information
            outputs = analysis.get('outputs', {})
            if outputs:
                summary += f"\n\nGenerated files:\n"
                for key, path in outputs.items():
                    if isinstance(path, str) and os.path.exists(path):
                        summary += f"- {key}: {path}\n"

            return {"content": [{"type": "text", "text": summary}]}
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _detect_data_types(self, csv_path: str) -> Dict[str, Any]:
        """Detect data types in a CSV file"""
        if not os.path.exists(csv_path):
            return {"error": f"CSV file not found: {csv_path}"}

        try:
            data_info = self.data_analyzer.analyze_csv(csv_path)

            if "error" in data_info:
                return {"error": f"Data analysis error: {data_info['error']}"}

            # Format the results
            result = "Data Analysis Results:\n\n"
            result += f"Total records: {data_info.get('total_records', 'N/A')}\n"
            result += f"Columns: {len(data_info.get('columns', []))}\n\n"

            result += "Time column: "
            if data_info.get('time_column'):
                result += f"{data_info['time_column']}\n"
                time_range = data_info.get('time_range', {})
                if time_range:
                    result += f"Time range: {time_range.get('start', 'N/A')} to {time_range.get('end', 'N/A')}\n"
            else:
                result += "Not detected\n"

            result += f"\nNumeric columns: {', '.join(data_info.get('numeric_columns', []))}\n"
            result += f"Categorical columns: {', '.join(data_info.get('categorical_columns', []))}\n"

            missing = data_info.get('missing_values', {})
            if missing:
                result += "\nMissing values:\n"
                for col, count in missing.items():
                    result += f"  {col}: {count}\n"

            return {"content": [{"type": "text", "text": result}]}
        except Exception as e:
            return {"error": f"Data type detection failed: {str(e)}"}

    def _analyze_time_series(self, csv_path: str, column: str = None, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform time series analysis"""
        if not os.path.exists(csv_path):
            return {"error": f"CSV file not found: {csv_path}"}

        try:
            # Load and prepare data
            df = pd.read_csv(csv_path)
            from data_detector import DataTypeDetector
            detector = DataTypeDetector()
            info = detector.prepare_time_series_data(df)

            prepared_data = info.get('prepared_data', df)
            time_col = info.get('time_column')
            numeric_cols = info.get('numeric_columns', [])

            if not isinstance(prepared_data.index, pd.DatetimeIndex):
                return {"error": "No valid time series data found"}

            if not numeric_cols:
                return {"error": "No numeric columns found for analysis"}

            # Select column
            if not column:
                column = numeric_cols[0]

            if column not in prepared_data.columns:
                return {"error": f"Column '{column}' not found"}

            series = prepared_data[column].dropna()

            # Perform analysis based on type
            if analysis_type == "trend":
                result = self.ts_analyzer.analyze_trend(series)
                summary = "Trend Analysis:\n"
                summary += f"Direction: {result['trend_direction']}\n"
                summary += f"Strength: {result['trend_strength']:.3f}\n"
                summary += f"Slope: {result['trend_slope']:.6f}\n"

            elif analysis_type == "seasonality":
                result = self.ts_analyzer.detect_seasonality(series)
                summary = "Seasonality Analysis:\n"
                summary += f"Detected: {result['seasonality_detected']}\n"
                if result['seasonality_detected']:
                    summary += f"Period: {result.get('period', 'N/A')}\n"
                    summary += f"Strength: {result['seasonal_strength']:.3f}\n"

            elif analysis_type == "forecast":
                arima_result = self.ts_analyzer.forecast_arima(series, periods=30)
                summary = "Forecast Analysis:\n"
                if "error" not in arima_result:
                    summary += "ARIMA forecast available\n"
                else:
                    summary += f"ARIMA error: {arima_result['error']}\n"

            elif analysis_type == "anomaly":
                result = self.ts_analyzer.detect_anomalies_prophet(series)
                if isinstance(result, dict) and "anomalies" in result:
                    count = len(result['anomalies'])
                    summary = f"Anomaly Detection: {count} anomalies detected"
                else:
                    summary = "Anomaly detection failed"

            else:  # comprehensive
                results = self.ts_analyzer.comprehensive_analysis(series)
                summary = self.agent.get_analysis_summary({column: results})

            return {"content": [{"type": "text", "text": summary}]}
        except Exception as e:
            return {"error": f"Time series analysis failed: {str(e)}"}

    def _forecast_time_series(self, csv_path: str, column: str = None, periods: int = 30, method: str = "arima") -> Dict[str, Any]:
        """Generate time series forecasts"""
        if not os.path.exists(csv_path):
            return {"error": f"CSV file not found: {csv_path}"}

        try:
            # Load and prepare data
            df = pd.read_csv(csv_path)
            from data_detector import DataTypeDetector
            detector = DataTypeDetector()
            info = detector.prepare_time_series_data(df)

            prepared_data = info.get('prepared_data', df)
            numeric_cols = info.get('numeric_columns', [])

            if not column and numeric_cols:
                column = numeric_cols[0]

            if column not in prepared_data.columns:
                return {"error": f"Column '{column}' not found"}

            series = prepared_data[column].dropna()

            # Generate forecast
            if method == "arima":
                result = self.ts_analyzer.forecast_arima(series, periods=periods)
            else:  # exponential
                result = self.ts_analyzer.forecast_exponential_smoothing(series, periods=periods)

            if "error" in result:
                return {"error": f"Forecast failed: {result['error']}"}

            # Format results
            forecast = result['forecast']
            summary = f"{method.upper()} Forecast Results:\n"
            summary += f"Forecast periods: {periods}\n"
            summary += f"Forecast range: {forecast.index[0]} to {forecast.index[-1]}\n"
            summary += f"Mean forecast value: {forecast.mean():.3f}\n"

            if method == "arima":
                summary += f"AIC: {result.get('aic', 'N/A')}\n"
                summary += f"BIC: {result.get('bic', 'N/A')}\n"

            return {"content": [{"type": "text", "text": summary}]}
        except Exception as e:
            return {"error": f"Forecast generation failed: {str(e)}"}

    def _detect_anomalies(self, csv_path: str, column: str = None, interval_width: float = 0.99) -> Dict[str, Any]:
        """Detect anomalies in time series"""
        if not os.path.exists(csv_path):
            return {"error": f"CSV file not found: {csv_path}"}

        try:
            # Load and prepare data
            df = pd.read_csv(csv_path)
            from data_detector import DataTypeDetector
            detector = DataTypeDetector()
            info = detector.prepare_time_series_data(df)

            prepared_data = info.get('prepared_data', df)
            numeric_cols = info.get('numeric_columns', [])

            if not column and numeric_cols:
                column = numeric_cols[0]

            if column not in prepared_data.columns:
                return {"error": f"Column '{column}' not found"}

            series = prepared_data[column].dropna()

            # Detect anomalies
            result = self.ts_analyzer.detect_anomalies_prophet(series, interval_width)

            if not isinstance(result, dict) or "anomalies" not in result:
                return {"error": "Anomaly detection failed"}

            anomalies_df = result['anomalies']
            anomaly_count = len(anomalies_df)

            summary = f"Anomaly Detection Results:\n"
            summary += f"Anomalies detected: {anomaly_count}\n"
            summary += f"Confidence interval: {interval_width}\n"
            summary += f"Total data points: {len(series)}\n"
            summary += f"Anomaly percentage: {(anomaly_count / len(series) * 100):.2f}%\n"

            if anomaly_count > 0:
                summary += "\nAnomaly timestamps:\n"
                for idx in anomalies_df.head(10).index:  # Show first 10
                    summary += f"- {idx}\n"
                if anomaly_count > 10:
                    summary += f"... and {anomaly_count - 10} more\n"

            return {"content": [{"type": "text", "text": summary}]}
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}

    def _query_documents(self, question: str) -> Dict[str, Any]:
        """Query documents using RAG"""
        try:
            response = self.agent.chat(question)
            return {"content": [{"type": "text", "text": response}]}
        except Exception as e:
            return {"error": f"Document query failed: {str(e)}"}

    def _list_available_data(self) -> Dict[str, Any]:
        """List available CSV files"""
        data_files = []

        # Check root directory
        for file in os.listdir("."):
            if file.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    data_files.append({
                        'path': file,
                        'rows': len(df),
                        'columns': list(df.columns),
                        'size_mb': os.path.getsize(file) / (1024 * 1024)
                    })
                except:
                    continue

        # Check db directory
        db_dir = "db"
        if os.path.exists(db_dir):
            for file in os.listdir(db_dir):
                if file.lower().endswith('.csv'):
                    try:
                        path = os.path.join(db_dir, file)
                        df = pd.read_csv(path)
                        data_files.append({
                            'path': path,
                            'rows': len(df),
                            'columns': list(df.columns),
                            'size_mb': os.path.getsize(path) / (1024 * 1024)
                        })
                    except:
                        continue

        if not data_files:
            return {"content": [{"type": "text", "text": "No CSV files found"}]}

        result = "Available CSV Files:\n\n"
        for file_info in data_files:
            result += f"📄 {file_info['path']}\n"
            result += f"   Rows: {file_info['rows']}, Columns: {len(file_info['columns'])}\n"
            result += f"   Size: {file_info['size_mb']:.2f} MB\n"
            result += f"   Columns: {', '.join(file_info['columns'][:5])}\n"
            if len(file_info['columns']) > 5:
                result += f"   ... and {len(file_info['columns']) - 5} more\n"
            result += "\n"

        return {"content": [{"type": "text", "text": result}]}


# Global analysis client instance
analysis_client = DirectAnalysisClient()


def main():
    """Main Streamlit application"""
    # Check if running with proper Streamlit context
    try:
        st.set_page_config(
            page_title="Time Series Analysis MCP",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        st.error("❌ This app must be run with Streamlit!")
        st.error("Please use: `streamlit run mcp_ui.py`")
        st.error("Or use: `python run_mcp_ui.py`")
        st.stop()
        return

    st.title("📊 Time Series Analysis MCP")
    st.markdown("**Powered by Model Context Protocol**")

    # Sidebar for connection status and controls
    with st.sidebar:
        st.header("🔗 Analysis Tools")

        if not analysis_client.connected:
            if st.button("🚀 Initialize Analysis Tools", type="primary"):
                with st.spinner("Initializing analysis tools..."):
                    try:
                        success = analysis_client.connect_sync()
                        if success:
                            st.success("✅ Analysis tools initialized!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to initialize analysis tools")
                    except Exception as e:
                        st.error(f"❌ Initialization error: {e}")
        else:
            st.success("✅ Analysis Tools Ready")

            # Show available tools count
            if analysis_client.tools:
                st.info(f"🛠️ {len(analysis_client.tools)} tools available")

            if st.button("🔄 Refresh Tools"):
                st.success("Tools are always available!")

            if st.button("🛑 Reset Tools"):
                try:
                    analysis_client.disconnect()
                    st.success("Tools reset!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resetting tools: {e}")

    # Main content area
    if not analysis_client.connected:
        show_welcome_screen()
    else:
        show_tool_interface()


def show_welcome_screen():
    """Show welcome screen when not connected"""
    st.markdown("""
    ## Welcome to Time Series Analysis MCP! 🎉

    This application provides a beautiful web interface to interact with your Time Series Analysis agent through the Model Context Protocol (MCP).

    ### Features:
    - 📊 **Comprehensive Time Series Analysis**
    - 🔍 **Automatic Data Detection**
    - 📈 **Trend & Seasonality Analysis**
    - 🎯 **Anomaly Detection**
    - 🔮 **Forecasting** (ARIMA & Exponential Smoothing)
    - 📁 **Document Querying** (RAG)
    - 📤 **File Upload Support**

    ### How it works:
    1. Click **"Start MCP Server"** in the sidebar
    2. Upload CSV files or use sample data
    3. Select analysis tools and configure parameters
    4. View beautiful visualizations and reports

    ### Sample Data Available:
    - `Electric_Production.csv` - Monthly electric production data
    - `Weather_dataset.csv` - Weather data
    - `commit_history.csv` - Git commit history

    **Click the button in the sidebar to get started! 🚀**
    """)


def show_tool_interface():
    """Show the main tool interface"""
    st.header("🛠️ Available Tools")

    if not analysis_client.tools:
        st.warning("No tools available. Try reinitializing the tools.")
        return

    # Tool selection
    tool_names = [tool["name"] for tool in analysis_client.tools]
    selected_tool = st.selectbox(
        "Select a tool:",
        tool_names,
        help="Choose the analysis tool you want to use"
    )

    # Find selected tool details
    tool_info = next((t for t in analysis_client.tools if t["name"] == selected_tool), None)
    if not tool_info:
        st.error("Tool information not found")
        return

    # Tool description
    st.markdown(f"**Description:** {tool_info['description']}")

    # Tool parameters
    st.subheader("⚙️ Parameters")
    params = tool_info.get("inputSchema", {}).get("properties", {})

    # Build parameter form
    param_values = {}

    for param_name, param_info in params.items():
        param_type = param_info.get("type", "string")
        param_desc = param_info.get("description", "")
        param_required = param_name in tool_info.get("inputSchema", {}).get("required", [])
        param_default = param_info.get("default")

        # Handle different parameter types
        if param_name == "csv_path":
            # Special handling for CSV file selection
            param_values[param_name] = handle_csv_file_selection(param_desc, param_required)
        elif param_type == "string":
            if param_default:
                param_values[param_name] = st.text_input(
                    f"{param_name} { '(required)' if param_required else ''}",
                    value=param_default,
                    help=param_desc
                )
            else:
                param_values[param_name] = st.text_input(
                    f"{param_name} { '(required)' if param_required else ''}",
                    help=param_desc
                )
        elif param_type == "integer":
            param_values[param_name] = st.number_input(
                f"{param_name} { '(required)' if param_required else ''}",
                value=param_default or 0,
                step=1,
                help=param_desc
            )
        elif param_type == "number":
            param_values[param_name] = st.number_input(
                f"{param_name} { '(required)' if param_required else ''}",
                value=param_default or 0.0,
                step=0.1,
                help=param_desc
            )

    # Execute button
    if st.button("🚀 Execute Tool", type="primary", use_container_width=True):
        # Filter out empty optional parameters
        filtered_params = {
            k: v for k, v in param_values.items()
            if v is not None and str(v).strip() != ""
        }

        # Check required parameters
        required_params = tool_info.get("inputSchema", {}).get("required", [])
        missing_params = [p for p in required_params if p not in filtered_params or not filtered_params[p]]

        if missing_params:
            st.error(f"Missing required parameters: {', '.join(missing_params)}")
            return

        # Execute tool
        with st.spinner(f"Executing {selected_tool}..."):
            try:
                result = analysis_client.call_tool_sync(selected_tool, filtered_params)

                # Display results
                display_tool_result(result, selected_tool)

            except Exception as e:
                st.error(f"Tool execution failed: {e}")


def handle_csv_file_selection(description: str, required: bool) -> Optional[str]:
    """Handle CSV file selection with upload option"""
    col1, col2 = st.columns([2, 1])

    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file:",
            type=["csv"],
            help="Upload a CSV file for analysis"
        )

        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name

    with col2:
        # Sample file selection
        sample_files = get_available_csv_files()
        if sample_files:
            selected_sample = st.selectbox(
                "Or use sample data:",
                [""] + sample_files,
                help="Select from available sample CSV files"
            )
            if selected_sample:
                return selected_sample

    return None


def get_available_csv_files() -> List[str]:
    """Get list of available CSV files in the project"""
    csv_files = []

    # Check root directory
    for file in os.listdir("."):
        if file.lower().endswith('.csv'):
            csv_files.append(file)

    # Check db directory
    db_dir = "db"
    if os.path.exists(db_dir):
        for file in os.listdir(db_dir):
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(db_dir, file))

    return csv_files


def display_tool_result(result: Dict[str, Any], tool_name: str):
    """Display tool execution results"""
    st.success(f"✅ {tool_name} completed successfully!")

    if "error" in result:
        st.error(f"Tool Error: {result['error']}")
        return

    # Display content if available
    if "content" in result and result["content"]:
        content = result["content"][0] if isinstance(result["content"], list) else result["content"]

        if content.get("type") == "text":
            text_content = content.get("text", "")

            # Try to format as code if it looks like JSON or has structure
            if text_content.strip().startswith(("{", "[")) or "Analysis for" in text_content:
                st.code(text_content, language="text")
            else:
                st.markdown(text_content)

            # Special handling for analysis results
            if "Analysis for" in text_content or "anomalies detected" in text_content.lower():
                st.info("💡 **Tip:** Results include comprehensive analysis with trends, seasonality, stationarity tests, and anomaly detection!")

    # Additional result info
    if len(result) > 1:
        with st.expander("📋 Detailed Results"):
            st.json(result)


if __name__ == "__main__":
    main()
