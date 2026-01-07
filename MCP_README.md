# Time Series Agent MCP Server

This directory contains a Model Context Protocol (MCP) server implementation for the Time Series Analysis Agent. MCP allows AI assistants like Claude to connect to and use specialized tools.

## Overview

The MCP server exposes the time series analysis capabilities as tools that can be called by MCP-compatible clients:

- **Data Analysis**: CSV file analysis, data type detection
- **Time Series Analysis**: Trend analysis, seasonality detection, stationarity testing
- **Forecasting**: ARIMA and Exponential Smoothing forecasts
- **Anomaly Detection**: Prophet-based anomaly detection
- **Document Querying**: RAG-based question answering from uploaded documents

## Installation

1. **Install dependencies**:
   ```bash
   cd Time-Series-Agent
   pip install -r requirements.txt
   ```

2. **Set up environment variables** (optional):
   ```bash
   cp env_example.txt .env
   # Edit .env and add your GROQ_API_KEY if available
   ```

## MCP Server Usage

### Direct Server Execution

Run the MCP server directly:

```bash
python mcp_server.py
```

The server communicates via stdin/stdout using the MCP protocol.

### MCP Client Demo

Test the server with the included client:

```bash
# Run interactive mode
python mcp_client.py --interactive

# Run demonstration
python mcp_client.py --demo

# Specify custom server command
python mcp_client.py --server "python3 mcp_server.py"
```

### 🖥️ Time Series Analysis Web UI (Recommended!)

Beautiful web interface that uses the analysis tools directly:

```bash
# Easy launcher (recommended)
python run_mcp_ui.py

# Or run directly with streamlit
streamlit run mcp_ui.py

# Or on Windows
run_ui.bat

# IMPORTANT: Do NOT run with just 'python mcp_ui.py'
# Always use 'streamlit run' or the launcher script
```

The web UI provides:
- **Direct tool access** - No MCP server process needed
- **File upload support** - Drag & drop CSV files for analysis
- **Interactive tool selection** - Choose from 7 analysis tools
- **Beautiful results display** - Formatted analysis outputs
- **Sample data access** - Use included Electric_Production.csv, Weather_dataset.csv, etc.
- **Real-time analysis** - Immediate results without network overhead

### Claude Desktop Integration

To use with Claude Desktop:

1. **Edit your Claude Desktop configuration**:

   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

2. **Add the time-series-agent server**:
   ```json
   {
     "mcpServers": {
       "time-series-agent": {
         "command": "python",
         "args": ["mcp_server.py"],
         "cwd": "/full/path/to/Time-Series-Agent"
       }
     }
   }
   ```

3. **Restart Claude Desktop**

4. **Available Tools in Claude**:

   Once connected, you can ask Claude to use these tools:

   - `analyze_csv_file` - Comprehensive CSV analysis
   - `detect_data_types` - Data type detection
   - `analyze_time_series` - Time series analysis
   - `forecast_time_series` - Generate forecasts
   - `detect_anomalies` - Find anomalies
   - `query_documents` - Ask questions about documents
   - `list_available_data` - List CSV files

## Example Usage with Claude

Once integrated with Claude Desktop, you can have natural conversations like:

**You**: "Can you analyze the Electric_Production.csv file for time series patterns?"

**Claude**: (will use the analyze_csv_file tool and provide analysis)

**You**: "What anomalies are there in the commit_history.csv data?"

**Claude**: (will use the detect_anomalies tool)

**You**: "Forecast the next 12 months of electric production"

**Claude**: (will use the forecast_time_series tool)

## Tool Details

### analyze_csv_file
- **Purpose**: Complete time series analysis of a CSV file
- **Parameters**:
  - `csv_path`: Path to CSV file
  - `output_dir`: Output directory (optional)
- **Returns**: Comprehensive analysis summary with generated reports

### detect_data_types
- **Purpose**: Analyze data types and structure
- **Parameters**:
  - `csv_path`: Path to CSV file
- **Returns**: Data type information, column types, missing values

### analyze_time_series
- **Purpose**: Specific time series analysis
- **Parameters**:
  - `csv_path`: Path to CSV file
  - `column`: Column to analyze (optional)
  - `analysis_type`: Type of analysis (trend/seasonality/forecast/anomaly/comprehensive)
- **Returns**: Analysis results for specified type

### forecast_time_series
- **Purpose**: Generate time series forecasts
- **Parameters**:
  - `csv_path`: Path to CSV file
  - `column`: Column to forecast (optional)
  - `periods`: Number of periods to forecast (default: 30)
  - `method`: Forecasting method (arima/exponential)
- **Returns**: Forecast results and metrics

### detect_anomalies
- **Purpose**: Find anomalies using Prophet
- **Parameters**:
  - `csv_path`: Path to CSV file
  - `column`: Column to analyze (optional)
  - `interval_width`: Confidence interval (default: 0.99)
- **Returns**: Anomaly count and timestamps

### query_documents
- **Purpose**: Ask questions about uploaded documents
- **Parameters**:
  - `question`: Question to ask
- **Returns**: Answer based on document content

### list_available_data
- **Purpose**: List available CSV files
- **Parameters**: None
- **Returns**: List of CSV files with metadata

## Sample Data

The agent comes with sample data files:

- `Electric_Production.csv` - Monthly electric production data
- `Weather_dataset.csv` - Weather data
- `db/commit_history.csv` - Git commit history

## Architecture

```
Time Series Agent MCP Server
├── mcp_server.py      # Main MCP server implementation
├── mcp_client.py       # Test client and demo
├── ts_agent.py         # Core agent logic
├── time_series_tools.py # Analysis algorithms
├── data_detector.py    # Data type detection
├── reporting.py        # Report generation
└── mcp-config.json     # Claude Desktop config template
```

## Troubleshooting

### Server Won't Start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path and working directory
- Verify CSV files exist if testing with specific files

### Claude Desktop Connection Issues
- Verify the `cwd` path in config points to the correct directory
- Check that `python` is available in PATH
- Look at Claude Desktop logs for connection errors

### Analysis Errors
- Ensure CSV files have proper headers
- Check for time/date columns in time series data
- Verify numeric data exists for analysis

## Development

To extend the MCP server:

1. **Add new tools**: Implement methods in `TimeSeriesMCPServer` class
2. **Update tool list**: Add to `list_tools()` method
3. **Handle tool calls**: Add cases in `call_tool()` method

The server uses the official MCP Python SDK for protocol handling.

## License

This MCP implementation is part of the Time Series Analysis Agent project.
