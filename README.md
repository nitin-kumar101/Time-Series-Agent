# Time Series Analysis Agent


A comprehensive Python application that uses LangGraph and AI agents to automatically analyze CSV data for time series patterns. The agent intelligently detects data types, performs trend analysis, seasonality detection, and forecasting. Now includes a **Streamlit web interface** for easy interaction!

## Features

- **🤖 AI-Powered Agent**: Built with LangGraph for intelligent workflow orchestration
- **🌐 Web Interface**: Beautiful Streamlit UI for chatting, file uploads, and visualization
- **🔍 Automatic Data Detection**: Automatically identifies time columns, numeric data, and categorical variables
- **📈 Comprehensive Analysis**: 
  - Trend analysis with moving averages and linear regression
  - Seasonality detection with seasonal decomposition
  - Stationarity testing (ADF and KPSS tests)
  - Forecasting with ARIMA and Exponential Smoothing models
  - **🔍 Anomaly Detection**: Prophet-based anomaly detection with visualizations
- **📊 Interactive Visualizations**: Plotly-based charts and dashboards
- **📋 Detailed Reports**: HTML and JSON reports with analysis summaries
- **🎯 Multiple Interfaces**: Web UI, command-line, and interactive modes
- **📁 File Management**: Upload CSV/PDF files for RAG and time series analysis

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Groq API key (optional, for enhanced AI features):
```bash
export GROQ_API_KEY='your-api-key-here'
```

## Usage

### 🌐 Web Interface (Recommended)
```bash
# Start the Streamlit web app
python run_app.py

# Or directly with streamlit
streamlit run streamlit_app.py
```

The web interface provides:
- **💬 Chat Interface**: Ask questions about your data or request time series analysis
- **📁 File Upload**: Upload CSV/PDF files for analysis and RAG
- **📊 Live Visualizations**: View interactive plots and anomaly detection charts
- **⚡ Quick Actions**: One-click analysis of common files

### Interactive Mode
```bash
python main.py
```

### Command Line Mode
```bash
# Analyze a specific CSV file
python main.py data.csv

# Specify output directory
python main.py data.csv --output results

# Force interactive mode
python main.py --interactive
```

### Programmatic Usage
```python
from ts_agent import TimeSeriesAnalysisAgent

# Create agent
agent = TimeSeriesAnalysisAgent()

# Analyze CSV
results = agent.analyze_csv("your_data.csv")

if results["success"]:
    print("Analysis completed!")
    print(agent.get_analysis_summary(results["analysis_results"]))
```

## Project Structure

```
time-series-tool/
├── main.py                 # Main application entry point
├── ts_agent.py            # LangGraph agent implementation
├── streamlit_app.py       # Streamlit web interface
├── run_app.py             # Streamlit launcher script
├── data_detector.py       # Automatic data type detection
├── time_series_tools.py  # Time series analysis algorithms
├── reporting.py           # Report generation and visualization
├── requirements.txt       # Python dependencies
├── commit_history.csv     # Sample data file
└── db/                    # Directory for uploaded files (PDF, CSV)
```

## How It Works

1. **Data Loading**: Automatically loads and validates CSV files
2. **Data Detection**: Intelligently identifies:
   - Time/date columns using pattern matching
   - Numeric columns for analysis
   - Categorical variables
3. **Data Preparation**: Converts time columns to datetime and sets as index
4. **Analysis Pipeline**:
   - Trend analysis with multiple methods
   - Seasonality detection and decomposition
   - Stationarity testing
   - Forecasting with multiple models
5. **Visualization**: Generates interactive plots and dashboards
6. **Reporting**: Creates comprehensive HTML and JSON reports

## Supported Data Formats

The agent can handle various CSV formats with:
- Different date/time formats (ISO, US, European, etc.)
- Mixed data types
- Missing values
- Irregular time intervals

## Output Files

For each analyzed column, the agent generates:
- `analysis_report.html` - Interactive HTML report
- `analysis_report.json` - Detailed JSON results
- `plot_*.html` - Individual visualization files
- `data_summary.csv` - Statistical summary
- `analysis_summary.txt` - Human-readable summary

## Example Analysis

The included `commit_history.csv` demonstrates the agent's capabilities:
- Detects the `date` column as time series data
- Analyzes commit patterns over time
- Identifies trends and seasonality in development activity
- Generates forecasts for future activity

## Dependencies

- **LangGraph**: Agent workflow orchestration
- **LangChain**: LLM integration and tools
- **Streamlit**: Web interface framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Statsmodels**: Statistical models and tests
- **Scikit-learn**: Machine learning utilities
- **Prophet**: Advanced forecasting and anomaly detection
- **ChromaDB**: Vector database for RAG
- **Sentence Transformers**: Embeddings for document search

## Troubleshooting

### Common Issues

1. **"No time column detected"**: Ensure your CSV has a column with dates/times
2. **"No numeric columns found"**: Make sure you have numeric data to analyze
3. **"Analysis failed"**: Check that your CSV file is properly formatted

### Getting Help

The interactive mode provides helpful error messages and suggestions. For command-line usage, check the generated error reports in the output directory.

## Contributing

This is a demonstration project showcasing LangGraph agent capabilities for time series analysis. Feel free to extend and modify for your specific needs.

## License

This project is for educational and demonstration purposes.
