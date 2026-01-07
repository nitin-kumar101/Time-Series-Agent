#!/usr/bin/env python3
"""
MCP Server for Time Series Analysis Agent
Exposes time series analysis capabilities through the Model Context Protocol
"""

import asyncio
import sys
import os
from typing import Any, Dict, List
from mcp.server import FastMCP

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import our time series analysis modules
from ts_agent import HybridChatAgent
from data_detector import DataAnalyzer
from time_series_tools import TimeSeriesAnalyzer, TimeSeriesVisualizer
from reporting import ReportGenerator, DashboardGenerator, ExportManager
import pandas as pd


# Create FastMCP app
app = FastMCP("time-series-agent")

# Initialize our components
agent = HybridChatAgent()
data_analyzer = DataAnalyzer()
ts_analyzer = TimeSeriesAnalyzer()
visualizer = TimeSeriesVisualizer()
report_generator = ReportGenerator()
export_manager = ExportManager()


@app.tool()
async def analyze_csv_file(csv_path: str, output_dir: str = "output/ts") -> str:
    """Analyze a CSV file for time series patterns, trends, seasonality, and anomalies"""
    if not os.path.exists(csv_path):
        return f"CSV file not found: {csv_path}"

    try:
        # Use the agent's analysis method
        analysis = agent._analyze_csv_path(csv_path)
        summary = analysis['summary']

        # Add output file information
        outputs = analysis.get('outputs', {})
        if outputs:
            summary += f"\n\nGenerated files:\n"
            for key, path in outputs.items():
                if isinstance(path, str) and os.path.exists(path):
                    summary += f"- {key}: {path}\n"

        return summary

    except Exception as e:
        return f"Analysis failed: {str(e)}"


@app.tool()
async def detect_data_types(csv_path: str) -> str:
    """Detect and analyze data types in a CSV file"""
    if not os.path.exists(csv_path):
        return f"CSV file not found: {csv_path}"

    try:
        data_info = data_analyzer.analyze_csv(csv_path)

        if "error" in data_info:
            return f"Data analysis error: {data_info['error']}"

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

        return result

    except Exception as e:
        return f"Data type detection failed: {str(e)}"


@app.tool()
async def analyze_time_series(csv_path: str, column: str = None, analysis_type: str = "comprehensive") -> str:
    """Perform comprehensive time series analysis on a prepared dataset"""
    if not os.path.exists(csv_path):
        return f"CSV file not found: {csv_path}"

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
            return "No valid time series data found"

        if not numeric_cols:
            return "No numeric columns found for analysis"

        # Select column
        if not column:
            column = numeric_cols[0]

        if column not in prepared_data.columns:
            return f"Column '{column}' not found"

        series = prepared_data[column].dropna()

        # Perform analysis based on type
        if analysis_type == "trend":
            result = ts_analyzer.analyze_trend(series)
            summary = "Trend Analysis:\n"
            summary += f"Direction: {result['trend_direction']}\n"
            summary += f"Strength: {result['trend_strength']:.3f}\n"
            summary += f"Slope: {result['trend_slope']:.6f}\n"

        elif analysis_type == "seasonality":
            result = ts_analyzer.detect_seasonality(series)
            summary = "Seasonality Analysis:\n"
            summary += f"Detected: {result['seasonality_detected']}\n"
            if result['seasonality_detected']:
                summary += f"Period: {result.get('period', 'N/A')}\n"
                summary += f"Strength: {result['seasonal_strength']:.3f}\n"

        elif analysis_type == "forecast":
            arima_result = ts_analyzer.forecast_arima(series, periods=30)
            summary = "Forecast Analysis:\n"
            if "error" not in arima_result:
                summary += "ARIMA forecast available\n"
            else:
                summary += f"ARIMA error: {arima_result['error']}\n"

        elif analysis_type == "anomaly":
            result = ts_analyzer.detect_anomalies_prophet(series)
            if isinstance(result, dict) and "anomalies" in result:
                count = len(result['anomalies'])
                summary = f"Anomaly Detection: {count} anomalies detected"
            else:
                summary = "Anomaly detection failed"

        else:  # comprehensive
            results = ts_analyzer.comprehensive_analysis(series)
            summary = agent.get_analysis_summary({column: results})

        return summary

    except Exception as e:
        return f"Time series analysis failed: {str(e)}"


@app.tool()
async def forecast_time_series(csv_path: str, column: str = None, periods: int = 30, method: str = "arima") -> str:
    """Generate forecasts using ARIMA or Exponential Smoothing"""
    if not os.path.exists(csv_path):
        return f"CSV file not found: {csv_path}"

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
            return f"Column '{column}' not found"

        series = prepared_data[column].dropna()

        # Generate forecast
        if method == "arima":
            result = ts_analyzer.forecast_arima(series, periods=periods)
        else:  # exponential
            result = ts_analyzer.forecast_exponential_smoothing(series, periods=periods)

        if "error" in result:
            return f"Forecast failed: {result['error']}"

        # Format results
        forecast = result['forecast']
        summary = f"{method.upper()} Forecast Results:\n"
        summary += f"Forecast periods: {periods}\n"
        summary += f"Forecast range: {forecast.index[0]} to {forecast.index[-1]}\n"
        summary += f"Mean forecast value: {forecast.mean():.3f}\n"

        if method == "arima":
            summary += f"AIC: {result.get('aic', 'N/A')}\n"
            summary += f"BIC: {result.get('bic', 'N/A')}\n"

        return summary

    except Exception as e:
        return f"Forecast generation failed: {str(e)}"


@app.tool()
async def detect_anomalies(csv_path: str, column: str = None, interval_width: float = 0.99) -> str:
    """Detect anomalies in time series data using Prophet"""
    if not os.path.exists(csv_path):
        return f"CSV file not found: {csv_path}"

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
            return f"Column '{column}' not found"

        series = prepared_data[column].dropna()

        # Detect anomalies
        result = ts_analyzer.detect_anomalies_prophet(series, interval_width)

        if not isinstance(result, dict) or "anomalies" not in result:
            return "Anomaly detection failed"

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

        return summary

    except Exception as e:
        return f"Anomaly detection failed: {str(e)}"


@app.tool()
async def query_documents(question: str) -> str:
    """Query documents in the knowledge base using RAG"""
    try:
        response = agent.chat(question)
        return response
    except Exception as e:
        return f"Document query failed: {str(e)}"


@app.tool()
async def list_available_data() -> str:
    """List available CSV files and their basic information"""
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
        return "No CSV files found"

    result = "Available CSV Files:\n\n"
    for file_info in data_files:
        result += f"📄 {file_info['path']}\n"
        result += f"   Rows: {file_info['rows']}, Columns: {len(file_info['columns'])}\n"
        result += f"   Size: {file_info['size_mb']:.2f} MB\n"
        result += f"   Columns: {', '.join(file_info['columns'][:5])}\n"
        if len(file_info['columns']) > 5:
            result += f"   ... and {len(file_info['columns']) - 5} more\n"
        result += "\n"

    return result


# Note: Using FastMCP instead of legacy MCP implementation


if __name__ == "__main__":
    # Run the FastMCP server
    app.run()
