#!/usr/bin/env python3
"""
Demo script showing how to use the Time Series MCP Server
"""

import os
import sys

def main():
    print("Time Series MCP Server Demo")
    print("=" * 40)

    print("\n1. Testing MCP Server:")
    print("   python mcp_server.py")
    print("   (This starts the MCP server that can be connected to by MCP clients)")

    print("\n2. Testing MCP Client:")
    print("   python mcp_client.py --demo")
    print("   (Runs a demonstration of all tools)")

    print("\n3. Interactive Client:")
    print("   python mcp_client.py --interactive")
    print("   (Interactive mode for manual testing)")

    print("\n4. MCP Web UI (Recommended):")
    print("   python run_mcp_ui.py")
    print("   # Or: streamlit run mcp_ui.py")
    print("   # Or on Windows: run_ui.bat")
    print("   (Beautiful web interface - easiest to use!)")

    print("\n5. Claude Desktop Integration:")
    print("   Add to your claude_desktop_config.json:")
    print("""
   {
     "mcpServers": {
       "time-series-agent": {
         "command": "python",
         "args": ["mcp_server.py"],
         "cwd": "/path/to/Time-Series-Agent"
       }
     }
   }
   """)

    print("\n6. Available Tools:")
    tools = [
        "analyze_csv_file - Complete time series analysis",
        "detect_data_types - Data type detection",
        "analyze_time_series - Specific analysis types",
        "forecast_time_series - Generate forecasts",
        "detect_anomalies - Find anomalies",
        "query_documents - Ask questions about documents",
        "list_available_data - List CSV files"
    ]

    for tool in tools:
        print(f"   - {tool}")

    print("\n7. Sample Usage with Claude:")
    print('   "Analyze the Electric_Production.csv file for time series patterns"')
    print('   "What anomalies are in the commit_history.csv data?"')
    print('   "Forecast the next 12 months of electric production"')

    print("\n8. Sample Data Files:")
    data_files = ["Electric_Production.csv", "Weather_dataset.csv", "db/commit_history.csv"]
    for file in data_files:
        if os.path.exists(file):
            print(f"   [OK] {file}")
        else:
            print(f"   [MISSING] {file}")

    print("\n9. Run Tests:")
    print("   python test_mcp.py")
    print("   (Verifies all components are working)")

if __name__ == "__main__":
    main()