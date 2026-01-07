@echo off
echo 🚀 Starting Time Series Analysis MCP UI...
echo.
echo This will open a web browser with the interactive interface.
echo If it doesn't work, try: streamlit run mcp_ui.py
echo.
cd /d "%~dp0"
call venv\Scripts\activate
streamlit run mcp_ui.py
pause
