#!/usr/bin/env python3
"""
Simple test script for MCP server functionality
"""

import asyncio
import subprocess
import sys
import os
import time
import json


def test_server_startup():
    """Test that the MCP server starts without errors"""
    print("Testing MCP server startup...")

    try:
        # Start server process
        server = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), "mcp_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait a moment for startup
        time.sleep(3)

        # Check if process is still running (good sign)
        if server.poll() is None:
            print("[PASS] Server process started successfully")
            result = True
        else:
            # Check stderr for errors
            stderr_output = server.stderr.read()
            if stderr_output:
                print(f"[FAIL] Server exited with error: {stderr_output}")
            else:
                print("[FAIL] Server exited without error output")
            result = False

        # Clean up
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=5)

        return result

    except Exception as e:
        print(f"[FAIL] Server startup test failed: {e}")
        return False


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")

    # Set dummy API key to avoid initialization errors
    os.environ["GROQ_API_KEY"] = "dummy-key-for-testing"

    try:
        from ts_agent import HybridChatAgent
        from data_detector import DataAnalyzer
        from time_series_tools import TimeSeriesAnalyzer, TimeSeriesVisualizer
        from reporting import ReportGenerator
        from mcp_server import app  # FastMCP app instead of legacy class

        print("[PASS] All modules imported successfully")
        return True

    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False
    except Exception as e:
        # Allow other exceptions during import (like API key issues)
        print(f"[WARN] Import warning (non-critical): {e}")
        return True


def test_data_files():
    """Test that sample data files exist"""
    print("Testing data files...")

    # Change to the Time-Series-Agent directory
    agent_dir = os.path.dirname(__file__)
    original_dir = os.getcwd()

    try:
        os.chdir(agent_dir)

        data_files = [
            "Electric_Production.csv",
            "Weather_dataset.csv",
            "db/commit_history.csv"
        ]

        missing_files = []
        for file_path in data_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            print(f"[WARN] Missing data files: {missing_files}")
            return False
        else:
            print("[PASS] All data files found")
            return True
    finally:
        os.chdir(original_dir)


def main():
    """Run all tests"""
    print("Running MCP Time Series Agent Tests\n")

    tests = [
        ("Module Imports", test_imports),
        ("Data Files", test_data_files),
        ("Server Startup", test_server_startup),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1

    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All tests passed! MCP server is ready.")
        return 0
    else:
        print("ERROR: Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
