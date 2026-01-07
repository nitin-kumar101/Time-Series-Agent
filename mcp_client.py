#!/usr/bin/env python3
"""
MCP Client for Time Series Analysis Agent
Demonstrates how to connect to and use the MCP server
"""

import asyncio
import json
import subprocess
import sys
import os
from typing import Dict, Any, List, Optional
import argparse


class TimeSeriesMCPClient:
    """Client for interacting with the Time Series MCP server"""

    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.server_process: Optional[subprocess.Popen] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

    async def start_server(self):
        """Start the MCP server process"""
        try:
            # Start the server
            self.server_process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(__file__)
            )

            # Create asyncio streams
            self.reader = asyncio.StreamReader()
            reader_protocol = asyncio.StreamReaderProtocol(self.reader)
            transport, _ = await asyncio.get_event_loop().connect_read_pipe(
                lambda: reader_protocol, self.server_process.stdout
            )

            # Create writer
            self.writer = asyncio.StreamWriter(
                transport, None, self.reader, asyncio.get_event_loop()
            )

            # Wait a moment for server to initialize
            await asyncio.sleep(1)

            print("[SUCCESS] MCP server started successfully")

        except Exception as e:
            print(f"[ERROR] Failed to start MCP server: {e}")
            raise

    async def stop_server(self):
        """Stop the MCP server process"""
        if self.server_process:
            self.server_process.terminate()
            try:
                # On Windows, wait() is not async, so we need to handle it differently
                import concurrent.futures
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.server_process.wait),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.server_process.kill()
            print("[SUCCESS] MCP server stopped")

    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the server"""
        if not self.writer:
            raise RuntimeError("Server not started")

        # Send request
        request_json = json.dumps(request) + "\n"
        self.writer.write(request_json.encode())
        await self.writer.drain()

        # Read response
        try:
            response_line = await self.reader.readline()
            if not response_line:
                raise RuntimeError("Server closed connection")

            response = json.loads(response_line.decode().strip())
            return response
        except Exception as e:
            print(f"Error reading response: {e}")
            raise

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
                    "name": "time-series-mcp-client",
                    "version": "1.0.0"
                }
            }
        }

        try:
            response = await self.send_request(request)
            if "error" in response:
                print(f"[ERROR] Initialization failed: {response['error']}")
                return False
            else:
                print("[SUCCESS] MCP connection initialized")
                return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize: {e}")
            return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        response = await self.send_request(request)
        if "error" in response:
            print(f"[ERROR] Failed to list tools: {response['error']}")
            return []

        return response.get("result", {}).get("tools", [])

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool"""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        response = await self.send_request(request)
        if "error" in response:
            print(f"[ERROR] Tool call failed: {response['error']}")
            return {}

        return response.get("result", {})

    async def demo_analysis(self):
        """Run a demonstration of the time series analysis tools"""
        print("\n🚀 Starting Time Series Analysis Demo\n")

        # List available tools
        print("📋 Available Tools:")
        tools = await self.list_tools()
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")

        print("\n" + "="*50)

        # Demo 1: List available data
        print("\n1️⃣ Listing available data files...")
        result = await self.call_tool("list_available_data", {})
        if result and "content" in result:
            print(result["content"][0]["text"])

        # Demo 2: Analyze a CSV file
        csv_file = "Electric_Production.csv"
        if os.path.exists(csv_file):
            print(f"\n2️⃣ Analyzing CSV file: {csv_file}")
            result = await self.call_tool("analyze_csv_file", {
                "csv_path": csv_file,
                "output_dir": "output/demo"
            })
            if result and "content" in result:
                print(result["content"][0]["text"])

        # Demo 3: Detect data types
        if os.path.exists(csv_file):
            print(f"\n3️⃣ Detecting data types in: {csv_file}")
            result = await self.call_tool("detect_data_types", {
                "csv_path": csv_file
            })
            if result and "content" in result:
                print(result["content"][0]["text"])

        # Demo 4: Time series analysis
        if os.path.exists(csv_file):
            print(f"\n4️⃣ Performing comprehensive time series analysis on: {csv_file}")
            result = await self.call_tool("analyze_time_series", {
                "csv_path": csv_file,
                "analysis_type": "comprehensive"
            })
            if result and "content" in result:
                print(result["content"][0]["text"])

        # Demo 5: Forecasting
        if os.path.exists(csv_file):
            print(f"\n5️⃣ Generating forecast for: {csv_file}")
            result = await self.call_tool("forecast_time_series", {
                "csv_path": csv_file,
                "periods": 12,
                "method": "arima"
            })
            if result and "content" in result:
                print(result["content"][0]["text"])

        # Demo 6: Anomaly detection
        if os.path.exists(csv_file):
            print(f"\n6️⃣ Detecting anomalies in: {csv_file}")
            result = await self.call_tool("detect_anomalies", {
                "csv_path": csv_file,
                "interval_width": 0.95
            })
            if result and "content" in result:
                print(result["content"][0]["text"])

        # Demo 7: Document querying (if documents exist)
        print("\n7. Testing document querying...")
        result = await self.call_tool("query_documents", {
            "question": "What kind of data analysis can I perform?"
        })
        if result and "content" in result:
            print(result["content"][0]["text"])

        print("\n🎉 Demo completed!")

    async def interactive_mode(self):
        """Run in interactive mode"""
        print("🤖 Time Series MCP Client - Interactive Mode")
        print("Type 'help' for available commands, 'quit' to exit\n")

        # Show available tools
        tools = await self.list_tools()
        tool_names = [tool["name"] for tool in tools]

        while True:
            try:
                command = input("mcp> ").strip()

                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help          - Show this help")
                    print("  tools         - List available tools")
                    print("  demo          - Run demonstration")
                    print("  analyze <csv> - Analyze a CSV file")
                    print("  forecast <csv>- Generate forecast")
                    print("  anomalies <csv>- Detect anomalies")
                    print("  query <text>  - Query documents")
                    print("  quit          - Exit")
                    print("\nAvailable tools:")
                    for tool in tools:
                        print(f"  {tool['name']} - {tool['description'][:60]}...")

                elif command.lower() == 'tools':
                    print("\nAvailable Tools:")
                    for tool in tools:
                        print(f"  • {tool['name']}")
                        print(f"    {tool['description']}")

                elif command.lower() == 'demo':
                    await self.demo_analysis()

                elif command.startswith('analyze '):
                    csv_file = command[8:].strip()
                    if os.path.exists(csv_file):
                        result = await self.call_tool("analyze_csv_file", {
                            "csv_path": csv_file
                        })
                        if result and "content" in result:
                            print(result["content"][0]["text"])
                    else:
                        print(f"File not found: {csv_file}")

                elif command.startswith('forecast '):
                    csv_file = command[9:].strip()
                    if os.path.exists(csv_file):
                        result = await self.call_tool("forecast_time_series", {
                            "csv_path": csv_file,
                            "periods": 30
                        })
                        if result and "content" in result:
                            print(result["content"][0]["text"])
                    else:
                        print(f"File not found: {csv_file}")

                elif command.startswith('anomalies '):
                    csv_file = command[10:].strip()
                    if os.path.exists(csv_file):
                        result = await self.call_tool("detect_anomalies", {
                            "csv_path": csv_file
                        })
                        if result and "content" in result:
                            print(result["content"][0]["text"])
                    else:
                        print(f"File not found: {csv_file}")

                elif command.startswith('query '):
                    question = command[6:].strip()
                    result = await self.call_tool("query_documents", {
                        "question": question
                    })
                    if result and "content" in result:
                        print(result["content"][0]["text"])

                else:
                    print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\n👋 Goodbye!")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Time Series MCP Client")
    parser.add_argument("--server", default="python mcp_server.py",
                       help="Command to start the MCP server")
    parser.add_argument("--demo", action="store_true",
                       help="Run demonstration mode")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")

    args = parser.parse_args()

    # Parse server command
    server_command = args.server.split()

    # Create client
    client = TimeSeriesMCPClient(server_command)

    try:
        # Start server
        await client.start_server()

        # Initialize connection
        if not await client.initialize():
            return

        # Run mode
        if args.demo:
            await client.demo_analysis()
        elif args.interactive:
            await client.interactive_mode()
        else:
            # Default: run demo
            await client.demo_analysis()

    except Exception as e:
        print(f"[ERROR] Error: {e}")
    finally:
        await client.stop_server()


if __name__ == "__main__":
    asyncio.run(main())
