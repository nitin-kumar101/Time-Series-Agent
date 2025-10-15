"""
Time Series Analysis Agent - Main Application
A comprehensive tool for automatic time series analysis using LangGraph and AI agents
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Import our custom modules
from ts_agent import TimeSeriesAnalysisAgent, analyze_time_series_csv, get_analysis_summary
from data_detector import DataAnalyzer
from time_series_tools import TimeSeriesAnalyzer, TimeSeriesVisualizer
from reporting import ReportGenerator, DashboardGenerator, ExportManager


class TimeSeriesAnalysisApp:
    """Main application class for time series analysis"""
    
    def __init__(self):
        self.agent = TimeSeriesAnalysisAgent()
        self.data_analyzer = DataAnalyzer()
        
    def run_interactive_mode(self):
        """Run the application in interactive mode"""
        print("=" * 60)
        print("🕒 Time Series Analysis Agent")
        print("=" * 60)
        print("Welcome! This agent will automatically analyze your CSV data for time series patterns.")
        print("Features:")
        print("• Automatic data type detection")
        print("• Trend analysis and seasonality detection")
        print("• Forecasting with ARIMA and Exponential Smoothing")
        print("• Interactive visualizations")
        print("• Comprehensive reports")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("1. Analyze CSV file")
            print("2. Quick data preview")
            print("3. View analysis summary")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                self._analyze_csv_interactive()
            elif choice == "2":
                self._preview_data_interactive()
            elif choice == "3":
                self._view_summary_interactive()
            elif choice == "4":
                print("Goodbye! 👋")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _analyze_csv_interactive(self):
        """Interactive CSV analysis"""
        csv_path = input("Enter the path to your CSV file: ").strip()
        
        if not csv_path:
            print("No file path provided.")
            return
            
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            return
            
        print(f"\n🔍 Analyzing {csv_path}...")
        print("This may take a few moments...")
        
        try:
            results = self.agent.analyze_csv(csv_path)
            
            if results["success"]:
                print("✅ Analysis completed successfully!")
                print(f"📊 Task: {results['current_task']}")
                
                # Show summary
                summary = self.agent.get_analysis_summary(results["analysis_results"])
                print("\n📈 Analysis Summary:")
                print(summary)
                
                # Show output files
                print("\n📁 Output files generated:")
                for col, files in results["output_files"].items():
                    print(f"\n  📊 {col}:")
                    for file_type, file_path in files.items():
                        print(f"    • {file_type}: {file_path}")
                
                print(f"\n💡 Tip: Open the HTML reports in your browser to view interactive visualizations!")
                
            else:
                print(f"❌ Analysis failed: {results['error']}")
                if "error_report" in results:
                    print("\n🔧 Suggestions:")
                    for suggestion in results["error_report"].get("suggestions", []):
                        print(f"  • {suggestion}")
                        
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
    
    def _preview_data_interactive(self):
        """Interactive data preview"""
        csv_path = input("Enter the path to your CSV file: ").strip()
        
        if not csv_path:
            print("No file path provided.")
            return
            
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            return
            
        try:
            print(f"\n🔍 Previewing {csv_path}...")
            
            # Quick data analysis
            data_info = self.data_analyzer.analyze_csv(csv_path)
            
            if "error" in data_info:
                print(f"❌ Error: {data_info['error']}")
                return
                
            print("✅ Data preview completed!")
            
            # Show basic info
            print(f"\n📊 Data Overview:")
            print(f"  • Total records: {data_info['data_info']['total_records']}")
            print(f"  • Time column: {data_info.get('time_column', 'Not detected')}")
            print(f"  • Numeric columns: {data_info['data_info']['numeric_columns']}")
            print(f"  • Categorical columns: {data_info['data_info']['categorical_columns']}")
            
            if data_info.get('time_column'):
                time_range = data_info['data_info']['time_range']
                print(f"  • Time range: {time_range['start']} to {time_range['end']}")
                print(f"  • Time span: {data_info['data_info']['time_span_days']} days")
            
            # Show first few rows
            df = data_info['original_data']
            print(f"\n📋 First 5 rows:")
            print(df.head().to_string())
            
        except Exception as e:
            print(f"❌ Error previewing data: {str(e)}")
    
    def _view_summary_interactive(self):
        """Interactive summary viewing"""
        csv_path = input("Enter the path to your CSV file: ").strip()
        
        if not csv_path:
            print("No file path provided.")
            return
            
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            return
            
        try:
            print(f"\n🔍 Getting analysis summary for {csv_path}...")
            
            summary = get_analysis_summary(csv_path)
            print("\n📈 Analysis Summary:")
            print(summary)
            
        except Exception as e:
            print(f"❌ Error getting summary: {str(e)}")
    
    def run_command_line(self, csv_path: str, output_dir: str = "output"):
        """Run analysis from command line"""
        print(f"🔍 Analyzing {csv_path}...")
        
        try:
            results = self.agent.analyze_csv(csv_path)
            
            if results["success"]:
                print("✅ Analysis completed successfully!")
                
                # Generate summary
                summary = self.agent.get_analysis_summary(results["analysis_results"])
                print("\n📈 Analysis Summary:")
                print(summary)
                
                # Save summary to file
                summary_file = os.path.join(output_dir, "analysis_summary.txt")
                os.makedirs(output_dir, exist_ok=True)
                with open(summary_file, 'w') as f:
                    f.write(summary)
                
                print(f"\n📁 Summary saved to: {summary_file}")
                print(f"📁 Other output files in: {output_dir}")
                
                return True
            else:
                print(f"❌ Analysis failed: {results['error']}")
                return False
                
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Time Series Analysis Agent - Automatic CSV time series analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run in interactive mode
  python main.py data.csv                 # Analyze specific CSV file
  python main.py data.csv --output results # Analyze and save to specific directory
        """
    )
    
    parser.add_argument(
        "csv_file", 
        nargs="?", 
        help="Path to CSV file to analyze"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for results (default: output)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Force interactive mode"
    )
    
    args = parser.parse_args()
    
    # Check for Groq API key
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY not found in environment variables.")
        print("   Some features may not work properly.")
        print("   Please set your Groq API key:")
        print("   export GROQ_API_KEY='your-api-key-here'")
        print()
    
    app = TimeSeriesAnalysisApp()
    
    if args.interactive or not args.csv_file:
        # Interactive mode
        app.run_interactive_mode()
    else:
        # Command line mode
        success = app.run_command_line(args.csv_file, args.output)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()