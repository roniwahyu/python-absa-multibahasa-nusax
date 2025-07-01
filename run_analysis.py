#!/usr/bin/env python3
"""
Simple script to run the Instagram reviews sentiment analysis.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from playstore_sentiment_analyzer import ReviewAnalyzer
from config import APP_CONFIG, OUTPUT_CONFIG
import argparse

def main():
    """Main function to run the analysis with command line arguments."""
    parser = argparse.ArgumentParser(description='Instagram Reviews Sentiment Analyzer')
    parser.add_argument('--app-id', default=APP_CONFIG['app_id'], 
                       help='Google Play Store app ID')
    parser.add_argument('--country', default=APP_CONFIG['country'],
                       help='Country code (default: id)')
    parser.add_argument('--lang', default=APP_CONFIG['lang'],
                       help='Language code (default: id)')
    parser.add_argument('--count', type=int, default=APP_CONFIG['review_count'],
                       help='Number of reviews to scrape (default: 2000)')
    parser.add_argument('--output-dir', default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Instagram Reviews Sentiment Analyzer")
    print("Using NusaX for Indonesian, Javanese, and Sundanese")
    print("=" * 60)
    print(f"App ID: {args.app_id}")
    print(f"Country: {args.country}")
    print(f"Language: {args.lang}")
    print(f"Review Count: {args.count}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ReviewAnalyzer()
    
    try:
        # Setup
        print("Setting up analyzer...")
        analyzer.setup(args.app_id, args.country, args.lang)
        
        # Scrape and analyze
        print("Starting analysis...")
        df = analyzer.scrape_and_analyze(args.count)
        
        if df.empty:
            print("No reviews were scraped. Exiting.")
            return 1
        
        # Change to output directory for saving files
        original_cwd = os.getcwd()
        os.chdir(args.output_dir)
        
        try:
            # Save results
            print("Saving results...")
            analyzer.save_results()
            
            # Generate report
            print("Generating report...")
            analyzer.generate_report()
            
        finally:
            # Change back to original directory
            os.chdir(original_cwd)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())