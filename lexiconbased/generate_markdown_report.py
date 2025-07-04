"""
Generate Markdown Report for Indonesian Sentiment Analysis Comparison
This script runs a focused comparison and generates a comprehensive markdown report.
"""

import sys
import os

# Add parent directory to path to import VADER Indonesia
sys.path.append('..')

from sentiment_analyzer_comparison import SentimentAnalyzerComparison

def generate_markdown_report():
    """Generate a comprehensive markdown report"""
    print("🚀 Generating Indonesian Sentiment Analysis Markdown Report")
    print("=" * 60)
    
    try:
        # Create comparison instance
        comparison = SentimentAnalyzerComparison()
        
        # Run comprehensive comparison
        print("Running comprehensive comparison...")
        results = comparison.run_comprehensive_comparison()
        
        # Generate markdown report
        print("\n📝 Generating Markdown report...")
        markdown_report = comparison.generate_detailed_report(format='markdown')
        
        # Also generate text report for comparison
        print("📄 Generating Text report...")
        text_report = comparison.generate_detailed_report(format='text')
        
        # Print summary to console
        comparison.print_comparison_summary()
        
        print("\n" + "="*60)
        print("✅ REPORTS GENERATED SUCCESSFULLY!")
        print("="*60)
        print("📁 Files created:")
        print("  - sentiment_analysis_comparison_report.md (Markdown format)")
        print("  - sentiment_analysis_comparison_report.txt (Text format)")
        print("\n📊 The markdown report includes:")
        print("  - Executive summary with key findings")
        print("  - Detailed performance comparison tables")
        print("  - Speed analysis and benchmarks")
        print("  - Method agreement analysis")
        print("  - Strengths and weaknesses breakdown")
        print("  - Actionable recommendations")
        print("  - Technical implementation details")
        print("  - Professional conclusion")
        
        return comparison, markdown_report, text_report
        
    except Exception as e:
        print(f"\n❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def preview_markdown_report():
    """Generate and preview a sample markdown report"""
    print("\n🔍 Generating Preview Report...")
    print("-" * 40)
    
    try:
        comparison = SentimentAnalyzerComparison()
        comparison.initialize_analyzers()
        
        # Create minimal test data for preview
        preview_texts = [
            "Film ini sangat bagus dan menghibur sekali",
            "Saya suka banget dengan ceritanya yang menarik",
            "Film ini buruk sekali dan sangat membosankan",
            "Sangat mengecewakan dan tidak worth it",
            "Film yang biasa saja, tidak terlalu istimewa",
            "Lumayan untuk ditonton, tidak bagus tidak jelek"
        ]
        preview_labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
        
        # Train Naive Bayes with minimal data
        print("Training with preview data...")
        comparison.nb_analyzer.train(preview_texts, preview_labels, test_size=0.3)
        
        # Create mock test results for preview
        nb_pred, nb_conf = comparison.predict_naive_bayes(preview_texts)
        vader_pred, vader_conf = comparison.predict_vader(preview_texts)
        
        # Create simplified test results
        comparison.test_results = {
            'training_data': {
                'naive_bayes': {
                    'accuracy': 0.85,
                    'precision': 0.83,
                    'recall': 0.87,
                    'f1_score': 0.85,
                    'classification_report': {
                        'positive': {'precision': 0.90, 'recall': 0.85, 'f1-score': 0.87},
                        'negative': {'precision': 0.80, 'recall': 0.90, 'f1-score': 0.85},
                        'neutral': {'precision': 0.85, 'recall': 0.80, 'f1-score': 0.82}
                    }
                },
                'vader': {
                    'accuracy': 0.78,
                    'precision': 0.76,
                    'recall': 0.80,
                    'f1_score': 0.78,
                    'classification_report': {
                        'positive': {'precision': 0.85, 'recall': 0.75, 'f1-score': 0.80},
                        'negative': {'precision': 0.70, 'recall': 0.85, 'f1-score': 0.77},
                        'neutral': {'precision': 0.75, 'recall': 0.70, 'f1-score': 0.72}
                    }
                },
                'nb_predictions': nb_pred,
                'vader_predictions': vader_pred,
                'nb_confidences': nb_conf,
                'vader_confidences': vader_conf,
                'true_labels': preview_labels,
                'texts': preview_texts
            },
            'challenging_data': {
                'naive_bayes': {'accuracy': 0.70, 'precision': 0.68, 'recall': 0.72, 'f1_score': 0.70},
                'vader': {'accuracy': 0.65, 'precision': 0.63, 'recall': 0.67, 'f1_score': 0.65},
                'nb_predictions': ['positive', 'negative'],
                'vader_predictions': ['neutral', 'negative'],
                'nb_confidences': [0.8, 0.9],
                'vader_confidences': [0.6, 0.7],
                'true_labels': ['positive', 'negative'],
                'texts': ['Test challenging case 1', 'Test challenging case 2']
            },
            'speed': {
                'naive_bayes_avg_time': 0.0234,
                'vader_avg_time': 0.0045,
                'naive_bayes_std': 0.0012,
                'vader_std': 0.0008
            },
            'training_time': 2.45
        }
        
        # Generate preview markdown report
        preview_report = comparison.generate_detailed_report(format='markdown', save_to_file=False)
        
        # Show first few lines of the report
        lines = preview_report.split('\n')
        print("\n📋 Preview of Markdown Report (first 20 lines):")
        print("-" * 50)
        for i, line in enumerate(lines[:20]):
            print(f"{i+1:2d}: {line}")
        
        if len(lines) > 20:
            print(f"... and {len(lines) - 20} more lines")
        
        print(f"\n📊 Full report would be {len(lines)} lines long")
        print("✅ Preview generation completed!")
        
        return preview_report
        
    except Exception as e:
        print(f"❌ Preview generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function with user options"""
    print("Indonesian Sentiment Analysis - Markdown Report Generator")
    print("=" * 60)
    
    print("\nOptions:")
    print("1. Generate full comparison report (recommended)")
    print("2. Generate preview report (quick test)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting full comparison...")
            comparison, markdown_report, text_report = generate_markdown_report()
            if comparison:
                print("\n🎉 Full comparison completed successfully!")
                print("Check the generated .md and .txt files for detailed results.")
            break
            
        elif choice == '2':
            print("\n🔍 Generating preview...")
            preview_report = preview_markdown_report()
            if preview_report:
                print("\n✨ Preview completed! This shows the structure of the full report.")
            break
            
        elif choice == '3':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
