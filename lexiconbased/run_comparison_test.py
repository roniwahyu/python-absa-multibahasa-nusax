"""
Quick test script for Indonesian Sentiment Analysis Comparison
"""

import sys
import os

# Add parent directory to path to import VADER Indonesia
sys.path.append('..')

from sentiment_analyzer_comparison import SentimentAnalyzerComparison

def quick_test():
    """Run a quick test of the comparison framework"""
    print("🚀 Quick Test: Indonesian Sentiment Analysis Comparison")
    print("=" * 60)
    
    try:
        # Create comparison instance
        comparison = SentimentAnalyzerComparison()
        
        # Test individual components first
        print("\n1. Testing analyzer initialization...")
        comparison.initialize_analyzers()
        
        print("\n2. Testing dataset creation...")
        train_texts, train_labels = comparison.create_comprehensive_test_dataset()
        challenge_texts, challenge_labels = comparison.create_challenging_test_cases()
        
        print(f"✓ Training dataset: {len(train_texts)} samples")
        print(f"✓ Challenge dataset: {len(challenge_texts)} samples")
        
        print("\n3. Testing individual predictions...")
        
        # Test a few examples
        test_examples = [
            "Film ini sangat bagus dan menghibur",
            "Saya tidak suka dengan ceritanya",
            "Film yang biasa saja"
        ]
        
        print("\nNaive Bayes predictions (before training):")
        for text in test_examples[:2]:  # Test only 2 examples before training
            try:
                # This should fail since model isn't trained yet
                result = comparison.nb_analyzer.predict_sentiment(text)
                print(f"  {text}: {result['sentiment']}")
            except Exception as e:
                print(f"  Expected error (not trained): {str(e)[:50]}...")
        
        print("\nVADER Indonesia predictions:")
        for text in test_examples:
            vader_pred, vader_conf = comparison.predict_vader([text])
            print(f"  {text}: {vader_pred[0]} (conf: {vader_conf[0]:.3f})")
        
        print("\n4. Testing training process...")
        # Use smaller dataset for quick test
        small_train_texts = train_texts[:20]  # Use only 20 samples for quick test
        small_train_labels = train_labels[:20]
        
        nb_results, training_time = comparison.train_naive_bayes(small_train_texts, small_train_labels)
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        print("\n5. Testing trained Naive Bayes predictions...")
        for text in test_examples:
            result = comparison.nb_analyzer.predict_sentiment(text)
            print(f"  {text}: {result['sentiment']} (conf: {result['confidence']:.3f})")
        
        print("\n6. Testing speed comparison...")
        speed_results = comparison.run_speed_test(test_examples, num_iterations=2)
        print(f"✓ Speed test completed")
        print(f"  NB avg time: {speed_results['naive_bayes_avg_time']:.4f}s")
        print(f"  VADER avg time: {speed_results['vader_avg_time']:.4f}s")
        
        print("\n✅ Quick test completed successfully!")
        print("You can now run the full comparison with: python sentiment_analyzer_comparison.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def run_mini_comparison():
    """Run a mini version of the full comparison"""
    print("\n🔬 Mini Comparison Test")
    print("=" * 40)
    
    try:
        comparison = SentimentAnalyzerComparison()
        comparison.initialize_analyzers()
        
        # Create small test dataset
        mini_texts = [
            "Film ini sangat bagus dan menghibur sekali",
            "Saya suka banget dengan ceritanya",
            "Film ini buruk sekali dan membosankan",
            "Sangat mengecewakan dan tidak menarik",
            "Film yang biasa saja, tidak istimewa",
            "Lumayan untuk ditonton"
        ]
        mini_labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
        
        # Train Naive Bayes
        print("Training Naive Bayes...")
        comparison.nb_analyzer.train(mini_texts, mini_labels, test_size=0.3)
        
        # Test both methods
        print("\nTesting both methods:")
        print("-" * 40)
        
        test_text = "Film yang sangat bagus dan tidak mengecewakan"
        
        # Naive Bayes prediction
        nb_result = comparison.nb_analyzer.predict_sentiment(test_text)
        
        # VADER prediction
        vader_pred, vader_conf = comparison.predict_vader([test_text])
        
        print(f"Test text: {test_text}")
        print(f"Naive Bayes: {nb_result['sentiment']} (confidence: {nb_result['confidence']:.3f})")
        print(f"VADER Indonesia: {vader_pred[0]} (confidence: {vader_conf[0]:.3f})")
        
        # Check agreement
        agreement = nb_result['sentiment'] == vader_pred[0]
        print(f"Methods agree: {'Yes' if agreement else 'No'}")
        
        print("\n✅ Mini comparison completed!")
        
    except Exception as e:
        print(f"\n❌ Mini comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Indonesian Sentiment Analysis Comparison - Test Suite")
    print("=" * 60)
    
    # Run quick test first
    quick_test()
    
    # Ask user if they want to run mini comparison
    print("\n" + "="*60)
    response = input("Run mini comparison test? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_mini_comparison()
    
    print("\n" + "="*60)
    print("Test suite completed!")
    print("To run the full comparison, execute: python sentiment_analyzer_comparison.py")
