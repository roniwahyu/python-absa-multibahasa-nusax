"""
Example usage of Indonesian Naive Bayes Sentiment Analyzer
"""

from indonesian_naive_bayes_analyzer import IndonesianNaiveBayesAnalyzer, create_sample_dataset

def main():
    print("Indonesian Naive Bayes Sentiment Analyzer - Example Usage")
    print("=" * 60)
    
    # Initialize analyzer
    print("\n1. Initializing analyzer...")
    analyzer = IndonesianNaiveBayesAnalyzer()
    
    # Create sample dataset
    print("\n2. Creating sample dataset...")
    texts, labels = create_sample_dataset()
    print(f"Created dataset with {len(texts)} examples")
    
    # Analyze lexicon coverage
    print("\n3. Analyzing lexicon coverage...")
    coverage = analyzer.analyze_lexicon_coverage(texts)
    
    # Train model
    print("\n4. Training model...")
    results = analyzer.train(texts, labels)
    
    # Test predictions
    print("\n5. Testing predictions...")
    test_texts = [
        "Film ini sangat bagus dan menghibur sekali",
        "Saya tidak suka dengan film ini, sangat buruk",
        "Film yang biasa saja, tidak terlalu istimewa",
        "Ceritanya hebat dan aktingnya luar biasa",
        "Sangat membosankan dan tidak menarik"
    ]
    
    print("\nPrediction Results:")
    print("-" * 40)
    for text in test_texts:
        result = analyzer.predict_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"Lexicon score: {result['lexicon_score']:.3f}")
        print()
    
    # Save model
    print("6. Saving model...")
    analyzer.save_model('indonesian_sentiment_model.pkl')
    
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
