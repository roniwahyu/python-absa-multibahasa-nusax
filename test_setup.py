#!/usr/bin/env python3
"""
Test script to verify the setup and basic functionality.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        from google_play_scraper import app, reviews, Sort
        print("✓ google-play-scraper imported successfully")
    except ImportError as e:
        print(f"✗ google-play-scraper import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        print("✓ transformers imported successfully")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ torch imported successfully (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available, using CPU")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✓ visualization libraries imported successfully")
    except ImportError as e:
        print(f"✗ visualization libraries import failed: {e}")
        return False
    
    return True

def test_google_play_scraper():
    """Test Google Play Scraper with a simple query."""
    print("\nTesting Google Play Scraper...")
    
    try:
        from google_play_scraper import app
        
        # Test with Instagram app
        app_info = app('com.instagram.android', lang='id', country='id')
        print(f"✓ Successfully retrieved app info for: {app_info['title']}")
        print(f"  Developer: {app_info['developer']}")
        print(f"  Rating: {app_info['score']}")
        print(f"  Reviews: {app_info['reviews']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Google Play Scraper test failed: {e}")
        return False

def test_sentiment_model():
    """Test sentiment analysis model loading."""
    print("\nTesting sentiment analysis model...")
    
    try:
        from transformers import pipeline
        
        # Test with a simple multilingual model
        sentiment_pipeline = pipeline("sentiment-analysis", 
                                    model="nlptown/bert-base-multilingual-uncased-sentiment")
        
        # Test with Indonesian text
        test_text = "Aplikasi ini sangat bagus dan mudah digunakan"
        result = sentiment_pipeline(test_text)
        
        print(f"✓ Sentiment model loaded successfully")
        print(f"  Test text: {test_text}")
        print(f"  Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sentiment model test failed: {e}")
        return False

def test_nusax_utils():
    """Test NusaX utilities."""
    print("\nTesting NusaX utilities...")
    
    try:
        from nusax_utils import NusaXLanguageDetector, NusaXTextPreprocessor
        
        # Test language detection
        detector = NusaXLanguageDetector()
        test_texts = [
            "Aplikasi ini bagus banget",
            "Aplikasi iki apik tenan",
            "Aplikasi ieu saé pisan"
        ]
        
        for text in test_texts:
            lang = detector.get_dominant_language(text)
            print(f"  '{text}' -> {lang}")
        
        # Test text preprocessing
        preprocessor = NusaXTextPreprocessor()
        test_text = "Gk tau knp aplikasi ini sering error :("
        cleaned = preprocessor.clean_text(test_text)
        print(f"  Preprocessing: '{test_text}' -> '{cleaned}'")
        
        print("✓ NusaX utilities working correctly")
        return True
        
    except Exception as e:
        print(f"✗ NusaX utilities test failed: {e}")
        return False

def test_main_analyzer():
    """Test the main analyzer class."""
    print("\nTesting main analyzer...")
    
    try:
        from playstore_sentiment_analyzer import ReviewAnalyzer
        
        # Initialize analyzer
        analyzer = ReviewAnalyzer()
        print("✓ ReviewAnalyzer initialized successfully")
        
        # Test setup (without actually scraping)
        analyzer.scraper = None  # We'll test setup without scraping
        analyzer.sentiment_analyzer.load_models()
        print("✓ Sentiment models loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Main analyzer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Instagram Reviews Sentiment Analyzer - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Google Play Scraper Test", test_google_play_scraper),
        ("Sentiment Model Test", test_sentiment_model),
        ("NusaX Utilities Test", test_nusax_utils),
        ("Main Analyzer Test", test_main_analyzer)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The setup is ready for use.")
        print("\nYou can now run the main analysis with:")
        print("  python run_analysis.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)