#!/usr/bin/env python3
"""
Quick test script for Aspect-Based Sentiment Analysis
Tests basic functionality with a smaller subset of experiments
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def quick_test():
    """Run a quick test with limited experiments"""
    print("="*60)
    print("QUICK TEST - ASPECT-BASED SENTIMENT ANALYSIS")
    print("="*60)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('google_play_reviews_DigitalBank_sentiment_analysis.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Use a smaller sample for quick testing
    sample_size = 1000
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"Using sample of {sample_size} records for quick test")
    
    # Prepare text data
    texts = df_sample['stemmed_text'].fillna('').astype(str)
    
    # Extract TF-IDF features (simplified)
    print("\nExtracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,  # Reduced for speed
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X = tfidf_vectorizer.fit_transform(texts).toarray()
    print(f"Feature shape: {X.shape}")
    
    # Test with one sentiment method
    sentiment_method = 'sentiment_score_based'
    print(f"\nTesting with: {sentiment_method}")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df_sample[sentiment_method])
    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Test different kernels
    kernels = ['linear', 'rbf']
    results = []
    
    for kernel in kernels:
        print(f"\nTesting SVM with {kernel} kernel...")
        
        # Train SVM
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train, y_train)
        
        # Predict
        y_pred = svm.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        results.append({
            'kernel': kernel,
            'accuracy': accuracy
        })
        
        # Show classification report
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Show confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(cm)
    
    # Summary
    print("\n" + "="*60)
    print("QUICK TEST RESULTS SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"{result['kernel']} kernel: {result['accuracy']:.4f} accuracy")
    
    best_kernel = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest performing kernel: {best_kernel['kernel']} ({best_kernel['accuracy']:.4f})")
    
    print("\nQuick test completed successfully!")
    print("You can now run the full analysis with: python aspect_based_sentiment_analysis.py")

if __name__ == "__main__":
    quick_test()
