#!/usr/bin/env python3
"""
Quick Demo of Comprehensive Statistical Analysis
Demonstrates key functionality with a smaller sample for faster execution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def quick_demo():
    """Run a quick demonstration of all analysis components"""
    print("="*80)
    print("QUICK DEMO - COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*80)
    
    # Load data (sample for speed)
    print("Loading dataset...")
    df = pd.read_csv('google_play_reviews_DigitalBank_sentiment_analysis.csv')
    
    # Use smaller sample for demo
    sample_size = 2000
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"Using sample of {sample_size} records for demo")
    
    sentiment_columns = ['sentiment_score_based', 'sentiment_textblob', 
                        'sentiment_vader', 'sentiment_ensemble']
    
    # 1. SENTIMENT LABEL COMPARISON
    print("\n" + "="*60)
    print("1. SENTIMENT LABEL COMPARISON")
    print("="*60)
    
    # Basic statistics
    print("\nSentiment Distribution:")
    for col in sentiment_columns:
        print(f"\n{col.replace('sentiment_', '').replace('_', ' ').title()}:")
        distribution = df_sample[col].value_counts()
        percentage = df_sample[col].value_counts(normalize=True) * 100
        for label in distribution.index:
            print(f"  {label}: {distribution[label]} ({percentage[label]:.1f}%)")
    
    # Agreement analysis
    print("\nAgreement Analysis:")
    agreement_matrix = pd.DataFrame(index=sentiment_columns, columns=sentiment_columns)
    for col1 in sentiment_columns:
        for col2 in sentiment_columns:
            if col1 == col2:
                agreement_matrix.loc[col1, col2] = 1.0
            else:
                agreement = (df_sample[col1] == df_sample[col2]).mean()
                agreement_matrix.loc[col1, col2] = agreement
    
    agreement_matrix = agreement_matrix.astype(float)
    print("Agreement Matrix:")
    print(agreement_matrix.round(3))
    
    # 2. FEATURE EXTRACTION COMPARISON
    print("\n" + "="*60)
    print("2. FEATURE EXTRACTION COMPARISON")
    print("="*60)
    
    texts = df_sample['stemmed_text'].fillna('').astype(str)
    
    # TF-IDF features
    print("Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
    
    # Simple word count features (as Word2Vec substitute for demo)
    print("Creating simple word count features...")
    word_count_features = np.array([[len(text.split())] for text in texts])
    
    feature_data = {
        'TF-IDF': tfidf_features,
        'Word_Count': word_count_features
    }
    
    print(f"TF-IDF shape: {tfidf_features.shape}")
    print(f"Word Count shape: {word_count_features.shape}")
    
    # 3. ML ALGORITHM COMPARISON
    print("\n" + "="*60)
    print("3. ML ALGORITHM COMPARISON")
    print("="*60)
    
    algorithms = {
        'SVM_Linear': SVC(kernel='linear', random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42),
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random_Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Naive_Bayes': MultinomialNB()
    }
    
    # Test with one sentiment method for demo
    sentiment_method = 'sentiment_score_based'
    le = LabelEncoder()
    y = le.fit_transform(df_sample[sentiment_method])
    
    algorithm_results = []
    
    for feature_name, X in feature_data.items():
        if feature_name == 'Word_Count':
            # Skip some algorithms for word count (too simple)
            test_algorithms = {'Logistic_Regression': algorithms['Logistic_Regression']}
        else:
            test_algorithms = algorithms
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        for algo_name, algorithm in test_algorithms.items():
            try:
                algorithm.fit(X_train, y_train)
                y_pred = algorithm.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                algorithm_results.append({
                    'Feature_Type': feature_name,
                    'Algorithm': algo_name,
                    'Accuracy': accuracy
                })
                
                print(f"{feature_name} + {algo_name}: {accuracy:.4f}")
                
            except Exception as e:
                print(f"Error with {algo_name} + {feature_name}: {e}")
    
    # 4. DATA SPLIT COMPARISON
    print("\n" + "="*60)
    print("4. DATA SPLIT COMPARISON")
    print("="*60)
    
    split_scenarios = [0.25, 0.30, 0.35, 0.65, 0.70, 0.75]
    best_algorithm = SVC(kernel='rbf', random_state=42)
    X = tfidf_features  # Use TF-IDF for split comparison
    
    split_results = []
    
    for train_size in split_scenarios:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=42, stratify=y
            )
            
            best_algorithm.fit(X_train, y_train)
            y_pred = best_algorithm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            split_results.append({
                'Train_Size': train_size,
                'Test_Size': 1 - train_size,
                'Train_Samples': len(X_train),
                'Test_Samples': len(X_test),
                'Accuracy': accuracy
            })
            
            print(f"Train: {train_size*100:.0f}% ({len(X_train)} samples) | "
                  f"Test: {(1-train_size)*100:.0f}% ({len(X_test)} samples) | "
                  f"Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error with split {train_size}: {e}")
    
    # SUMMARY AND VISUALIZATIONS
    print("\n" + "="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    # Best algorithm
    if algorithm_results:
        best_algo = max(algorithm_results, key=lambda x: x['Accuracy'])
        print(f"Best Algorithm: {best_algo['Algorithm']} with {best_algo['Feature_Type']}")
        print(f"Best Accuracy: {best_algo['Accuracy']:.4f}")
    
    # Best split
    if split_results:
        best_split = max(split_results, key=lambda x: x['Accuracy'])
        print(f"Best Split: {best_split['Train_Size']*100:.0f}% training")
        print(f"Best Split Accuracy: {best_split['Accuracy']:.4f}")
    
    # Create simple visualizations
    print("\nCreating visualizations...")
    
    # 1. Sentiment distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Quick Demo - Statistical Analysis Results', fontsize=14)
    
    # Sentiment distribution
    ax1 = axes[0, 0]
    sentiment_counts = []
    labels = []
    for col in sentiment_columns[:2]:  # Show first 2 for space
        counts = df_sample[col].value_counts()
        sentiment_counts.extend(counts.values)
        labels.extend([f"{col.replace('sentiment_', '')}\n{label}" for label in counts.index])
    
    ax1.bar(range(len(sentiment_counts)), sentiment_counts)
    ax1.set_title('Sentiment Distribution (Sample)')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # Algorithm comparison
    ax2 = axes[0, 1]
    if algorithm_results:
        algo_df = pd.DataFrame(algorithm_results)
        algo_summary = algo_df.groupby('Algorithm')['Accuracy'].mean()
        algo_summary.plot(kind='bar', ax=ax2)
        ax2.set_title('Algorithm Performance')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
    
    # Split comparison
    ax3 = axes[1, 0]
    if split_results:
        split_df = pd.DataFrame(split_results)
        ax3.plot(split_df['Train_Size'], split_df['Accuracy'], 'o-')
        ax3.set_title('Performance vs Training Size')
        ax3.set_xlabel('Training Size')
        ax3.set_ylabel('Accuracy')
    
    # Agreement heatmap
    ax4 = axes[1, 1]
    method_names = [col.replace('sentiment_', '').replace('_', ' ') for col in sentiment_columns]
    agreement_renamed = agreement_matrix.copy()
    agreement_renamed.index = method_names
    agreement_renamed.columns = method_names
    
    sns.heatmap(agreement_renamed, annot=True, cmap='Blues', ax=ax4, fmt='.2f')
    ax4.set_title('Method Agreement')
    
    plt.tight_layout()
    plt.savefig('quick_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("QUICK DEMO COMPLETED!")
    print("="*80)
    print("For full analysis, run: python comprehensive_analysis_statistics.py")

if __name__ == "__main__":
    quick_demo()
