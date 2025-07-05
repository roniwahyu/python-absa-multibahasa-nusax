#!/usr/bin/env python3
"""
Quick Comparison Demo - Shows sample statistics and visualizations
Demonstrates the types of comparisons that will be generated
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def create_sample_comparison():
    """Create sample comparison with limited data for demonstration"""
    print("="*70)
    print("QUICK COMPARISON DEMO")
    print("="*70)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('google_play_reviews_DigitalBank_sentiment_analysis.csv')
    
    # Use sample for quick demo
    sample_size = 2000
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"Using sample of {sample_size} records")
    
    # Prepare features
    texts = df_sample['stemmed_text'].fillna('').astype(str)
    
    # Quick TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
    
    # Quick Word2Vec
    tokenized_texts = [simple_preprocess(text) for text in texts]
    w2v_model = Word2Vec(tokenized_texts, vector_size=50, window=3, min_count=2, epochs=5)
    
    def get_doc_vector(tokens, model):
        vectors = [model.wv[token] for token in tokens if token in model.wv.key_to_index]
        return np.mean(vectors, axis=0) if vectors else np.zeros(50)
    
    w2v_features = np.array([get_doc_vector(tokens, w2v_model) for tokens in tokenized_texts])
    
    # Prepare labels
    le = LabelEncoder()
    y = le.fit_transform(df_sample['sentiment_score_based'])
    
    # Test configurations
    feature_types = {'TF-IDF': tfidf_features, 'Word2Vec': w2v_features}
    kernels = ['linear', 'rbf']
    train_sizes = [0.65, 0.70, 0.75]
    
    results = []
    
    print("\nRunning quick experiments...")
    for feature_name, X in feature_types.items():
        for train_size in train_sizes:
            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=42, stratify=y
            )
            
            for kernel in kernels:
                print(f"Testing: {feature_name} | {kernel} | {train_size*100:.0f}%/{test_size*100:.0f}%")
                
                svm = SVC(kernel=kernel, random_state=42)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results.append({
                    'Feature': feature_name,
                    'Kernel': kernel,
                    'Train_Size': f"{train_size*100:.0f}%",
                    'Test_Size': f"{test_size*100:.0f}%",
                    'Split_Label': f"{train_size*100:.0f}%/{test_size*100:.0f}%",
                    'Accuracy': accuracy,
                    'Train_Samples': len(X_train),
                    'Test_Samples': len(X_test)
                })
    
    df_results = pd.DataFrame(results)
    
    # Generate comparison tables
    print("\n" + "="*70)
    print("1. FEATURE EXTRACTION COMPARISON")
    print("="*70)
    
    feature_comparison = df_results.groupby('Feature').agg({
        'Accuracy': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    print(feature_comparison)
    
    print("\n" + "="*70)
    print("2. KERNEL COMPARISON")
    print("="*70)
    
    kernel_comparison = df_results.groupby('Kernel').agg({
        'Accuracy': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    print(kernel_comparison)
    
    print("\n" + "="*70)
    print("3. DATA SPLIT COMPARISON")
    print("="*70)
    
    split_comparison = df_results.groupby('Split_Label').agg({
        'Accuracy': ['mean', 'std', 'min', 'max'],
        'Train_Samples': ['mean'],
        'Test_Samples': ['mean']
    }).round(4)
    print(split_comparison)
    
    # Create visualizations
    create_demo_visualizations(df_results)
    
    # Detailed statistics
    print("\n" + "="*70)
    print("4. DETAILED STATISTICS")
    print("="*70)
    
    print("\nOverall Performance Summary:")
    print(f"Best Accuracy: {df_results['Accuracy'].max():.4f}")
    print(f"Mean Accuracy: {df_results['Accuracy'].mean():.4f} ± {df_results['Accuracy'].std():.4f}")
    print(f"Worst Accuracy: {df_results['Accuracy'].min():.4f}")
    
    print("\nBest Configuration:")
    best_config = df_results.loc[df_results['Accuracy'].idxmax()]
    print(f"Feature: {best_config['Feature']}")
    print(f"Kernel: {best_config['Kernel']}")
    print(f"Split: {best_config['Split_Label']}")
    print(f"Accuracy: {best_config['Accuracy']:.4f}")
    
    print("\nFeature Type Performance:")
    for feature in df_results['Feature'].unique():
        mean_acc = df_results[df_results['Feature'] == feature]['Accuracy'].mean()
        print(f"{feature}: {mean_acc:.4f} average accuracy")
    
    print("\nKernel Performance:")
    for kernel in df_results['Kernel'].unique():
        mean_acc = df_results[df_results['Kernel'] == kernel]['Accuracy'].mean()
        print(f"{kernel}: {mean_acc:.4f} average accuracy")
    
    print("\nData Split Performance:")
    for split in df_results['Split_Label'].unique():
        mean_acc = df_results[df_results['Split_Label'] == split]['Accuracy'].mean()
        print(f"{split}: {mean_acc:.4f} average accuracy")
    
    return df_results

def create_demo_visualizations(df_results):
    """Create demonstration visualizations"""
    print("\nCreating demonstration visualizations...")
    
    # 1. Feature Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Quick Comparison Demo - Performance Analysis', fontsize=16, fontweight='bold')
    
    # Feature comparison
    ax1 = axes[0, 0]
    sns.boxplot(data=df_results, x='Feature', y='Accuracy', ax=ax1)
    ax1.set_title('Accuracy by Feature Type')
    ax1.set_ylabel('Accuracy')
    
    # Add mean values
    for i, feature in enumerate(df_results['Feature'].unique()):
        mean_val = df_results[df_results['Feature'] == feature]['Accuracy'].mean()
        ax1.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Kernel comparison
    ax2 = axes[0, 1]
    sns.boxplot(data=df_results, x='Kernel', y='Accuracy', ax=ax2)
    ax2.set_title('Accuracy by Kernel Type')
    ax2.set_ylabel('Accuracy')
    
    # Split comparison
    ax3 = axes[1, 0]
    sns.boxplot(data=df_results, x='Split_Label', y='Accuracy', ax=ax3)
    ax3.set_title('Accuracy by Train/Test Split')
    ax3.set_ylabel('Accuracy')
    ax3.tick_params(axis='x', rotation=45)
    
    # Heatmap
    ax4 = axes[1, 1]
    pivot_table = df_results.pivot_table(values='Accuracy', index='Feature', columns='Kernel', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Feature vs Kernel Heatmap')
    
    plt.tight_layout()
    plt.savefig('quick_comparison_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Detailed comparison chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create grouped bar chart
    feature_kernel_means = df_results.groupby(['Feature', 'Kernel'])['Accuracy'].mean().unstack()
    feature_kernel_means.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Mean Accuracy by Feature Type and Kernel', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Type')
    ax.set_ylabel('Mean Accuracy')
    ax.legend(title='Kernel', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
    
    plt.tight_layout()
    plt.savefig('feature_kernel_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Demo visualizations saved!")

def main():
    """Main execution"""
    df_results = create_sample_comparison()
    
    print("\n" + "="*70)
    print("DEMO COMPLETED!")
    print("="*70)
    print("This demonstrates the types of comparisons and visualizations")
    print("that will be generated in the full comprehensive analysis.")
    print("\nThe full analysis will include:")
    print("- All 4 sentiment methods")
    print("- Complete TF-IDF and Word2Vec feature extraction")
    print("- All 4 SVM kernels (linear, rbf, poly, sigmoid)")
    print("- All data split scenarios (65%/35%, 70%/30%, 75%/25%)")
    print("- Statistical significance tests")
    print("- Detailed performance metrics")
    print("- Comprehensive visualizations")

if __name__ == "__main__":
    main()
