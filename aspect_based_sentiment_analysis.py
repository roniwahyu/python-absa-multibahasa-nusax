#!/usr/bin/env python3
"""
Aspect-Based Sentiment Analysis with Multi-Class SVM

This script implements comprehensive aspect-based sentiment analysis using:
- Feature extraction: TF-IDF and Word2Vec
- Sentiment labeling: Score-based, TextBlob, VADER, and Ensemble methods
- Classification: Multi-Class SVM with various kernels
- Data splits: 65%, 70%, 75% training scenarios
- Evaluation: Confusion Matrix and ROC/AUC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import warnings
import os
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data(file_path):
    """Load and explore the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Data exploration
    sentiment_columns = ['sentiment_score_based', 'sentiment_textblob', 'sentiment_vader', 'sentiment_ensemble']
    
    print("\nSentiment Distribution:")
    for col in sentiment_columns:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print(f"Percentage distribution:")
        print(df[col].value_counts(normalize=True) * 100)
    
    return df, sentiment_columns

def create_visualizations(df, sentiment_columns, save_plots=True):
    """Create and save visualization plots"""
    print("\nCreating visualizations...")
    
    # Sentiment distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sentiment Distribution Across Different Methods', fontsize=16)
    
    for i, col in enumerate(sentiment_columns):
        ax = axes[i//2, i%2]
        df[col].value_counts().plot(kind='bar', ax=ax, title=col.replace('_', ' ').title())
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def extract_features(texts):
    """Extract TF-IDF and Word2Vec features"""
    print("\nExtracting features...")
    
    # 1. TF-IDF Feature Extraction
    print("Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words=None  # Already preprocessed
    )
    
    tfidf_features = tfidf_vectorizer.fit_transform(texts)
    print(f"TF-IDF feature shape: {tfidf_features.shape}")
    
    # 2. Word2Vec Feature Extraction
    print("Training Word2Vec model...")
    tokenized_texts = [simple_preprocess(text) for text in texts]
    
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
    )
    
    print(f"Word2Vec vocabulary size: {len(w2v_model.wv.key_to_index)}")
    
    # Create document vectors
    def get_document_vector(tokens, model, vector_size=100):
        vectors = []
        for token in tokens:
            if token in model.wv.key_to_index:
                vectors.append(model.wv[token])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vector_size)
    
    print("Creating Word2Vec document vectors...")
    w2v_features = np.array([get_document_vector(tokens, w2v_model) for tokens in tokenized_texts])
    print(f"Word2Vec feature shape: {w2v_features.shape}")
    
    # Convert TF-IDF to dense array
    tfidf_features_dense = tfidf_features.toarray()
    
    return {
        'TF-IDF': tfidf_features_dense,
        'Word2Vec': w2v_features
    }, tfidf_vectorizer, w2v_model

def prepare_labels(df, sentiment_methods):
    """Encode sentiment labels"""
    print("\nPreparing labels...")
    
    label_encoders = {}
    encoded_labels = {}
    
    for method in sentiment_methods:
        le = LabelEncoder()
        encoded_labels[method] = le.fit_transform(df[method])
        label_encoders[method] = le
        print(f"{method} classes: {le.classes_}")
    
    return label_encoders, encoded_labels

def evaluate_svm_model(X_train, X_test, y_train, y_test, kernel, method_name, feature_name, train_size):
    """Train and evaluate SVM model"""
    
    # Train SVM
    svm_model = SVC(kernel=kernel, probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = svm_model.predict(X_test)
    y_pred_proba = svm_model.predict_proba(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC (for multiclass)
    try:
        if len(np.unique(y_test)) > 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    except:
        roc_auc = None
    
    return {
        'method': method_name,
        'feature': feature_name,
        'kernel': kernel,
        'train_size': train_size,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'model': svm_model
    }

def run_comprehensive_evaluation(feature_data, encoded_labels, sentiment_methods):
    """Run comprehensive SVM evaluation"""
    print("\nStarting comprehensive SVM evaluation...")
    
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    train_sizes = [0.65, 0.70, 0.75]
    feature_types = list(feature_data.keys())
    
    results = []
    total_experiments = len(sentiment_methods) * len(feature_types) * len(kernels) * len(train_sizes)
    current_experiment = 0
    
    start_time = time.time()
    
    for method in sentiment_methods:
        y = encoded_labels[method]
        
        for feature_name, X in feature_data.items():
            for train_size in train_sizes:
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=train_size, random_state=42, stratify=y
                )
                
                for kernel in kernels:
                    current_experiment += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_exp = elapsed_time / current_experiment if current_experiment > 0 else 0
                    remaining_time = avg_time_per_exp * (total_experiments - current_experiment)
                    
                    print(f"Experiment {current_experiment}/{total_experiments}: {method} | {feature_name} | {kernel} | Train: {train_size*100:.0f}% | ETA: {remaining_time/60:.1f}min")
                    
                    try:
                        result = evaluate_svm_model(
                            X_train, X_test, y_train, y_test, 
                            kernel, method, feature_name, train_size
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"Error in experiment: {e}")
                        continue
    
    total_time = time.time() - start_time
    print(f"\nCompleted {len(results)} successful experiments in {total_time/60:.1f} minutes!")
    
    return results

def analyze_results(results, label_encoders, save_plots=True):
    """Analyze and visualize results"""
    print("\nAnalyzing results...")
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'Method': r['method'],
            'Feature': r['feature'],
            'Kernel': r['kernel'],
            'Train_Size': r['train_size'],
            'Accuracy': r['accuracy'],
            'ROC_AUC': r['roc_auc']
        }
        for r in results
    ])
    
    print("Results Summary:")
    summary = results_df.groupby(['Method', 'Feature', 'Kernel']).agg({
        'Accuracy': ['mean', 'std'],
        'ROC_AUC': ['mean', 'std']
    }).round(4)
    print(summary)
    
    # Find best performing models
    print("\nTop 10 Best Performing Models by Accuracy:")
    top_accuracy = results_df.nlargest(10, 'Accuracy')
    print(top_accuracy[['Method', 'Feature', 'Kernel', 'Train_Size', 'Accuracy', 'ROC_AUC']].to_string(index=False))
    
    if not results_df['ROC_AUC'].isna().all():
        print("\nTop 10 Best Performing Models by ROC AUC:")
        top_roc = results_df.dropna(subset=['ROC_AUC']).nlargest(10, 'ROC_AUC')
        print(top_roc[['Method', 'Feature', 'Kernel', 'Train_Size', 'Accuracy', 'ROC_AUC']].to_string(index=False))
    
    return results_df

def create_performance_visualizations(results_df, save_plots=True):
    """Create comprehensive performance visualization plots"""
    print("\nCreating performance visualizations...")

    # Performance comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('SVM Performance Comparison Across Different Configurations', fontsize=16)

    # 1. Accuracy by Method and Feature
    ax1 = axes[0, 0]
    sns.boxplot(data=results_df, x='Method', y='Accuracy', hue='Feature', ax=ax1)
    ax1.set_title('Accuracy by Sentiment Method and Feature Type')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Accuracy by Kernel
    ax2 = axes[0, 1]
    sns.boxplot(data=results_df, x='Kernel', y='Accuracy', ax=ax2)
    ax2.set_title('Accuracy by SVM Kernel')

    # 3. Accuracy by Training Size
    ax3 = axes[1, 0]
    sns.boxplot(data=results_df, x='Train_Size', y='Accuracy', ax=ax3)
    ax3.set_title('Accuracy by Training Size')

    # 4. ROC AUC by Method and Feature
    ax4 = axes[1, 1]
    results_df_clean = results_df.dropna(subset=['ROC_AUC'])
    if not results_df_clean.empty:
        sns.boxplot(data=results_df_clean, x='Method', y='ROC_AUC', hue='Feature', ax=ax4)
        ax4.set_title('ROC AUC by Sentiment Method and Feature Type')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No ROC AUC data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('ROC AUC by Sentiment Method and Feature Type')

    plt.tight_layout()
    if save_plots:
        plt.savefig('svm_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrices(results, label_encoders, save_plots=True):
    """Create confusion matrix visualizations for best models"""
    print("\nCreating confusion matrices for best models...")

    # Get best model for each sentiment method
    results_df = pd.DataFrame([
        {
            'Method': r['method'],
            'Feature': r['feature'],
            'Kernel': r['kernel'],
            'Train_Size': r['train_size'],
            'Accuracy': r['accuracy'],
            'ROC_AUC': r['roc_auc']
        }
        for r in results
    ])

    best_models_by_method = results_df.loc[results_df.groupby('Method')['Accuracy'].idxmax()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Confusion Matrices for Best Models by Sentiment Method', fontsize=16)

    for i, (_, row) in enumerate(best_models_by_method.iterrows()):
        if i >= 4:  # Only show first 4
            break

        ax = axes[i//2, i%2]

        # Find corresponding result
        result = next(r for r in results if (
            r['method'] == row['Method'] and
            r['feature'] == row['Feature'] and
            r['kernel'] == row['Kernel'] and
            r['train_size'] == row['Train_Size']
        ))

        cm = result['confusion_matrix']
        class_names = label_encoders[result['method']].classes_

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"{row['Method']}\n{row['Feature']} | {row['Kernel']} | Acc: {row['Accuracy']:.3f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    if save_plots:
        plt.savefig('confusion_matrices_best_models.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_summary(results_df, feature_data):
    """Generate comprehensive analysis summary"""
    print("\n" + "="*80)
    print("ASPECT-BASED SENTIMENT ANALYSIS - COMPREHENSIVE SUMMARY")
    print("="*80)

    print(f"\nDataset Information:")
    print(f"- Total samples: {len(results_df) // (len(results_df['Method'].unique()) * len(results_df['Feature'].unique()) * len(results_df['Kernel'].unique()) * len(results_df['Train_Size'].unique()))}")
    print(f"- Features extracted: TF-IDF ({feature_data['TF-IDF'].shape[1]} features), Word2Vec (100 dimensions)")
    print(f"- Sentiment methods compared: {len(results_df['Method'].unique())}")
    print(f"- SVM kernels tested: {results_df['Kernel'].unique().tolist()}")
    print(f"- Training sizes tested: {[f'{size*100:.0f}%' for size in sorted(results_df['Train_Size'].unique())]}")
    print(f"- Total experiments conducted: {len(results_df)}")

    print(f"\nOverall Performance Summary:")
    print(f"- Best accuracy achieved: {results_df['Accuracy'].max():.4f}")
    print(f"- Average accuracy across all experiments: {results_df['Accuracy'].mean():.4f} ± {results_df['Accuracy'].std():.4f}")
    if not results_df['ROC_AUC'].isna().all():
        print(f"- Best ROC AUC achieved: {results_df['ROC_AUC'].max():.4f}")
        print(f"- Average ROC AUC: {results_df['ROC_AUC'].mean():.4f} ± {results_df['ROC_AUC'].std():.4f}")

    print(f"\nBest Performing Configuration:")
    best_config = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"- Sentiment Method: {best_config['Method']}")
    print(f"- Feature Type: {best_config['Feature']}")
    print(f"- SVM Kernel: {best_config['Kernel']}")
    print(f"- Training Size: {best_config['Train_Size']*100:.0f}%")
    print(f"- Accuracy: {best_config['Accuracy']:.4f}")
    if pd.notna(best_config['ROC_AUC']):
        print(f"- ROC AUC: {best_config['ROC_AUC']:.4f}")

    print(f"\nKey Insights:")
    # Feature type performance
    feature_performance = results_df.groupby('Feature')['Accuracy'].mean()
    best_feature = feature_performance.idxmax()
    print(f"- Best feature type: {best_feature} (avg accuracy: {feature_performance[best_feature]:.4f})")

    # Kernel performance
    kernel_performance = results_df.groupby('Kernel')['Accuracy'].mean()
    best_kernel = kernel_performance.idxmax()
    print(f"- Best SVM kernel: {best_kernel} (avg accuracy: {kernel_performance[best_kernel]:.4f})")

    # Training size impact
    size_performance = results_df.groupby('Train_Size')['Accuracy'].mean()
    best_size = size_performance.idxmax()
    print(f"- Optimal training size: {best_size*100:.0f}% (avg accuracy: {size_performance[best_size]:.4f})")

    # Sentiment method performance
    method_performance = results_df.groupby('Method')['Accuracy'].mean()
    best_method = method_performance.idxmax()
    print(f"- Best sentiment method: {best_method} (avg accuracy: {method_performance[best_method]:.4f})")

    print(f"\nRecommendations:")
    print(f"1. Use {best_feature} features for better performance")
    print(f"2. {best_kernel} kernel shows best results for this dataset")
    print(f"3. {best_size*100:.0f}% training split provides optimal balance")
    print(f"4. {best_method} sentiment labeling method is most suitable")
    print(f"5. Consider ensemble methods for production deployment")

def main():
    """Main execution function"""
    print("="*80)
    print("ASPECT-BASED SENTIMENT ANALYSIS WITH MULTI-CLASS SVM")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    file_path = 'google_play_reviews_DigitalBank_sentiment_analysis.csv'
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at {file_path}")
        return
    
    df, sentiment_methods = load_and_explore_data(file_path)
    
    # Create visualizations
    create_visualizations(df, sentiment_methods)
    
    # Prepare text data
    texts = df['stemmed_text'].fillna('').astype(str)
    
    # Extract features
    feature_data, _, _ = extract_features(texts)

    # Prepare labels
    label_encoders, encoded_labels = prepare_labels(df, sentiment_methods)

    # Run evaluation
    results = run_comprehensive_evaluation(feature_data, encoded_labels, sentiment_methods)

    # Analyze results
    results_df = analyze_results(results, label_encoders)

    # Create additional visualizations
    create_performance_visualizations(results_df)
    create_confusion_matrices(results, label_encoders)

    # Generate comprehensive summary
    generate_comprehensive_summary(results_df, feature_data)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'svm_results_{timestamp}.csv', index=False)
    print(f"\nResults saved to: svm_results_{timestamp}.csv")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
