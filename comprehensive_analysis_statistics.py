#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis and Visualization
for Aspect-Based Sentiment Analysis

This script generates detailed statistics, tables, and visualizations for:
1. Sentiment label comparison (score_based, textblob, vader, ensemble)
2. Feature extraction comparison (TF-IDF vs Word2Vec)
3. ML algorithm comparison (SVM, Linear Regression, Random Forest, Naive Bayes)
4. Data split scenario comparison (25%, 30%, 35%, 65%, 70%, 75%)
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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import warnings
import time
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with dataset"""
        self.data_path = data_path
        self.df = None
        self.sentiment_columns = ['sentiment_score_based', 'sentiment_textblob', 
                                'sentiment_vader', 'sentiment_ensemble']
        self.feature_data = {}
        self.label_encoders = {}
        self.encoded_labels = {}
        self.results = []
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Prepare text data
        self.texts = self.df['stemmed_text'].fillna('').astype(str)
        
        # Encode labels
        for col in self.sentiment_columns:
            le = LabelEncoder()
            self.encoded_labels[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            
        print("Data loaded and labels encoded successfully!")
        
    def extract_features(self):
        """Extract TF-IDF and Word2Vec features"""
        print("\nExtracting features...")
        
        # TF-IDF Feature Extraction
        print("Extracting TF-IDF features...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        tfidf_features = tfidf_vectorizer.fit_transform(self.texts).toarray()
        self.feature_data['TF-IDF'] = tfidf_features
        
        # Word2Vec Feature Extraction
        print("Training Word2Vec model...")
        tokenized_texts = [simple_preprocess(text) for text in self.texts]
        w2v_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            epochs=10
        )
        
        def get_document_vector(tokens, model, vector_size=100):
            vectors = []
            for token in tokens:
                if token in model.wv.key_to_index:
                    vectors.append(model.wv[token])
            return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
        
        w2v_features = np.array([get_document_vector(tokens, w2v_model) 
                                for tokens in tokenized_texts])
        self.feature_data['Word2Vec'] = w2v_features
        
        print(f"TF-IDF shape: {tfidf_features.shape}")
        print(f"Word2Vec shape: {w2v_features.shape}")
        
    def analyze_sentiment_labels(self):
        """1. Comprehensive sentiment label analysis"""
        print("\n" + "="*80)
        print("1. SENTIMENT LABEL COMPARISON ANALYSIS")
        print("="*80)
        
        # Basic statistics
        sentiment_stats = {}
        for col in self.sentiment_columns:
            stats_dict = {
                'total_samples': len(self.df[col]),
                'unique_labels': self.df[col].nunique(),
                'labels': self.df[col].unique().tolist(),
                'distribution': self.df[col].value_counts().to_dict(),
                'percentage': (self.df[col].value_counts(normalize=True) * 100).to_dict()
            }
            sentiment_stats[col] = stats_dict
        
        # Create comparison table
        comparison_data = []
        for col in self.sentiment_columns:
            for label in ['positive', 'negative', 'neutral']:
                count = sentiment_stats[col]['distribution'].get(label, 0)
                percentage = sentiment_stats[col]['percentage'].get(label, 0)
                comparison_data.append({
                    'Method': col.replace('sentiment_', '').replace('_', ' ').title(),
                    'Label': label.title(),
                    'Count': count,
                    'Percentage': round(percentage, 2)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nSentiment Label Distribution Comparison:")
        print(comparison_df.pivot(index='Method', columns='Label', values='Count').fillna(0))
        print("\nPercentage Distribution:")
        print(comparison_df.pivot(index='Method', columns='Label', values='Percentage').fillna(0))
        
        # Statistical tests
        print("\nStatistical Analysis:")
        
        # Chi-square test for independence
        from scipy.stats import chi2_contingency
        
        contingency_tables = {}
        for i, col1 in enumerate(self.sentiment_columns):
            for col2 in self.sentiment_columns[i+1:]:
                contingency_table = pd.crosstab(self.df[col1], self.df[col2])
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                contingency_tables[f"{col1}_vs_{col2}"] = {
                    'chi2': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        print("Chi-square tests for label independence:")
        for comparison, result in contingency_tables.items():
            methods = comparison.replace('sentiment_', '').replace('_vs_', ' vs ').replace('_', ' ')
            print(f"{methods}: χ² = {result['chi2']:.4f}, p = {result['p_value']:.4f}, "
                  f"Significant: {result['significant']}")
        
        # Agreement analysis
        print("\nAgreement Analysis:")
        agreement_matrix = pd.DataFrame(index=self.sentiment_columns, columns=self.sentiment_columns)
        
        for col1 in self.sentiment_columns:
            for col2 in self.sentiment_columns:
                if col1 == col2:
                    agreement_matrix.loc[col1, col2] = 1.0
                else:
                    agreement = (self.df[col1] == self.df[col2]).mean()
                    agreement_matrix.loc[col1, col2] = agreement
        
        agreement_matrix = agreement_matrix.astype(float)
        print("Agreement Matrix (proportion of matching labels):")
        print(agreement_matrix.round(4))
        
        return sentiment_stats, comparison_df, agreement_matrix
        
    def visualize_sentiment_comparison(self, comparison_df, agreement_matrix):
        """Create visualizations for sentiment label comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Sentiment Label Analysis', fontsize=16)
        
        # 1. Distribution comparison
        ax1 = axes[0, 0]
        pivot_counts = comparison_df.pivot(index='Method', columns='Label', values='Count').fillna(0)
        pivot_counts.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Label Count Distribution by Method')
        ax1.set_xlabel('Sentiment Method')
        ax1.set_ylabel('Count')
        ax1.legend(title='Sentiment Label')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Percentage comparison
        ax2 = axes[0, 1]
        pivot_pct = comparison_df.pivot(index='Method', columns='Label', values='Percentage').fillna(0)
        pivot_pct.plot(kind='bar', ax=ax2, width=0.8, stacked=True)
        ax2.set_title('Percentage Distribution by Method')
        ax2.set_xlabel('Sentiment Method')
        ax2.set_ylabel('Percentage')
        ax2.legend(title='Sentiment Label')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Agreement heatmap
        ax3 = axes[0, 2]
        method_names = [col.replace('sentiment_', '').replace('_', ' ').title() 
                       for col in self.sentiment_columns]
        agreement_renamed = agreement_matrix.copy()
        agreement_renamed.index = method_names
        agreement_renamed.columns = method_names
        
        sns.heatmap(agreement_renamed, annot=True, cmap='Blues', ax=ax3, 
                   vmin=0, vmax=1, fmt='.3f')
        ax3.set_title('Inter-Method Agreement Matrix')
        
        # 4. Individual distributions
        for i, col in enumerate(self.sentiment_columns):
            if i < 3:
                ax = axes[1, i]
                self.df[col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
                ax.set_title(f'{col.replace("sentiment_", "").replace("_", " ").title()} Distribution')
                ax.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig('sentiment_label_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def compare_feature_extraction(self):
        """2. Feature extraction comparison"""
        print("\n" + "="*80)
        print("2. FEATURE EXTRACTION COMPARISON (TF-IDF vs Word2Vec)")
        print("="*80)
        
        feature_comparison_results = []
        
        # Test with different algorithms for feature comparison
        algorithms = {
            'SVM_Linear': SVC(kernel='linear', random_state=42),
            'SVM_RBF': SVC(kernel='rbf', random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive_Bayes': MultinomialNB()
        }
        
        for sentiment_method in self.sentiment_columns:
            y = self.encoded_labels[sentiment_method]
            
            for feature_name, X in self.feature_data.items():
                # Handle negative values for Naive Bayes
                if feature_name == 'Word2Vec':
                    X_nb = X - X.min() + 1  # Make all values positive
                else:
                    X_nb = X
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                if feature_name == 'Word2Vec':
                    X_train_nb, X_test_nb = train_test_split(
                        X_nb, test_size=0.3, random_state=42, stratify=y
                    )[0], train_test_split(
                        X_nb, test_size=0.3, random_state=42, stratify=y
                    )[1]
                else:
                    X_train_nb, X_test_nb = X_train, X_test
                
                for algo_name, algorithm in algorithms.items():
                    try:
                        if algo_name == 'Naive_Bayes' and feature_name == 'Word2Vec':
                            algorithm.fit(X_train_nb, y_train)
                            y_pred = algorithm.predict(X_test_nb)
                        else:
                            algorithm.fit(X_train, y_train)
                            y_pred = algorithm.predict(X_test)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_test, y_pred, average='weighted'
                        )
                        
                        feature_comparison_results.append({
                            'Sentiment_Method': sentiment_method.replace('sentiment_', ''),
                            'Feature_Type': feature_name,
                            'Algorithm': algo_name,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1_Score': f1
                        })
                        
                    except Exception as e:
                        print(f"Error with {algo_name} + {feature_name}: {e}")
                        continue
        
        feature_results_df = pd.DataFrame(feature_comparison_results)
        
        # Summary statistics
        print("\nFeature Extraction Performance Summary:")
        summary = feature_results_df.groupby(['Feature_Type', 'Algorithm']).agg({
            'Accuracy': ['mean', 'std'],
            'F1_Score': ['mean', 'std']
        }).round(4)
        print(summary)
        
        # Statistical significance test
        print("\nStatistical Significance Test (TF-IDF vs Word2Vec):")
        for algo in feature_results_df['Algorithm'].unique():
            tfidf_scores = feature_results_df[
                (feature_results_df['Feature_Type'] == 'TF-IDF') & 
                (feature_results_df['Algorithm'] == algo)
            ]['Accuracy'].values
            
            w2v_scores = feature_results_df[
                (feature_results_df['Feature_Type'] == 'Word2Vec') & 
                (feature_results_df['Algorithm'] == algo)
            ]['Accuracy'].values
            
            if len(tfidf_scores) > 0 and len(w2v_scores) > 0:
                t_stat, p_value = stats.ttest_ind(tfidf_scores, w2v_scores)
                print(f"{algo}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        
        return feature_results_df

    def visualize_feature_comparison(self, feature_results_df):
        """Create visualizations for feature extraction comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Extraction Comparison: TF-IDF vs Word2Vec', fontsize=16)

        # 1. Accuracy comparison by algorithm
        ax1 = axes[0, 0]
        sns.boxplot(data=feature_results_df, x='Algorithm', y='Accuracy',
                   hue='Feature_Type', ax=ax1)
        ax1.set_title('Accuracy by Algorithm and Feature Type')
        ax1.tick_params(axis='x', rotation=45)

        # 2. F1-Score comparison
        ax2 = axes[0, 1]
        sns.boxplot(data=feature_results_df, x='Algorithm', y='F1_Score',
                   hue='Feature_Type', ax=ax2)
        ax2.set_title('F1-Score by Algorithm and Feature Type')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Performance by sentiment method
        ax3 = axes[1, 0]
        sns.boxplot(data=feature_results_df, x='Sentiment_Method', y='Accuracy',
                   hue='Feature_Type', ax=ax3)
        ax3.set_title('Accuracy by Sentiment Method and Feature Type')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Overall comparison
        ax4 = axes[1, 1]
        feature_summary = feature_results_df.groupby('Feature_Type').agg({
            'Accuracy': 'mean',
            'Precision': 'mean',
            'Recall': 'mean',
            'F1_Score': 'mean'
        })

        feature_summary.plot(kind='bar', ax=ax4)
        ax4.set_title('Overall Performance Comparison')
        ax4.set_xlabel('Feature Type')
        ax4.set_ylabel('Score')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.savefig('feature_extraction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def compare_ml_algorithms(self):
        """3. ML Algorithm comparison"""
        print("\n" + "="*80)
        print("3. ML ALGORITHM COMPARISON")
        print("="*80)

        # Extended algorithm comparison
        algorithms = {
            'SVM_Linear': SVC(kernel='linear', random_state=42, probability=True),
            'SVM_RBF': SVC(kernel='rbf', random_state=42, probability=True),
            'SVM_Poly': SVC(kernel='poly', random_state=42, probability=True),
            'SVM_Sigmoid': SVC(kernel='sigmoid', random_state=42, probability=True),
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive_Bayes': MultinomialNB()
        }

        algorithm_results = []

        for sentiment_method in self.sentiment_columns:
            y = self.encoded_labels[sentiment_method]

            for feature_name, X in self.feature_data.items():
                # Handle negative values for Naive Bayes
                if feature_name == 'Word2Vec':
                    X_processed = X - X.min() + 1
                else:
                    X_processed = X

                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.3, random_state=42, stratify=y
                )

                for algo_name, algorithm in algorithms.items():
                    try:
                        start_time = time.time()
                        algorithm.fit(X_train, y_train)
                        training_time = time.time() - start_time

                        start_time = time.time()
                        y_pred = algorithm.predict(X_test)
                        prediction_time = time.time() - start_time

                        accuracy = accuracy_score(y_test, y_pred)
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_test, y_pred, average='weighted'
                        )

                        # ROC AUC for multiclass
                        try:
                            if hasattr(algorithm, 'predict_proba'):
                                y_pred_proba = algorithm.predict_proba(X_test)
                                if len(np.unique(y_test)) > 2:
                                    roc_auc = roc_auc_score(y_test, y_pred_proba,
                                                          multi_class='ovr', average='weighted')
                                else:
                                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                            else:
                                roc_auc = None
                        except:
                            roc_auc = None

                        algorithm_results.append({
                            'Sentiment_Method': sentiment_method.replace('sentiment_', ''),
                            'Feature_Type': feature_name,
                            'Algorithm': algo_name,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1_Score': f1,
                            'ROC_AUC': roc_auc,
                            'Training_Time': training_time,
                            'Prediction_Time': prediction_time
                        })

                    except Exception as e:
                        print(f"Error with {algo_name} + {feature_name} + {sentiment_method}: {e}")
                        continue

        algorithm_results_df = pd.DataFrame(algorithm_results)

        # Performance summary
        print("\nAlgorithm Performance Summary:")
        algo_summary = algorithm_results_df.groupby('Algorithm').agg({
            'Accuracy': ['mean', 'std', 'max'],
            'F1_Score': ['mean', 'std', 'max'],
            'Training_Time': ['mean', 'std'],
            'Prediction_Time': ['mean', 'std']
        }).round(4)
        print(algo_summary)

        # Best performing algorithms
        print("\nTop 5 Best Performing Algorithms by Accuracy:")
        top_accuracy = algorithm_results_df.nlargest(5, 'Accuracy')[
            ['Algorithm', 'Feature_Type', 'Sentiment_Method', 'Accuracy', 'F1_Score']
        ]
        print(top_accuracy.to_string(index=False))

        # Statistical significance tests
        print("\nStatistical Significance Tests (ANOVA):")
        from scipy.stats import f_oneway

        # Group by algorithm type
        svm_scores = algorithm_results_df[
            algorithm_results_df['Algorithm'].str.contains('SVM')
        ]['Accuracy'].values

        lr_scores = algorithm_results_df[
            algorithm_results_df['Algorithm'] == 'Logistic_Regression'
        ]['Accuracy'].values

        rf_scores = algorithm_results_df[
            algorithm_results_df['Algorithm'] == 'Random_Forest'
        ]['Accuracy'].values

        nb_scores = algorithm_results_df[
            algorithm_results_df['Algorithm'] == 'Naive_Bayes'
        ]['Accuracy'].values

        if all(len(scores) > 0 for scores in [svm_scores, lr_scores, rf_scores, nb_scores]):
            f_stat, p_value = f_oneway(svm_scores, lr_scores, rf_scores, nb_scores)
            print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

        return algorithm_results_df

    def visualize_algorithm_comparison(self, algorithm_results_df):
        """Create visualizations for algorithm comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('ML Algorithm Performance Comparison', fontsize=16)

        # 1. Accuracy by algorithm
        ax1 = axes[0, 0]
        sns.boxplot(data=algorithm_results_df, x='Algorithm', y='Accuracy', ax=ax1)
        ax1.set_title('Accuracy Distribution by Algorithm')
        ax1.tick_params(axis='x', rotation=45)

        # 2. F1-Score by algorithm
        ax2 = axes[0, 1]
        sns.boxplot(data=algorithm_results_df, x='Algorithm', y='F1_Score', ax=ax2)
        ax2.set_title('F1-Score Distribution by Algorithm')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Training time comparison
        ax3 = axes[0, 2]
        sns.boxplot(data=algorithm_results_df, x='Algorithm', y='Training_Time', ax=ax3)
        ax3.set_title('Training Time by Algorithm')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylabel('Time (seconds)')

        # 4. Performance vs Time scatter
        ax4 = axes[1, 0]
        scatter_data = algorithm_results_df.groupby('Algorithm').agg({
            'Accuracy': 'mean',
            'Training_Time': 'mean'
        }).reset_index()

        sns.scatterplot(data=scatter_data, x='Training_Time', y='Accuracy',
                       s=100, ax=ax4)
        for i, row in scatter_data.iterrows():
            ax4.annotate(row['Algorithm'], (row['Training_Time'], row['Accuracy']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_title('Accuracy vs Training Time')
        ax4.set_xlabel('Training Time (seconds)')

        # 5. Algorithm performance heatmap
        ax5 = axes[1, 1]
        heatmap_data = algorithm_results_df.groupby(['Algorithm', 'Feature_Type'])['Accuracy'].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=ax5, fmt='.3f')
        ax5.set_title('Accuracy Heatmap: Algorithm vs Feature Type')

        # 6. ROC AUC comparison (if available)
        ax6 = axes[1, 2]
        roc_data = algorithm_results_df.dropna(subset=['ROC_AUC'])
        if not roc_data.empty:
            sns.boxplot(data=roc_data, x='Algorithm', y='ROC_AUC', ax=ax6)
            ax6.set_title('ROC AUC Distribution by Algorithm')
            ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, 'No ROC AUC data available', ha='center', va='center',
                    transform=ax6.transAxes)
            ax6.set_title('ROC AUC Distribution')

        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def compare_data_splits(self):
        """4. Data split scenario comparison"""
        print("\n" + "="*80)
        print("4. DATA SPLIT SCENARIO COMPARISON")
        print("="*80)

        # Test different split scenarios
        split_scenarios = [0.25, 0.30, 0.35, 0.65, 0.70, 0.75]

        # Use best performing algorithms from previous analysis
        best_algorithms = {
            'SVM_RBF': SVC(kernel='rbf', random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        split_results = []

        for sentiment_method in self.sentiment_columns:
            y = self.encoded_labels[sentiment_method]

            for feature_name, X in self.feature_data.items():
                # Handle negative values for algorithms that need positive values
                if feature_name == 'Word2Vec':
                    X_processed = X - X.min() + 1
                else:
                    X_processed = X

                for train_size in split_scenarios:
                    test_size = 1 - train_size

                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_processed, y, train_size=train_size, random_state=42, stratify=y
                        )

                        for algo_name, algorithm in best_algorithms.items():
                            try:
                                algorithm.fit(X_train, y_train)
                                y_pred = algorithm.predict(X_test)

                                accuracy = accuracy_score(y_test, y_pred)
                                precision, recall, f1, _ = precision_recall_fscore_support(
                                    y_test, y_pred, average='weighted'
                                )

                                split_results.append({
                                    'Sentiment_Method': sentiment_method.replace('sentiment_', ''),
                                    'Feature_Type': feature_name,
                                    'Algorithm': algo_name,
                                    'Train_Size': train_size,
                                    'Test_Size': test_size,
                                    'Train_Samples': len(X_train),
                                    'Test_Samples': len(X_test),
                                    'Accuracy': accuracy,
                                    'Precision': precision,
                                    'Recall': recall,
                                    'F1_Score': f1
                                })

                            except Exception as e:
                                print(f"Error with {algo_name}: {e}")
                                continue

                    except Exception as e:
                        print(f"Error with split {train_size}: {e}")
                        continue

        split_results_df = pd.DataFrame(split_results)

        # Analysis by split size
        print("\nPerformance by Training Size:")
        split_summary = split_results_df.groupby('Train_Size').agg({
            'Accuracy': ['mean', 'std', 'min', 'max'],
            'F1_Score': ['mean', 'std', 'min', 'max']
        }).round(4)
        print(split_summary)

        # Optimal split analysis
        print("\nOptimal Split Analysis:")
        optimal_splits = split_results_df.groupby(['Algorithm', 'Feature_Type']).apply(
            lambda x: x.loc[x['Accuracy'].idxmax()]
        ).reset_index(drop=True)

        print("Best performing splits by Algorithm and Feature Type:")
        print(optimal_splits[['Algorithm', 'Feature_Type', 'Train_Size', 'Accuracy', 'F1_Score']].to_string(index=False))

        # Statistical analysis of split impact
        print("\nStatistical Analysis of Split Impact:")
        for algo in split_results_df['Algorithm'].unique():
            algo_data = split_results_df[split_results_df['Algorithm'] == algo]
            correlation = algo_data['Train_Size'].corr(algo_data['Accuracy'])
            print(f"{algo}: Correlation between train size and accuracy = {correlation:.4f}")

        return split_results_df

    def visualize_data_splits(self, split_results_df):
        """Create visualizations for data split comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Split Scenario Analysis', fontsize=16)

        # 1. Accuracy vs Training Size
        ax1 = axes[0, 0]
        sns.lineplot(data=split_results_df, x='Train_Size', y='Accuracy',
                    hue='Algorithm', marker='o', ax=ax1)
        ax1.set_title('Accuracy vs Training Size')
        ax1.set_xlabel('Training Size Proportion')
        ax1.set_ylabel('Accuracy')

        # 2. F1-Score vs Training Size
        ax2 = axes[0, 1]
        sns.lineplot(data=split_results_df, x='Train_Size', y='F1_Score',
                    hue='Algorithm', marker='o', ax=ax2)
        ax2.set_title('F1-Score vs Training Size')
        ax2.set_xlabel('Training Size Proportion')
        ax2.set_ylabel('F1-Score')

        # 3. Performance by Feature Type
        ax3 = axes[1, 0]
        sns.boxplot(data=split_results_df, x='Train_Size', y='Accuracy',
                   hue='Feature_Type', ax=ax3)
        ax3.set_title('Accuracy Distribution by Training Size and Feature Type')
        ax3.set_xlabel('Training Size Proportion')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Heatmap of performance
        ax4 = axes[1, 1]
        heatmap_data = split_results_df.groupby(['Algorithm', 'Train_Size'])['Accuracy'].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=ax4, fmt='.3f')
        ax4.set_title('Accuracy Heatmap: Algorithm vs Training Size')
        ax4.set_xlabel('Training Size Proportion')

        plt.tight_layout()
        plt.savefig('data_split_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_comprehensive_report(self, sentiment_stats, comparison_df, agreement_matrix,
                                    feature_results_df, algorithm_results_df, split_results_df):
        """Generate comprehensive statistical report"""
        print("\n" + "="*100)
        print("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        print("="*100)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Generated on: {timestamp}")
        print(f"Dataset: {self.data_path}")
        print(f"Total samples: {len(self.df)}")

        # 1. Sentiment Label Analysis Summary
        print("\n" + "-"*80)
        print("1. SENTIMENT LABEL ANALYSIS SUMMARY")
        print("-"*80)

        print("Label Distribution Summary:")
        for col in self.sentiment_columns:
            method_name = col.replace('sentiment_', '').replace('_', ' ').title()
            stats = sentiment_stats[col]
            print(f"\n{method_name}:")
            print(f"  - Unique labels: {stats['unique_labels']}")
            print(f"  - Most common: {max(stats['distribution'], key=stats['distribution'].get)} "
                  f"({max(stats['percentage'].values()):.1f}%)")
            print(f"  - Least common: {min(stats['distribution'], key=stats['distribution'].get)} "
                  f"({min(stats['percentage'].values()):.1f}%)")

        print(f"\nHighest Agreement: {agreement_matrix.values[np.triu_indices_from(agreement_matrix.values, k=1)].max():.3f}")
        print(f"Lowest Agreement: {agreement_matrix.values[np.triu_indices_from(agreement_matrix.values, k=1)].min():.3f}")

        # 2. Feature Extraction Summary
        print("\n" + "-"*80)
        print("2. FEATURE EXTRACTION ANALYSIS SUMMARY")
        print("-"*80)

        feature_summary = feature_results_df.groupby('Feature_Type').agg({
            'Accuracy': ['mean', 'std', 'max'],
            'F1_Score': ['mean', 'std', 'max']
        }).round(4)

        print("Feature Type Performance:")
        print(feature_summary)

        best_feature = feature_results_df.loc[feature_results_df['Accuracy'].idxmax()]
        print(f"\nBest Feature Configuration:")
        print(f"  - Feature Type: {best_feature['Feature_Type']}")
        print(f"  - Algorithm: {best_feature['Algorithm']}")
        print(f"  - Accuracy: {best_feature['Accuracy']:.4f}")
        print(f"  - F1-Score: {best_feature['F1_Score']:.4f}")

        # 3. Algorithm Comparison Summary
        print("\n" + "-"*80)
        print("3. ML ALGORITHM ANALYSIS SUMMARY")
        print("-"*80)

        algo_summary = algorithm_results_df.groupby('Algorithm').agg({
            'Accuracy': ['mean', 'std', 'max'],
            'Training_Time': ['mean', 'std']
        }).round(4)

        print("Algorithm Performance Summary:")
        print(algo_summary)

        best_algorithm = algorithm_results_df.loc[algorithm_results_df['Accuracy'].idxmax()]
        print(f"\nBest Algorithm Configuration:")
        print(f"  - Algorithm: {best_algorithm['Algorithm']}")
        print(f"  - Feature Type: {best_algorithm['Feature_Type']}")
        print(f"  - Sentiment Method: {best_algorithm['Sentiment_Method']}")
        print(f"  - Accuracy: {best_algorithm['Accuracy']:.4f}")
        print(f"  - Training Time: {best_algorithm['Training_Time']:.4f}s")

        # 4. Data Split Analysis Summary
        print("\n" + "-"*80)
        print("4. DATA SPLIT ANALYSIS SUMMARY")
        print("-"*80)

        split_summary = split_results_df.groupby('Train_Size')['Accuracy'].agg(['mean', 'std', 'max']).round(4)
        print("Performance by Training Size:")
        print(split_summary)

        optimal_split = split_results_df.loc[split_results_df['Accuracy'].idxmax()]
        print(f"\nOptimal Split Configuration:")
        print(f"  - Training Size: {optimal_split['Train_Size']*100:.0f}%")
        print(f"  - Algorithm: {optimal_split['Algorithm']}")
        print(f"  - Feature Type: {optimal_split['Feature_Type']}")
        print(f"  - Accuracy: {optimal_split['Accuracy']:.4f}")

        # 5. Overall Recommendations
        print("\n" + "-"*80)
        print("5. RECOMMENDATIONS")
        print("-"*80)

        print("Based on comprehensive analysis:")
        print(f"1. Best sentiment labeling method: {best_algorithm['Sentiment_Method']}")
        print(f"2. Recommended feature extraction: {best_feature['Feature_Type']}")
        print(f"3. Optimal ML algorithm: {best_algorithm['Algorithm']}")
        print(f"4. Recommended training split: {optimal_split['Train_Size']*100:.0f}%")
        print(f"5. Expected accuracy: {optimal_split['Accuracy']:.4f}")

        # Save detailed results
        timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save all results to CSV files
        comparison_df.to_csv(f'sentiment_comparison_{timestamp_file}.csv', index=False)
        feature_results_df.to_csv(f'feature_comparison_{timestamp_file}.csv', index=False)
        algorithm_results_df.to_csv(f'algorithm_comparison_{timestamp_file}.csv', index=False)
        split_results_df.to_csv(f'split_comparison_{timestamp_file}.csv', index=False)

        print(f"\nDetailed results saved to CSV files with timestamp: {timestamp_file}")

    def run_comprehensive_analysis(self):
        """Run the complete comprehensive analysis"""
        print("Starting Comprehensive Statistical Analysis...")
        print("="*100)

        # Load data and extract features
        self.load_data()
        self.extract_features()

        # Run all analyses
        sentiment_stats, comparison_df, agreement_matrix = self.analyze_sentiment_labels()
        self.visualize_sentiment_comparison(comparison_df, agreement_matrix)

        feature_results_df = self.compare_feature_extraction()
        self.visualize_feature_comparison(feature_results_df)

        algorithm_results_df = self.compare_ml_algorithms()
        self.visualize_algorithm_comparison(algorithm_results_df)

        split_results_df = self.compare_data_splits()
        self.visualize_data_splits(split_results_df)

        # Generate comprehensive report
        self.generate_comprehensive_report(
            sentiment_stats, comparison_df, agreement_matrix,
            feature_results_df, algorithm_results_df, split_results_df
        )

        print("\n" + "="*100)
        print("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*100)

def main():
    """Main execution function"""
    data_path = 'google_play_reviews_DigitalBank_sentiment_analysis.csv'

    analyzer = ComprehensiveAnalyzer(data_path)
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
