#!/usr/bin/env python3
"""
Comprehensive Comparison Analysis for Aspect-Based Sentiment Analysis
Generates detailed statistics, tables, and visualizations for:
1. TF-IDF vs Word2Vec feature extraction comparison
2. Multi-class SVM kernel comparison
3. Data split scenarios comparison (65%, 70%, 75% vs 25%, 30%, 35%)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
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
        self.data_path = data_path
        self.df = None
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
        texts = self.df['stemmed_text'].fillna('').astype(str)
        
        # Extract features
        self._extract_features(texts)
        
        # Prepare labels
        self._prepare_labels()
        
    def _extract_features(self, texts):
        """Extract TF-IDF and Word2Vec features"""
        print("\nExtracting features...")
        
        # TF-IDF
        print("Extracting TF-IDF features...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
        
        # Word2Vec
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
        
        def get_document_vector(tokens, model, vector_size=100):
            vectors = []
            for token in tokens:
                if token in model.wv.key_to_index:
                    vectors.append(model.wv[token])
            return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
        
        w2v_features = np.array([get_document_vector(tokens, w2v_model) for tokens in tokenized_texts])
        
        self.feature_data = {
            'TF-IDF': tfidf_features,
            'Word2Vec': w2v_features
        }
        
        print(f"TF-IDF shape: {tfidf_features.shape}")
        print(f"Word2Vec shape: {w2v_features.shape}")
        
    def _prepare_labels(self):
        """Prepare sentiment labels"""
        sentiment_methods = ['sentiment_score_based', 'sentiment_textblob', 
                           'sentiment_vader', 'sentiment_ensemble']
        
        for method in sentiment_methods:
            le = LabelEncoder()
            self.encoded_labels[method] = le.fit_transform(self.df[method])
            self.label_encoders[method] = le
            
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis with all combinations"""
        print("\nRunning comprehensive analysis...")
        
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        train_sizes = [0.65, 0.70, 0.75]  # Training percentages
        test_sizes = [0.35, 0.30, 0.25]   # Corresponding test percentages
        sentiment_methods = list(self.encoded_labels.keys())
        
        total_experiments = len(sentiment_methods) * len(self.feature_data) * len(kernels) * len(train_sizes)
        current_experiment = 0
        
        for method in sentiment_methods:
            y = self.encoded_labels[method]
            
            for feature_name, X in self.feature_data.items():
                for i, train_size in enumerate(train_sizes):
                    test_size = test_sizes[i]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, train_size=train_size, random_state=42, stratify=y
                    )
                    
                    for kernel in kernels:
                        current_experiment += 1
                        print(f"Experiment {current_experiment}/{total_experiments}: "
                              f"{method} | {feature_name} | {kernel} | "
                              f"Train: {train_size*100:.0f}% | Test: {test_size*100:.0f}%")
                        
                        try:
                            result = self._evaluate_model(
                                X_train, X_test, y_train, y_test, 
                                kernel, method, feature_name, train_size, test_size
                            )
                            self.results.append(result)
                        except Exception as e:
                            print(f"Error: {e}")
                            continue
        
        print(f"\nCompleted {len(self.results)} experiments!")
        
    def _evaluate_model(self, X_train, X_test, y_train, y_test, kernel, 
                       method, feature_name, train_size, test_size):
        """Evaluate a single model configuration"""
        
        # Train SVM
        svm = SVC(kernel=kernel, probability=True, random_state=42)
        svm.fit(X_train, y_train)
        
        # Predictions
        y_pred = svm.predict(X_test)
        y_pred_proba = svm.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # ROC AUC
        try:
            if len(np.unique(y_test)) > 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        except:
            roc_auc = None
            
        return {
            'sentiment_method': method,
            'feature_type': feature_name,
            'kernel': kernel,
            'train_size': train_size,
            'test_size': test_size,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
    def generate_comparison_tables(self):
        """Generate comprehensive comparison tables"""
        print("\nGenerating comparison tables...")
        
        df_results = pd.DataFrame(self.results)
        
        # 1. Feature Extraction Comparison
        print("\n" + "="*80)
        print("1. FEATURE EXTRACTION COMPARISON (TF-IDF vs Word2Vec)")
        print("="*80)
        
        feature_comparison = df_results.groupby('feature_type').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'roc_auc': ['mean', 'std']
        }).round(4)
        
        print(feature_comparison)
        
        # Statistical significance test
        tfidf_acc = df_results[df_results['feature_type'] == 'TF-IDF']['accuracy']
        w2v_acc = df_results[df_results['feature_type'] == 'Word2Vec']['accuracy']
        t_stat, p_value = stats.ttest_ind(tfidf_acc, w2v_acc)
        print(f"\nStatistical Test (t-test):")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # 2. Kernel Comparison
        print("\n" + "="*80)
        print("2. SVM KERNEL COMPARISON")
        print("="*80)
        
        kernel_comparison = df_results.groupby('kernel').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'roc_auc': ['mean', 'std']
        }).round(4)
        
        print(kernel_comparison)
        
        # 3. Data Split Comparison
        print("\n" + "="*80)
        print("3. DATA SPLIT SCENARIOS COMPARISON")
        print("="*80)
        
        split_comparison = df_results.groupby(['train_size', 'test_size']).agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'roc_auc': ['mean', 'std']
        }).round(4)
        
        print(split_comparison)
        
        return df_results, feature_comparison, kernel_comparison, split_comparison
        
    def create_visualizations(self, df_results):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (20, 15)
        
        # 1. Feature Extraction Comparison
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Feature Extraction Comparison: TF-IDF vs Word2Vec', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for i, metric in enumerate(metrics):
            if i < 5:
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                # Box plot
                sns.boxplot(data=df_results, x='feature_type', y=metric, ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Feature Type')
                ax.set_ylabel(metric.replace("_", " ").title())
                
                # Add mean values as text
                means = df_results.groupby('feature_type')[metric].mean()
                for j, (feature, mean_val) in enumerate(means.items()):
                    if not np.isnan(mean_val):
                        ax.text(j, mean_val, f'{mean_val:.3f}', 
                               ha='center', va='bottom', fontweight='bold')
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig('feature_extraction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Kernel Comparison
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('SVM Kernel Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            if i < 5:
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                sns.boxplot(data=df_results, x='kernel', y=metric, ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Kernel Type')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)
        
        axes[1, 2].remove()
        plt.tight_layout()
        plt.savefig('kernel_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Data Split Comparison
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Data Split Scenarios Comparison', fontsize=16, fontweight='bold')

        # Create train/test split labels
        df_results['split_label'] = df_results.apply(
            lambda x: f"{int(x['train_size']*100)}%/{int(x['test_size']*100)}%", axis=1
        )

        for i, metric in enumerate(metrics):
            if i < 5:
                row, col = i // 3, i % 3
                ax = axes[row, col]

                sns.boxplot(data=df_results, x='split_label', y=metric, ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Train/Test Split')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)

        axes[1, 2].remove()
        plt.tight_layout()
        plt.savefig('data_split_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Heatmap Comparison
        self._create_heatmap_comparisons(df_results)

        # 5. Performance Distribution
        self._create_performance_distributions(df_results)

        return True

    def _create_heatmap_comparisons(self, df_results):
        """Create heatmap visualizations for detailed comparisons"""

        # 1. Feature vs Kernel Heatmap
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Performance Heatmaps: Feature Type vs Kernel', fontsize=16, fontweight='bold')

        metrics = ['accuracy', 'precision', 'f1_score']

        for i, metric in enumerate(metrics):
            pivot_table = df_results.pivot_table(
                values=metric,
                index='feature_type',
                columns='kernel',
                aggfunc='mean'
            )

            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd',
                       ax=axes[i], cbar_kws={'label': metric.replace('_', ' ').title()})
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Kernel')
            axes[i].set_ylabel('Feature Type')

        plt.tight_layout()
        plt.savefig('heatmap_feature_kernel.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Split vs Kernel Heatmap
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Performance Heatmaps: Data Split vs Kernel', fontsize=16, fontweight='bold')

        for i, metric in enumerate(metrics):
            pivot_table = df_results.pivot_table(
                values=metric,
                index='split_label',
                columns='kernel',
                aggfunc='mean'
            )

            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd',
                       ax=axes[i], cbar_kws={'label': metric.replace('_', ' ').title()})
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Kernel')
            axes[i].set_ylabel('Train/Test Split')

        plt.tight_layout()
        plt.savefig('heatmap_split_kernel.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _create_performance_distributions(self, df_results):
        """Create performance distribution plots"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Distributions', fontsize=16, fontweight='bold')

        # 1. Accuracy distribution by feature type
        ax1 = axes[0, 0]
        for feature in df_results['feature_type'].unique():
            data = df_results[df_results['feature_type'] == feature]['accuracy']
            ax1.hist(data, alpha=0.7, label=feature, bins=20)
        ax1.set_title('Accuracy Distribution by Feature Type')
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # 2. Accuracy distribution by kernel
        ax2 = axes[0, 1]
        for kernel in df_results['kernel'].unique():
            data = df_results[df_results['kernel'] == kernel]['accuracy']
            ax2.hist(data, alpha=0.7, label=kernel, bins=20)
        ax2.set_title('Accuracy Distribution by Kernel')
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # 3. F1-score distribution by feature type
        ax3 = axes[1, 0]
        for feature in df_results['feature_type'].unique():
            data = df_results[df_results['feature_type'] == feature]['f1_score']
            ax3.hist(data, alpha=0.7, label=feature, bins=20)
        ax3.set_title('F1-Score Distribution by Feature Type')
        ax3.set_xlabel('F1-Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()

        # 4. Performance by split scenario
        ax4 = axes[1, 1]
        split_means = df_results.groupby('split_label')['accuracy'].mean().sort_values(ascending=False)
        bars = ax4.bar(range(len(split_means)), split_means.values)
        ax4.set_title('Mean Accuracy by Data Split')
        ax4.set_xlabel('Train/Test Split')
        ax4.set_ylabel('Mean Accuracy')
        ax4.set_xticks(range(len(split_means)))
        ax4.set_xticklabels(split_means.index, rotation=45)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('performance_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_detailed_statistics(self, df_results):
        """Generate detailed statistical analysis"""
        print("\n" + "="*80)
        print("DETAILED STATISTICAL ANALYSIS")
        print("="*80)

        # 1. Overall Statistics
        print("\n1. OVERALL PERFORMANCE STATISTICS")
        print("-" * 50)
        overall_stats = df_results[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].describe()
        print(overall_stats.round(4))

        # 2. Best Performing Configurations
        print("\n2. TOP 10 BEST PERFORMING CONFIGURATIONS")
        print("-" * 50)
        top_configs = df_results.nlargest(10, 'accuracy')[
            ['sentiment_method', 'feature_type', 'kernel', 'split_label', 'accuracy', 'f1_score']
        ]
        print(top_configs.to_string(index=False))

        # 3. Feature Type Analysis
        print("\n3. FEATURE TYPE DETAILED ANALYSIS")
        print("-" * 50)
        feature_stats = df_results.groupby('feature_type').agg({
            'accuracy': ['count', 'mean', 'std', 'min', 'max'],
            'f1_score': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std']
        }).round(4)
        print(feature_stats)

        # 4. Kernel Analysis
        print("\n4. KERNEL DETAILED ANALYSIS")
        print("-" * 50)
        kernel_stats = df_results.groupby('kernel').agg({
            'accuracy': ['count', 'mean', 'std', 'min', 'max'],
            'f1_score': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std']
        }).round(4)
        print(kernel_stats)

        # 5. Data Split Analysis
        print("\n5. DATA SPLIT DETAILED ANALYSIS")
        print("-" * 50)
        split_stats = df_results.groupby('split_label').agg({
            'accuracy': ['count', 'mean', 'std', 'min', 'max'],
            'f1_score': ['mean', 'std'],
            'train_samples': ['mean'],
            'test_samples': ['mean']
        }).round(4)
        print(split_stats)

        # 6. Correlation Analysis
        print("\n6. CORRELATION ANALYSIS")
        print("-" * 50)
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'train_size', 'test_size']
        correlation_matrix = df_results[numeric_cols].corr()
        print(correlation_matrix.round(4))

        return overall_stats, top_configs, feature_stats, kernel_stats, split_stats

    def save_results(self, df_results, feature_comparison, kernel_comparison, split_comparison):
        """Save all results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save main results
        df_results.to_csv(f'comprehensive_results_{timestamp}.csv', index=False)

        # Save comparison tables
        feature_comparison.to_csv(f'feature_comparison_{timestamp}.csv')
        kernel_comparison.to_csv(f'kernel_comparison_{timestamp}.csv')
        split_comparison.to_csv(f'split_comparison_{timestamp}.csv')

        print(f"\nResults saved with timestamp: {timestamp}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*80)
        print("COMPREHENSIVE COMPARISON ANALYSIS")
        print("="*80)

        # Load data and run analysis
        self.load_data()
        self.run_comprehensive_analysis()

        # Generate comparisons and visualizations
        df_results, feature_comp, kernel_comp, split_comp = self.generate_comparison_tables()
        self.create_visualizations(df_results)

        # Generate detailed statistics
        self.generate_detailed_statistics(df_results)

        # Save results
        self.save_results(df_results, feature_comp, kernel_comp, split_comp)

        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETED!")
        print("="*80)

def main():
    """Main execution function"""
    analyzer = ComprehensiveAnalyzer('google_play_reviews_DigitalBank_sentiment_analysis.csv')
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
