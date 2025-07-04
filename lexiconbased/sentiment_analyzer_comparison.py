"""
Sentiment Analysis Comparison: Indonesian Naive Bayes Analyzer vs VADER Indonesia
This module provides comprehensive testing and comparison between the two sentiment analysis methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import time
import warnings
warnings.filterwarnings('ignore')

# Import our analyzers
from indonesian_naive_bayes_analyzer import IndonesianNaiveBayesAnalyzer
import sys
import os
sys.path.append('..')  # Add parent directory to path
from sentiment_analysis_4_methods import VaderIndonesia

class SentimentAnalyzerComparison:
    """
    Comprehensive comparison framework for Indonesian sentiment analyzers
    """
    
    def __init__(self):
        self.nb_analyzer = None
        self.vader_analyzer = None
        self.test_results = {}
        
    def initialize_analyzers(self):
        """Initialize both analyzers"""
        print("Initializing analyzers...")
        
        # Initialize Naive Bayes Analyzer
        print("1. Loading Indonesian Naive Bayes Analyzer...")
        self.nb_analyzer = IndonesianNaiveBayesAnalyzer()
        
        # Initialize VADER Indonesia
        print("2. Loading VADER Indonesia...")
        self.vader_analyzer = VaderIndonesia()
        
        print("✓ Both analyzers initialized successfully!")
        
    def create_comprehensive_test_dataset(self):
        """Create a comprehensive test dataset with various Indonesian text types"""
        
        # Movie reviews
        movie_data = [
            ("Film ini sangat bagus dan menghibur sekali", "positive"),
            ("Ceritanya luar biasa hebat dan menyentuh hati", "positive"),
            ("Aktingnya mantap dan sinematografinya indah", "positive"),
            ("Saya suka banget dengan jalan ceritanya yang unik", "positive"),
            ("Film yang sangat memuaskan dan tidak mengecewakan", "positive"),
            
            ("Film ini buruk sekali dan sangat membosankan", "negative"),
            ("Saya tidak suka dengan ceritanya yang jelek", "negative"),
            ("Aktingnya parah dan sangat mengecewakan", "negative"),
            ("Film yang sangat buruk dan membuang waktu", "negative"),
            ("Sangat tidak memuaskan dan menjengkelkan", "negative"),
            
            ("Film ini biasa saja, tidak terlalu istimewa", "neutral"),
            ("Lumayan untuk ditonton, tidak bagus tidak jelek", "neutral"),
            ("Film standar dengan cerita yang cukup", "neutral"),
            ("Tidak ada yang menarik tapi juga tidak buruk", "neutral"),
            ("Film yang biasa, tidak terlalu berkesan", "neutral")
        ]
        
        # Product reviews
        product_data = [
            ("Produk ini sangat berkualitas dan awet", "positive"),
            ("Pelayanannya hebat dan pengiriman cepat", "positive"),
            ("Sangat puas dengan pembelian ini, recommended!", "positive"),
            ("Kualitas bagus sesuai dengan harga yang dibayar", "positive"),
            ("Produk mantap, penjual responsif dan ramah", "positive"),
            
            ("Produk jelek, tidak sesuai dengan deskripsi", "negative"),
            ("Pelayanan buruk dan pengiriman sangat lambat", "negative"),
            ("Sangat kecewa dengan kualitas produknya", "negative"),
            ("Produk rusak dan penjual tidak responsif", "negative"),
            ("Tidak recommended, buang-buang uang saja", "negative"),
            
            ("Produk standar, sesuai dengan harganya", "neutral"),
            ("Biasa saja, tidak ada yang istimewa", "neutral"),
            ("Kualitas cukup untuk penggunaan sehari-hari", "neutral"),
            ("Lumayan, tidak terlalu bagus tidak terlalu jelek", "neutral"),
            ("Produk biasa dengan kualitas yang cukup", "neutral")
        ]
        
        # Social media posts
        social_data = [
            ("Hari ini sangat menyenangkan, banyak hal baik terjadi!", "positive"),
            ("Alhamdulillah rezeki hari ini berlimpah", "positive"),
            ("Senang banget bisa ketemu teman lama", "positive"),
            ("Cuaca cerah, hati senang, semangat bekerja!", "positive"),
            ("Terima kasih untuk semua dukungannya, sangat berarti", "positive"),
            
            ("Hari yang sangat buruk, semua tidak berjalan lancar", "negative"),
            ("Kesal banget dengan pelayanan yang jelek ini", "negative"),
            ("Sedih dan kecewa dengan kejadian hari ini", "negative"),
            ("Capek dengan semua masalah yang tidak ada habisnya", "negative"),
            ("Marah dengan sikap orang yang tidak bertanggung jawab", "negative"),
            
            ("Hari biasa seperti hari-hari lainnya", "neutral"),
            ("Cuaca mendung, aktivitas seperti biasa", "neutral"),
            ("Tidak ada yang spesial hari ini", "neutral"),
            ("Rutinitas harian berjalan seperti biasa", "neutral"),
            ("Hari yang cukup normal tanpa kejadian khusus", "neutral")
        ]
        
        # Combine all data
        all_data = movie_data + product_data + social_data
        texts, labels = zip(*all_data)
        
        return list(texts), list(labels)
        
    def create_challenging_test_cases(self):
        """Create challenging test cases with mixed sentiments, sarcasm, etc."""
        
        challenging_cases = [
            # Mixed sentiments
            ("Film ini bagus tapi agak membosankan di bagian tengah", "neutral"),
            ("Suka dengan aktingnya tapi tidak suka dengan ceritanya", "neutral"),
            ("Produk berkualitas tapi harganya terlalu mahal", "neutral"),
            
            # Sarcasm and irony
            ("Wah hebat sekali pelayanannya, sampai 2 jam menunggu", "negative"),
            ("Bagus banget nih produk, langsung rusak setelah dibuka", "negative"),
            ("Mantap filmnya, bikin ngantuk dari awal sampai akhir", "negative"),
            
            # Negation handling
            ("Film ini tidak buruk, malah cukup bagus", "positive"),
            ("Saya tidak kecewa dengan pembelian ini", "positive"),
            ("Tidak ada yang jelek dari produk ini", "positive"),
            
            # Intensifiers
            ("Sangat sangat bagus sekali!", "positive"),
            ("Buruk banget parah sekali", "negative"),
            ("Biasa banget, tidak istimewa", "neutral"),
            
            # Colloquial Indonesian
            ("Keren abis filmnya, gak nyesel nonton", "positive"),
            ("Jelek banget dah, gak worth it", "negative"),
            ("Ya gitu deh, biasa aja", "neutral")
        ]
        
        texts, labels = zip(*challenging_cases)
        return list(texts), list(labels)
        
    def train_naive_bayes(self, train_texts, train_labels):
        """Train the Naive Bayes analyzer"""
        print("Training Naive Bayes analyzer...")
        start_time = time.time()
        
        results = self.nb_analyzer.train(train_texts, train_labels, test_size=0.2)
        
        training_time = time.time() - start_time
        print(f"✓ Naive Bayes training completed in {training_time:.2f} seconds")
        
        return results, training_time
        
    def predict_naive_bayes(self, texts):
        """Get predictions from Naive Bayes analyzer"""
        predictions = []
        confidences = []
        
        for text in texts:
            result = self.nb_analyzer.predict_sentiment(text)
            predictions.append(result['sentiment'])
            confidences.append(result['confidence'])
            
        return predictions, confidences
        
    def predict_vader(self, texts):
        """Get predictions from VADER Indonesia"""
        predictions = []
        confidences = []
        
        for text in texts:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Convert compound score to sentiment
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            # Use absolute compound score as confidence
            confidence = abs(compound)
            
            predictions.append(sentiment)
            confidences.append(confidence)
            
        return predictions, confidences
        
    def evaluate_performance(self, true_labels, predictions, method_name):
        """Evaluate performance metrics"""
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Class-wise metrics
        class_report = classification_report(true_labels, predictions, output_dict=True)
        
        results = {
            'method': method_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report
        }
        
        return results
        
    def run_speed_test(self, texts, num_iterations=3):
        """Test prediction speed for both methods"""
        print("Running speed tests...")
        
        # Test Naive Bayes speed
        nb_times = []
        for i in range(num_iterations):
            start_time = time.time()
            self.predict_naive_bayes(texts)
            nb_times.append(time.time() - start_time)
            
        # Test VADER speed
        vader_times = []
        for i in range(num_iterations):
            start_time = time.time()
            self.predict_vader(texts)
            vader_times.append(time.time() - start_time)
            
        return {
            'naive_bayes_avg_time': np.mean(nb_times),
            'vader_avg_time': np.mean(vader_times),
            'naive_bayes_std': np.std(nb_times),
            'vader_std': np.std(vader_times)
        }
        
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison between both methods"""
        print("=" * 60)
        print("COMPREHENSIVE SENTIMENT ANALYZER COMPARISON")
        print("Indonesian Naive Bayes vs VADER Indonesia")
        print("=" * 60)
        
        # Initialize analyzers
        self.initialize_analyzers()
        
        # Create datasets
        print("\nCreating test datasets...")
        train_texts, train_labels = self.create_comprehensive_test_dataset()
        challenge_texts, challenge_labels = self.create_challenging_test_cases()
        
        print(f"✓ Training dataset: {len(train_texts)} samples")
        print(f"✓ Challenge dataset: {len(challenge_texts)} samples")
        
        # Train Naive Bayes
        print("\n" + "="*40)
        print("TRAINING PHASE")
        print("="*40)
        nb_training_results, nb_training_time = self.train_naive_bayes(train_texts, train_labels)
        
        # Test on training data
        print("\n" + "="*40)
        print("EVALUATION ON TRAINING DATA")
        print("="*40)
        
        # Naive Bayes predictions
        print("Getting Naive Bayes predictions...")
        nb_pred_train, nb_conf_train = self.predict_naive_bayes(train_texts)
        nb_results_train = self.evaluate_performance(train_labels, nb_pred_train, "Naive Bayes")
        
        # VADER predictions
        print("Getting VADER Indonesia predictions...")
        vader_pred_train, vader_conf_train = self.predict_vader(train_texts)
        vader_results_train = self.evaluate_performance(train_labels, vader_pred_train, "VADER Indonesia")
        
        # Test on challenging cases
        print("\n" + "="*40)
        print("EVALUATION ON CHALLENGING CASES")
        print("="*40)
        
        # Naive Bayes on challenging cases
        nb_pred_challenge, nb_conf_challenge = self.predict_naive_bayes(challenge_texts)
        nb_results_challenge = self.evaluate_performance(challenge_labels, nb_pred_challenge, "Naive Bayes")
        
        # VADER on challenging cases
        vader_pred_challenge, vader_conf_challenge = self.predict_vader(challenge_texts)
        vader_results_challenge = self.evaluate_performance(challenge_labels, vader_pred_challenge, "VADER Indonesia")
        
        # Speed test
        print("\n" + "="*40)
        print("SPEED COMPARISON")
        print("="*40)
        speed_results = self.run_speed_test(train_texts)
        
        # Store all results
        self.test_results = {
            'training_data': {
                'naive_bayes': nb_results_train,
                'vader': vader_results_train,
                'nb_predictions': nb_pred_train,
                'vader_predictions': vader_pred_train,
                'nb_confidences': nb_conf_train,
                'vader_confidences': vader_conf_train,
                'true_labels': train_labels,
                'texts': train_texts
            },
            'challenging_data': {
                'naive_bayes': nb_results_challenge,
                'vader': vader_results_challenge,
                'nb_predictions': nb_pred_challenge,
                'vader_predictions': vader_pred_challenge,
                'nb_confidences': nb_conf_challenge,
                'vader_confidences': vader_conf_challenge,
                'true_labels': challenge_labels,
                'texts': challenge_texts
            },
            'speed': speed_results,
            'training_time': nb_training_time
        }
        
        return self.test_results

    def print_comparison_summary(self):
        """Print detailed comparison summary"""
        if not self.test_results:
            print("No test results available. Run comparison first.")
            return

        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)

        # Training data results
        print("\n📊 PERFORMANCE ON TRAINING DATA:")
        print("-" * 40)
        nb_train = self.test_results['training_data']['naive_bayes']
        vader_train = self.test_results['training_data']['vader']

        print(f"{'Metric':<15} {'Naive Bayes':<15} {'VADER Indonesia':<15} {'Winner':<10}")
        print("-" * 60)
        print(f"{'Accuracy':<15} {nb_train['accuracy']:<15.4f} {vader_train['accuracy']:<15.4f} {'NB' if nb_train['accuracy'] > vader_train['accuracy'] else 'VADER':<10}")
        print(f"{'Precision':<15} {nb_train['precision']:<15.4f} {vader_train['precision']:<15.4f} {'NB' if nb_train['precision'] > vader_train['precision'] else 'VADER':<10}")
        print(f"{'Recall':<15} {nb_train['recall']:<15.4f} {vader_train['recall']:<15.4f} {'NB' if nb_train['recall'] > vader_train['recall'] else 'VADER':<10}")
        print(f"{'F1-Score':<15} {nb_train['f1_score']:<15.4f} {vader_train['f1_score']:<15.4f} {'NB' if nb_train['f1_score'] > vader_train['f1_score'] else 'VADER':<10}")

        # Challenging cases results
        print("\n🎯 PERFORMANCE ON CHALLENGING CASES:")
        print("-" * 40)
        nb_challenge = self.test_results['challenging_data']['naive_bayes']
        vader_challenge = self.test_results['challenging_data']['vader']

        print(f"{'Metric':<15} {'Naive Bayes':<15} {'VADER Indonesia':<15} {'Winner':<10}")
        print("-" * 60)
        print(f"{'Accuracy':<15} {nb_challenge['accuracy']:<15.4f} {vader_challenge['accuracy']:<15.4f} {'NB' if nb_challenge['accuracy'] > vader_challenge['accuracy'] else 'VADER':<10}")
        print(f"{'Precision':<15} {nb_challenge['precision']:<15.4f} {vader_challenge['precision']:<15.4f} {'NB' if nb_challenge['precision'] > vader_challenge['precision'] else 'VADER':<10}")
        print(f"{'Recall':<15} {nb_challenge['recall']:<15.4f} {vader_challenge['recall']:<15.4f} {'NB' if nb_challenge['recall'] > vader_challenge['recall'] else 'VADER':<10}")
        print(f"{'F1-Score':<15} {nb_challenge['f1_score']:<15.4f} {vader_challenge['f1_score']:<15.4f} {'NB' if nb_challenge['f1_score'] > vader_challenge['f1_score'] else 'VADER':<10}")

        # Speed comparison
        print("\n⚡ SPEED COMPARISON:")
        print("-" * 40)
        speed = self.test_results['speed']
        print(f"Training Time (NB): {self.test_results['training_time']:.2f} seconds")
        print(f"Prediction Time (NB): {speed['naive_bayes_avg_time']:.4f} ± {speed['naive_bayes_std']:.4f} seconds")
        print(f"Prediction Time (VADER): {speed['vader_avg_time']:.4f} ± {speed['vader_std']:.4f} seconds")
        print(f"Speed Winner: {'VADER' if speed['vader_avg_time'] < speed['naive_bayes_avg_time'] else 'Naive Bayes'}")

        # Overall winner
        print("\n🏆 OVERALL ASSESSMENT:")
        print("-" * 40)

        # Calculate overall scores
        nb_overall = (nb_train['f1_score'] + nb_challenge['f1_score']) / 2
        vader_overall = (vader_train['f1_score'] + vader_challenge['f1_score']) / 2

        print(f"Overall F1-Score (NB): {nb_overall:.4f}")
        print(f"Overall F1-Score (VADER): {vader_overall:.4f}")
        print(f"Overall Winner: {'Naive Bayes' if nb_overall > vader_overall else 'VADER Indonesia'}")

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.test_results:
            print("No test results available. Run comparison first.")
            return

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. Performance Comparison Bar Chart
        ax1 = plt.subplot(3, 3, 1)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        nb_train = self.test_results['training_data']['naive_bayes']
        vader_train = self.test_results['training_data']['vader']

        nb_scores = [nb_train['accuracy'], nb_train['precision'], nb_train['recall'], nb_train['f1_score']]
        vader_scores = [vader_train['accuracy'], vader_train['precision'], vader_train['recall'], vader_train['f1_score']]

        x = np.arange(len(metrics))
        width = 0.35

        ax1.bar(x - width/2, nb_scores, width, label='Naive Bayes', alpha=0.8)
        ax1.bar(x + width/2, vader_scores, width, label='VADER Indonesia', alpha=0.8)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance on Training Data')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # 2. Challenging Cases Performance
        ax2 = plt.subplot(3, 3, 2)
        nb_challenge = self.test_results['challenging_data']['naive_bayes']
        vader_challenge = self.test_results['challenging_data']['vader']

        nb_scores_challenge = [nb_challenge['accuracy'], nb_challenge['precision'], nb_challenge['recall'], nb_challenge['f1_score']]
        vader_scores_challenge = [vader_challenge['accuracy'], vader_challenge['precision'], vader_challenge['recall'], vader_challenge['f1_score']]

        ax2.bar(x - width/2, nb_scores_challenge, width, label='Naive Bayes', alpha=0.8)
        ax2.bar(x + width/2, vader_scores_challenge, width, label='VADER Indonesia', alpha=0.8)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Performance on Challenging Cases')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_ylim(0, 1)

        # 3. Speed Comparison
        ax3 = plt.subplot(3, 3, 3)
        speed = self.test_results['speed']
        methods = ['Naive Bayes', 'VADER Indonesia']
        times = [speed['naive_bayes_avg_time'], speed['vader_avg_time']]
        errors = [speed['naive_bayes_std'], speed['vader_std']]

        bars = ax3.bar(methods, times, yerr=errors, capsize=5, alpha=0.8)
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Prediction Speed Comparison')

        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{time_val:.4f}s', ha='center', va='bottom')

        # 4. Confusion Matrix - Naive Bayes (Training)
        ax4 = plt.subplot(3, 3, 4)
        train_data = self.test_results['training_data']
        cm_nb = confusion_matrix(train_data['true_labels'], train_data['nb_predictions'])
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title('Confusion Matrix - Naive Bayes (Training)')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')

        # 5. Confusion Matrix - VADER (Training)
        ax5 = plt.subplot(3, 3, 5)
        cm_vader = confusion_matrix(train_data['true_labels'], train_data['vader_predictions'])
        sns.heatmap(cm_vader, annot=True, fmt='d', cmap='Oranges', ax=ax5)
        ax5.set_title('Confusion Matrix - VADER (Training)')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')

        # 6. Confidence Distribution
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(train_data['nb_confidences'], alpha=0.7, label='Naive Bayes', bins=20)
        ax6.hist(train_data['vader_confidences'], alpha=0.7, label='VADER Indonesia', bins=20)
        ax6.set_xlabel('Confidence Score')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Confidence Score Distribution')
        ax6.legend()

        # 7. Sentiment Distribution - Training Data
        ax7 = plt.subplot(3, 3, 7)
        sentiment_counts = pd.Series(train_data['true_labels']).value_counts()
        ax7.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax7.set_title('Training Data Sentiment Distribution')

        # 8. Challenging Cases Confusion Matrix - Naive Bayes
        ax8 = plt.subplot(3, 3, 8)
        challenge_data = self.test_results['challenging_data']
        cm_nb_challenge = confusion_matrix(challenge_data['true_labels'], challenge_data['nb_predictions'])
        sns.heatmap(cm_nb_challenge, annot=True, fmt='d', cmap='Blues', ax=ax8)
        ax8.set_title('Confusion Matrix - NB (Challenging)')
        ax8.set_xlabel('Predicted')
        ax8.set_ylabel('Actual')

        # 9. Challenging Cases Confusion Matrix - VADER
        ax9 = plt.subplot(3, 3, 9)
        cm_vader_challenge = confusion_matrix(challenge_data['true_labels'], challenge_data['vader_predictions'])
        sns.heatmap(cm_vader_challenge, annot=True, fmt='d', cmap='Oranges', ax=ax9)
        ax9.set_title('Confusion Matrix - VADER (Challenging)')
        ax9.set_xlabel('Predicted')
        ax9.set_ylabel('Actual')

        plt.tight_layout()
        plt.show()

        # Additional detailed analysis plots
        self._create_detailed_analysis_plots()

    def _create_detailed_analysis_plots(self):
        """Create additional detailed analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        train_data = self.test_results['training_data']
        challenge_data = self.test_results['challenging_data']

        # 1. Confidence vs Accuracy scatter plot
        ax1 = axes[0, 0]

        # Calculate accuracy for each prediction
        nb_correct = np.array(train_data['nb_predictions']) == np.array(train_data['true_labels'])
        vader_correct = np.array(train_data['vader_predictions']) == np.array(train_data['true_labels'])

        ax1.scatter(train_data['nb_confidences'], nb_correct, alpha=0.6, label='Naive Bayes')
        ax1.scatter(train_data['vader_confidences'], vader_correct, alpha=0.6, label='VADER Indonesia')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Correct Prediction (1=Correct, 0=Wrong)')
        ax1.set_title('Confidence vs Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Method Agreement Analysis
        ax2 = axes[0, 1]
        agreement = np.array(train_data['nb_predictions']) == np.array(train_data['vader_predictions'])
        agreement_rate = np.mean(agreement)

        ax2.bar(['Agree', 'Disagree'], [agreement_rate, 1-agreement_rate])
        ax2.set_ylabel('Proportion')
        ax2.set_title(f'Method Agreement Rate: {agreement_rate:.2%}')

        # Add percentage labels
        for i, v in enumerate([agreement_rate, 1-agreement_rate]):
            ax2.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')

        # 3. Performance by Sentiment Class
        ax3 = axes[1, 0]

        # Get class-wise F1 scores
        nb_class_f1 = []
        vader_class_f1 = []
        classes = ['negative', 'neutral', 'positive']

        for cls in classes:
            if cls in train_data['naive_bayes']['classification_report']:
                nb_class_f1.append(train_data['naive_bayes']['classification_report'][cls]['f1-score'])
                vader_class_f1.append(train_data['vader']['classification_report'][cls]['f1-score'])
            else:
                nb_class_f1.append(0)
                vader_class_f1.append(0)

        x = np.arange(len(classes))
        width = 0.35

        ax3.bar(x - width/2, nb_class_f1, width, label='Naive Bayes', alpha=0.8)
        ax3.bar(x + width/2, vader_class_f1, width, label='VADER Indonesia', alpha=0.8)
        ax3.set_xlabel('Sentiment Class')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('F1-Score by Sentiment Class')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes)
        ax3.legend()
        ax3.set_ylim(0, 1)

        # 4. Error Analysis
        ax4 = axes[1, 1]

        # Count prediction errors by true class
        nb_errors = {}
        vader_errors = {}

        for true_label, nb_pred, vader_pred in zip(train_data['true_labels'],
                                                  train_data['nb_predictions'],
                                                  train_data['vader_predictions']):
            if true_label not in nb_errors:
                nb_errors[true_label] = 0
                vader_errors[true_label] = 0

            if nb_pred != true_label:
                nb_errors[true_label] += 1
            if vader_pred != true_label:
                vader_errors[true_label] += 1

        classes = list(nb_errors.keys())
        nb_error_counts = [nb_errors[cls] for cls in classes]
        vader_error_counts = [vader_errors[cls] for cls in classes]

        x = np.arange(len(classes))
        ax4.bar(x - width/2, nb_error_counts, width, label='Naive Bayes', alpha=0.8)
        ax4.bar(x + width/2, vader_error_counts, width, label='VADER Indonesia', alpha=0.8)
        ax4.set_xlabel('True Sentiment Class')
        ax4.set_ylabel('Number of Errors')
        ax4.set_title('Prediction Errors by Class')
        ax4.set_xticks(x)
        ax4.set_xticklabels(classes)
        ax4.legend()

        plt.tight_layout()
        plt.show()

    def analyze_specific_examples(self, num_examples=10):
        """Analyze specific examples where methods disagree"""
        if not self.test_results:
            print("No test results available. Run comparison first.")
            return

        print("\n" + "="*60)
        print("DETAILED EXAMPLE ANALYSIS")
        print("="*60)

        train_data = self.test_results['training_data']

        # Find disagreements
        disagreements = []
        for i, (text, true_label, nb_pred, vader_pred, nb_conf, vader_conf) in enumerate(
            zip(train_data['texts'], train_data['true_labels'],
                train_data['nb_predictions'], train_data['vader_predictions'],
                train_data['nb_confidences'], train_data['vader_confidences'])):

            if nb_pred != vader_pred:
                disagreements.append({
                    'index': i,
                    'text': text,
                    'true_label': true_label,
                    'nb_prediction': nb_pred,
                    'vader_prediction': vader_pred,
                    'nb_confidence': nb_conf,
                    'vader_confidence': vader_conf,
                    'nb_correct': nb_pred == true_label,
                    'vader_correct': vader_pred == true_label
                })

        print(f"\nFound {len(disagreements)} disagreements out of {len(train_data['texts'])} examples")
        print(f"Agreement rate: {(1 - len(disagreements)/len(train_data['texts'])):.2%}")

        # Show examples
        print(f"\nShowing first {min(num_examples, len(disagreements))} disagreement examples:")
        print("-" * 80)

        for i, example in enumerate(disagreements[:num_examples]):
            print(f"\nExample {i+1}:")
            print(f"Text: {example['text']}")
            print(f"True Label: {example['true_label']}")
            print(f"Naive Bayes: {example['nb_prediction']} (conf: {example['nb_confidence']:.3f}) {'✓' if example['nb_correct'] else '✗'}")
            print(f"VADER: {example['vader_prediction']} (conf: {example['vader_confidence']:.3f}) {'✓' if example['vader_correct'] else '✗'}")
            print("-" * 80)

        return disagreements

    def generate_detailed_report(self, save_to_file=True, format='markdown'):
        """Generate a detailed comparison report in markdown or text format"""
        if not self.test_results:
            print("No test results available. Run comparison first.")
            return

        if format.lower() == 'markdown':
            return self._generate_markdown_report(save_to_file)
        else:
            return self._generate_text_report(save_to_file)

    def _generate_markdown_report(self, save_to_file=True):
        """Generate a detailed comparison report in Markdown format"""
        train_data = self.test_results['training_data']
        challenge_data = self.test_results['challenging_data']
        speed = self.test_results['speed']

        # Calculate overall scores
        nb_overall = (train_data['naive_bayes']['f1_score'] + challenge_data['naive_bayes']['f1_score']) / 2
        vader_overall = (train_data['vader']['f1_score'] + challenge_data['vader']['f1_score']) / 2
        winner = "Indonesian Naive Bayes" if nb_overall > vader_overall else "VADER Indonesia"

        # Calculate agreement rate
        agreement = np.array(train_data['nb_predictions']) == np.array(train_data['vader_predictions'])
        agreement_rate = np.mean(agreement)

        report = []

        # Header
        report.append("# Indonesian Sentiment Analysis Comparison Report")
        report.append("")
        report.append(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("---")
        report.append("")

        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append("")
        report.append(f"**Overall Winner:** {winner}")
        report.append("")
        report.append("| Method | Overall F1-Score | Training Data F1 | Challenging Cases F1 |")
        report.append("|--------|------------------|------------------|---------------------|")
        report.append(f"| Indonesian Naive Bayes | {nb_overall:.4f} | {train_data['naive_bayes']['f1_score']:.4f} | {challenge_data['naive_bayes']['f1_score']:.4f} |")
        report.append(f"| VADER Indonesia | {vader_overall:.4f} | {train_data['vader']['f1_score']:.4f} | {challenge_data['vader']['f1_score']:.4f} |")
        report.append("")

        # Key Findings
        report.append("### 🔍 Key Findings")
        report.append("")
        report.append(f"- **Method Agreement Rate:** {agreement_rate:.2%}")
        report.append(f"- **Speed Winner:** {'VADER Indonesia' if speed['vader_avg_time'] < speed['naive_bayes_avg_time'] else 'Indonesian Naive Bayes'}")
        report.append(f"- **Accuracy Winner:** {'Indonesian Naive Bayes' if nb_overall > vader_overall else 'VADER Indonesia'}")
        report.append("")

        # Performance Comparison
        report.append("## 📊 Performance Comparison")
        report.append("")

        # Training Data Performance
        report.append("### Training Data Performance")
        report.append("")
        report.append("| Metric | Indonesian Naive Bayes | VADER Indonesia | Winner |")
        report.append("|--------|------------------------|-----------------|--------|")
        report.append(f"| Accuracy | {train_data['naive_bayes']['accuracy']:.4f} | {train_data['vader']['accuracy']:.4f} | {'NB' if train_data['naive_bayes']['accuracy'] > train_data['vader']['accuracy'] else 'VADER'} |")
        report.append(f"| Precision | {train_data['naive_bayes']['precision']:.4f} | {train_data['vader']['precision']:.4f} | {'NB' if train_data['naive_bayes']['precision'] > train_data['vader']['precision'] else 'VADER'} |")
        report.append(f"| Recall | {train_data['naive_bayes']['recall']:.4f} | {train_data['vader']['recall']:.4f} | {'NB' if train_data['naive_bayes']['recall'] > train_data['vader']['recall'] else 'VADER'} |")
        report.append(f"| F1-Score | {train_data['naive_bayes']['f1_score']:.4f} | {train_data['vader']['f1_score']:.4f} | {'NB' if train_data['naive_bayes']['f1_score'] > train_data['vader']['f1_score'] else 'VADER'} |")
        report.append("")

        # Challenging Cases Performance
        report.append("### Challenging Cases Performance")
        report.append("")
        report.append("| Metric | Indonesian Naive Bayes | VADER Indonesia | Winner |")
        report.append("|--------|------------------------|-----------------|--------|")
        report.append(f"| Accuracy | {challenge_data['naive_bayes']['accuracy']:.4f} | {challenge_data['vader']['accuracy']:.4f} | {'NB' if challenge_data['naive_bayes']['accuracy'] > challenge_data['vader']['accuracy'] else 'VADER'} |")
        report.append(f"| Precision | {challenge_data['naive_bayes']['precision']:.4f} | {challenge_data['vader']['precision']:.4f} | {'NB' if challenge_data['naive_bayes']['precision'] > challenge_data['vader']['precision'] else 'VADER'} |")
        report.append(f"| Recall | {challenge_data['naive_bayes']['recall']:.4f} | {challenge_data['vader']['recall']:.4f} | {'NB' if challenge_data['naive_bayes']['recall'] > challenge_data['vader']['recall'] else 'VADER'} |")
        report.append(f"| F1-Score | {challenge_data['naive_bayes']['f1_score']:.4f} | {challenge_data['vader']['f1_score']:.4f} | {'NB' if challenge_data['naive_bayes']['f1_score'] > challenge_data['vader']['f1_score'] else 'VADER'} |")
        report.append("")

        # Class-wise Performance
        report.append("### Class-wise Performance (Training Data)")
        report.append("")

        # Get class-wise metrics
        nb_class_report = train_data['naive_bayes']['classification_report']
        vader_class_report = train_data['vader']['classification_report']

        classes = ['negative', 'neutral', 'positive']
        report.append("| Class | Method | Precision | Recall | F1-Score |")
        report.append("|-------|--------|-----------|--------|----------|")

        for cls in classes:
            if cls in nb_class_report:
                nb_metrics = nb_class_report[cls]
                vader_metrics = vader_class_report.get(cls, {'precision': 0, 'recall': 0, 'f1-score': 0})

                report.append(f"| {cls.title()} | Naive Bayes | {nb_metrics['precision']:.3f} | {nb_metrics['recall']:.3f} | {nb_metrics['f1-score']:.3f} |")
                report.append(f"| | VADER Indonesia | {vader_metrics['precision']:.3f} | {vader_metrics['recall']:.3f} | {vader_metrics['f1-score']:.3f} |")

        report.append("")

        # Speed Analysis
        report.append("## ⚡ Speed Analysis")
        report.append("")
        report.append("| Metric | Indonesian Naive Bayes | VADER Indonesia |")
        report.append("|--------|------------------------|-----------------|")
        report.append(f"| Training Time | {self.test_results['training_time']:.2f} seconds | N/A (no training required) |")
        report.append(f"| Prediction Time (avg) | {speed['naive_bayes_avg_time']:.4f} seconds | {speed['vader_avg_time']:.4f} seconds |")
        report.append(f"| Prediction Time (std) | {speed['naive_bayes_std']:.4f} seconds | {speed['vader_std']:.4f} seconds |")
        report.append("")

        speed_winner = "VADER Indonesia" if speed['vader_avg_time'] < speed['naive_bayes_avg_time'] else "Indonesian Naive Bayes"
        speed_ratio = max(speed['naive_bayes_avg_time'], speed['vader_avg_time']) / min(speed['naive_bayes_avg_time'], speed['vader_avg_time'])
        report.append(f"**Speed Winner:** {speed_winner} ({speed_ratio:.1f}x faster)")
        report.append("")


def main():
    """Main function to run the comparison"""
    print("Starting Indonesian Sentiment Analysis Comparison...")

    # Create comparison instance
    comparison = SentimentAnalyzerComparison()

    # Run comprehensive comparison
    results = comparison.run_comprehensive_comparison()

    # Print summary
    comparison.print_comparison_summary()

    # Create visualizations
    print("\nGenerating visualizations...")
    comparison.create_visualizations()

    # Analyze specific examples
    print("\nAnalyzing disagreement examples...")
    disagreements = comparison.analyze_specific_examples(5)

    # Generate detailed report
    print("\nGenerating detailed report...")
    report = comparison.generate_detailed_report()

    print("\n" + "="*60)
    print("COMPARISON COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Check the generated visualizations and report file for detailed analysis.")

    return comparison, results


if __name__ == "__main__":
    comparison, results = main()
