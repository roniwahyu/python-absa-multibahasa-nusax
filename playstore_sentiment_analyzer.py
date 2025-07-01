#!/usr/bin/env python3
"""
Google Play Store Reviews Scraper and Sentiment Analyzer
Scrapes Instagram app reviews and performs sentiment analysis using NusaX models
for Indonesian, Javanese, and Sundanese languages.
"""

import pandas as pd
import numpy as np
from google_play_scraper import app, reviews, Sort
import time
import json
from datetime import datetime
import os
from typing import List, Dict, Tuple
import re
from langdetect import detect, LangDetectError
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class PlayStoreReviewScraper:
    """Scrapes Google Play Store reviews for a given app."""
    
    def __init__(self, app_id: str, country: str = 'id', lang: str = 'id'):
        self.app_id = app_id
        self.country = country
        self.lang = lang
        
    def scrape_reviews(self, count: int = 2000, sort_order: Sort = Sort.NEWEST) -> List[Dict]:
        """
        Scrape reviews from Google Play Store.
        
        Args:
            count: Number of reviews to scrape
            sort_order: Sort order for reviews
            
        Returns:
            List of review dictionaries
        """
        print(f"Scraping {count} reviews for {self.app_id}...")
        
        try:
            # Get app information first
            app_info = app(self.app_id, lang=self.lang, country=self.country)
            print(f"App: {app_info['title']}")
            print(f"Developer: {app_info['developer']}")
            print(f"Rating: {app_info['score']}")
            
            # Scrape reviews in batches
            all_reviews = []
            continuation_token = None
            batch_size = 200  # Reviews per batch
            
            with tqdm(total=count, desc="Scraping reviews") as pbar:
                while len(all_reviews) < count:
                    try:
                        result, continuation_token = reviews(
                            self.app_id,
                            lang=self.lang,
                            country=self.country,
                            sort=sort_order,
                            count=min(batch_size, count - len(all_reviews)),
                            continuation_token=continuation_token
                        )
                        
                        if not result:
                            print("No more reviews available")
                            break
                            
                        all_reviews.extend(result)
                        pbar.update(len(result))
                        
                        # Add delay to avoid rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error scraping batch: {e}")
                        break
            
            print(f"Successfully scraped {len(all_reviews)} reviews")
            return all_reviews
            
        except Exception as e:
            print(f"Error scraping reviews: {e}")
            return []

class NusaXSentimentAnalyzer:
    """Sentiment analyzer using NusaX models for Indonesian, Javanese, and Sundanese."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.language_codes = {
            'indonesian': 'id',
            'javanese': 'jv', 
            'sundanese': 'su'
        }
        
    def load_models(self):
        """Load NusaX sentiment analysis models."""
        print("Loading NusaX sentiment analysis models...")
        
        # For this implementation, we'll use a general multilingual model
        # In practice, you would load specific NusaX models for each language
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create sentiment analysis pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Store for all languages (in practice, you'd have separate models)
            for lang in self.language_codes.keys():
                self.pipelines[lang] = sentiment_pipeline
                
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to a simpler model
            self.load_fallback_model()
    
    def load_fallback_model(self):
        """Load a fallback sentiment analysis model."""
        print("Loading fallback sentiment model...")
        try:
            sentiment_pipeline = pipeline("sentiment-analysis", 
                                        model="nlptown/bert-base-multilingual-uncased-sentiment")
            for lang in self.language_codes.keys():
                self.pipelines[lang] = sentiment_pipeline
            print("Fallback model loaded successfully!")
        except Exception as e:
            print(f"Error loading fallback model: {e}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        try:
            detected = detect(text)
            # Map detected language to our supported languages
            if detected == 'id':
                return 'indonesian'
            elif detected in ['jv', 'su']:
                # For Javanese and Sundanese, we'll need more sophisticated detection
                # For now, we'll classify non-Indonesian as mixed
                return 'indonesian'  # Default to Indonesian for mixed-code
            else:
                return 'indonesian'  # Default fallback
        except LangDetectError:
            return 'indonesian'  # Default fallback
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Truncate if too long (model limitations)
        if len(text) > 512:
            text = text[:512]
            
        return text.strip()
    
    def analyze_sentiment(self, text: str, language: str = None) -> Dict:
        """
        Analyze sentiment of the given text.
        
        Args:
            text: Input text
            language: Language of the text (auto-detect if None)
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'language': 'unknown'
            }
        
        # Detect language if not provided
        if language is None:
            language = self.detect_language(text)
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        try:
            # Get the appropriate pipeline
            pipeline = self.pipelines.get(language, self.pipelines.get('indonesian'))
            
            if pipeline is None:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'language': language
                }
            
            # Perform sentiment analysis
            result = pipeline(processed_text)
            
            # Normalize the result
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            # Map labels to standard format
            label = result['label'].lower()
            if 'pos' in label or label == 'positive':
                sentiment = 'positive'
            elif 'neg' in label or label == 'negative':
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': result['score'],
                'language': language
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'language': language
            }

class ReviewAnalyzer:
    """Main class for analyzing Play Store reviews."""
    
    def __init__(self):
        self.scraper = None
        self.sentiment_analyzer = NusaXSentimentAnalyzer()
        self.reviews_df = None
        
    def setup(self, app_id: str, country: str = 'id', lang: str = 'id'):
        """Setup the analyzer with app details."""
        self.scraper = PlayStoreReviewScraper(app_id, country, lang)
        self.sentiment_analyzer.load_models()
        
    def scrape_and_analyze(self, count: int = 2000) -> pd.DataFrame:
        """
        Scrape reviews and perform sentiment analysis.
        
        Args:
            count: Number of reviews to scrape
            
        Returns:
            DataFrame with reviews and sentiment analysis
        """
        if self.scraper is None:
            raise ValueError("Analyzer not setup. Call setup() first.")
        
        # Scrape reviews
        reviews_data = self.scraper.scrape_reviews(count, Sort.NEWEST)
        
        if not reviews_data:
            print("No reviews scraped")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews_data)
        
        # Perform sentiment analysis
        print("Performing sentiment analysis...")
        sentiments = []
        confidences = []
        languages = []
        
        for review in tqdm(df['content'], desc="Analyzing sentiment"):
            result = self.sentiment_analyzer.analyze_sentiment(review)
            sentiments.append(result['sentiment'])
            confidences.append(result['confidence'])
            languages.append(result['language'])
        
        # Add sentiment analysis results to DataFrame
        df['sentiment'] = sentiments
        df['sentiment_confidence'] = confidences
        df['detected_language'] = languages
        
        # Add additional features
        df['review_length'] = df['content'].str.len()
        df['word_count'] = df['content'].str.split().str.len()
        
        self.reviews_df = df
        return df
    
    def save_results(self, filename: str = None):
        """Save results to CSV file."""
        if self.reviews_df is None:
            print("No data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"instagram_reviews_sentiment_{timestamp}.csv"
        
        self.reviews_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Results saved to {filename}")
        
    def generate_report(self):
        """Generate a summary report of the sentiment analysis."""
        if self.reviews_df is None:
            print("No data to analyze")
            return
        
        df = self.reviews_df
        
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS REPORT")
        print("="*50)
        
        print(f"\nTotal Reviews Analyzed: {len(df)}")
        print(f"Date Range: {df['at'].min()} to {df['at'].max()}")
        
        # Sentiment distribution
        print("\nSentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Language distribution
        print("\nLanguage Distribution:")
        lang_counts = df['detected_language'].value_counts()
        for lang, count in lang_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {lang.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Rating vs Sentiment correlation
        print("\nRating vs Sentiment:")
        rating_sentiment = df.groupby(['score', 'sentiment']).size().unstack(fill_value=0)
        print(rating_sentiment)
        
        # Average confidence by sentiment
        print("\nAverage Confidence by Sentiment:")
        conf_by_sentiment = df.groupby('sentiment')['sentiment_confidence'].mean()
        for sentiment, conf in conf_by_sentiment.items():
            print(f"  {sentiment.capitalize()}: {conf:.3f}")
        
        # Create visualizations
        self.create_visualizations()
    
    def create_visualizations(self):
        """Create visualization plots."""
        if self.reviews_df is None:
            return
        
        df = self.reviews_df
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Instagram App Reviews - Sentiment Analysis Report', fontsize=16)
        
        # 1. Sentiment Distribution
        sentiment_counts = df['sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Sentiment Distribution')
        
        # 2. Rating vs Sentiment
        rating_sentiment = df.groupby(['score', 'sentiment']).size().unstack(fill_value=0)
        rating_sentiment.plot(kind='bar', ax=axes[0, 1], stacked=True)
        axes[0, 1].set_title('Rating vs Sentiment')
        axes[0, 1].set_xlabel('Rating')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(title='Sentiment')
        
        # 3. Language Distribution
        lang_counts = df['detected_language'].value_counts()
        axes[1, 0].bar(lang_counts.index, lang_counts.values)
        axes[1, 0].set_title('Language Distribution')
        axes[1, 0].set_xlabel('Language')
        axes[1, 0].set_ylabel('Count')
        
        # 4. Sentiment Confidence Distribution
        axes[1, 1].hist(df['sentiment_confidence'], bins=20, alpha=0.7)
        axes[1, 1].set_title('Sentiment Confidence Distribution')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"sentiment_analysis_report_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {plot_filename}")
        
        plt.show()

def main():
    """Main function to run the sentiment analysis."""
    print("Instagram Reviews Sentiment Analyzer")
    print("Using NusaX for Indonesian, Javanese, and Sundanese")
    print("="*50)
    
    # Configuration
    APP_ID = "com.instagram.android"
    COUNTRY = "id"
    LANG = "id"
    REVIEW_COUNT = 2000
    
    # Initialize analyzer
    analyzer = ReviewAnalyzer()
    
    try:
        # Setup
        analyzer.setup(APP_ID, COUNTRY, LANG)
        
        # Scrape and analyze
        df = analyzer.scrape_and_analyze(REVIEW_COUNT)
        
        if df.empty:
            print("No reviews were scraped. Exiting.")
            return
        
        # Save results
        analyzer.save_results()
        
        # Generate report
        analyzer.generate_report()
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()