"""
Utilities for working with NusaX datasets and models for sentiment analysis.
This module provides functions to work with Indonesian, Javanese, and Sundanese text.
"""

import re
import json
from typing import Dict, List, Tuple, Optional
import requests
from pathlib import Path

class NusaXLanguageDetector:
    """Language detector specifically for NusaX supported languages."""
    
    def __init__(self):
        # Common words and patterns for each language
        self.language_patterns = {
            'indonesian': {
                'words': ['yang', 'dan', 'ini', 'itu', 'dengan', 'untuk', 'dari', 'pada', 'dalam', 'tidak', 'adalah', 'akan', 'sudah', 'bisa', 'juga'],
                'patterns': [r'\byang\b', r'\bdan\b', r'\btidak\b', r'\badalah\b']
            },
            'javanese': {
                'words': ['lan', 'karo', 'iki', 'kuwi', 'saka', 'kanggo', 'ing', 'ora', 'iku', 'wis', 'iso', 'uga'],
                'patterns': [r'\blan\b', r'\bkaro\b', r'\bora\b', r'\biku\b']
            },
            'sundanese': {
                'words': ['jeung', 'sareng', 'ieu', 'eta', 'ti', 'pikeun', 'di', 'henteu', 'teu', 'geus', 'tiasa', 'oge'],
                'patterns': [r'\bjeung\b', r'\bsareng\b', r'\bhenteu\b', r'\bteu\b']
            }
        }
    
    def detect_language_mix(self, text: str) -> Dict[str, float]:
        """
        Detect the mix of languages in the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with language scores
        """
        text_lower = text.lower()
        scores = {'indonesian': 0, 'javanese': 0, 'sundanese': 0}
        
        for lang, patterns in self.language_patterns.items():
            # Count word matches
            word_matches = sum(1 for word in patterns['words'] if word in text_lower)
            
            # Count pattern matches
            pattern_matches = sum(1 for pattern in patterns['patterns'] if re.search(pattern, text_lower))
            
            # Calculate score
            total_words = len(text_lower.split())
            if total_words > 0:
                scores[lang] = (word_matches + pattern_matches) / total_words
        
        return scores
    
    def get_dominant_language(self, text: str) -> str:
        """
        Get the dominant language in the text.
        
        Args:
            text: Input text
            
        Returns:
            Dominant language name
        """
        scores = self.detect_language_mix(text)
        return max(scores, key=scores.get)

class NusaXTextPreprocessor:
    """Text preprocessor for NusaX languages."""
    
    def __init__(self):
        # Common abbreviations and slang in Indonesian/regional languages
        self.abbreviations = {
            'gk': 'tidak',
            'ga': 'tidak', 
            'gak': 'tidak',
            'udh': 'sudah',
            'udah': 'sudah',
            'blm': 'belum',
            'blom': 'belum',
            'krn': 'karena',
            'krna': 'karena',
            'dgn': 'dengan',
            'sm': 'sama',
            'tp': 'tapi',
            'trs': 'terus',
            'yg': 'yang',
            'utk': 'untuk',
            'dr': 'dari',
            'ke': 'ke',
            'di': 'di',
            'org': 'orang',
            'bgt': 'banget',
            'bgt': 'banget',
            'bener': 'benar',
            'emg': 'memang',
            'emang': 'memang'
        }
        
        # Emoticons and their sentiment
        self.emoticons = {
            ':)': 'positive',
            ':-)': 'positive', 
            ':D': 'positive',
            ':-D': 'positive',
            ':P': 'positive',
            ':(': 'negative',
            ':-(': 'negative',
            ':/': 'negative',
            ':-/': 'negative',
            ':\'(': 'negative'
        }
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand common abbreviations in the text.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded abbreviations
        """
        words = text.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.abbreviations:
                expanded_words.append(self.abbreviations[word_lower])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def extract_emoticons(self, text: str) -> List[str]:
        """
        Extract emoticons from text.
        
        Args:
            text: Input text
            
        Returns:
            List of emoticons found
        """
        found_emoticons = []
        for emoticon in self.emoticons.keys():
            if emoticon in text:
                found_emoticons.append(emoticon)
        return found_emoticons
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Expand abbreviations
        text = self.expand_abbreviations(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()

class NusaXDatasetLoader:
    """Loader for NusaX sentiment datasets."""
    
    def __init__(self, cache_dir: str = "./nusax_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # NusaX dataset URLs (these would be the actual URLs from the repository)
        self.dataset_urls = {
            'indonesian': 'https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian_sentiment.json',
            'javanese': 'https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/javanese_sentiment.json',
            'sundanese': 'https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/sundanese_sentiment.json'
        }
    
    def download_dataset(self, language: str) -> Optional[Dict]:
        """
        Download NusaX dataset for a specific language.
        
        Args:
            language: Language name (indonesian, javanese, sundanese)
            
        Returns:
            Dataset dictionary or None if failed
        """
        if language not in self.dataset_urls:
            print(f"Language {language} not supported")
            return None
        
        cache_file = self.cache_dir / f"{language}_sentiment.json"
        
        # Check if cached version exists
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cached dataset: {e}")
        
        # Download dataset
        try:
            print(f"Downloading {language} sentiment dataset...")
            response = requests.get(self.dataset_urls[language], timeout=30)
            response.raise_for_status()
            
            dataset = response.json()
            
            # Cache the dataset
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            print(f"Dataset cached to {cache_file}")
            return dataset
            
        except Exception as e:
            print(f"Error downloading dataset for {language}: {e}")
            return None
    
    def load_all_datasets(self) -> Dict[str, Dict]:
        """
        Load all NusaX sentiment datasets.
        
        Returns:
            Dictionary with datasets for each language
        """
        datasets = {}
        for language in self.dataset_urls.keys():
            dataset = self.download_dataset(language)
            if dataset:
                datasets[language] = dataset
        return datasets

def create_mixed_language_examples():
    """Create examples of mixed-language reviews for testing."""
    examples = [
        {
            'text': 'Instagram bagus banget, tapi kadang lemot juga sih',
            'expected_lang': 'indonesian',
            'expected_sentiment': 'neutral'
        },
        {
            'text': 'Aplikasi iki apik tenan, nanging kadhang angel dibukak',
            'expected_lang': 'javanese', 
            'expected_sentiment': 'neutral'
        },
        {
            'text': 'Aplikasi ieu saé pisan, tapi kadang hese dibuka',
            'expected_lang': 'sundanese',
            'expected_sentiment': 'neutral'
        },
        {
            'text': 'Love this app! Sangat mudah digunakan dan fiturnya lengkap',
            'expected_lang': 'indonesian',
            'expected_sentiment': 'positive'
        },
        {
            'text': 'Jelek banget aplikasinya, sering error lan ora iso dibuka',
            'expected_lang': 'mixed',
            'expected_sentiment': 'negative'
        }
    ]
    return examples

if __name__ == "__main__":
    # Test the utilities
    detector = NusaXLanguageDetector()
    preprocessor = NusaXTextPreprocessor()
    
    examples = create_mixed_language_examples()
    
    print("Testing NusaX utilities:")
    print("=" * 50)
    
    for i, example in enumerate(examples, 1):
        text = example['text']
        print(f"\nExample {i}: {text}")
        
        # Language detection
        lang_scores = detector.detect_language_mix(text)
        dominant_lang = detector.get_dominant_language(text)
        print(f"Language scores: {lang_scores}")
        print(f"Dominant language: {dominant_lang}")
        
        # Text preprocessing
        cleaned = preprocessor.clean_text(text)
        print(f"Cleaned text: {cleaned}")
        
        # Emoticons
        emoticons = preprocessor.extract_emoticons(text)
        if emoticons:
            print(f"Emoticons found: {emoticons}")