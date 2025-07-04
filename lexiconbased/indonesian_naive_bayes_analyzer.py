"""
Indonesian Sentiment Analyzer using Naive Bayes with Lexicon-based Features
Combines multiple Indonesian lexicon resources for enhanced sentiment analysis
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import re
import pickle
from collections import defaultdict, Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class IndonesianNaiveBayesAnalyzer:
    """
    Indonesian Sentiment Analyzer using Naive Bayes with lexicon-based features
    """
    
    def __init__(self):
        self.lexicons = {}
        self.vectorizer = None
        self.nb_model = None
        self.is_trained = False
        
        # Initialize lexicon sources
        self.lexicon_sources = {
            'sentiwords_id': 'https://raw.githubusercontent.com/lan666as/indonesia-twitter-sentiment-analysis/master/lexicon/sentiwords_id.txt',
            'combined_lexicon': 'https://raw.githubusercontent.com/lan666as/indonesia-twitter-sentiment-analysis/master/lexicon/combined_lexicon.txt',
            'sentistrength_valence': 'https://raw.githubusercontent.com/lan666as/indonesia-twitter-sentiment-analysis/master/lexicon/SentiStrengthID-valence.txt',
            'emoji_lexicon': 'https://raw.githubusercontent.com/lan666as/indonesia-twitter-sentiment-analysis/master/lexicon/emoji_utf8_lexicon.txt',
            'sentimentword': 'https://raw.githubusercontent.com/agusmakmun/SentiStrengthID/master/id_dict/sentimentword.txt',
            'idiom': 'https://raw.githubusercontent.com/agusmakmun/SentiStrengthID/master/id_dict/idiom.txt',
            'emoticon': 'https://raw.githubusercontent.com/agusmakmun/SentiStrengthID/master/id_dict/emoticon.txt',
            'boosterword': 'https://raw.githubusercontent.com/agusmakmun/SentiStrengthID/master/id_dict/boosterword.txt',
            'negatingword': 'https://raw.githubusercontent.com/agusmakmun/SentiStrengthID/master/id_dict/negatingword.txt'
        }
        
        # Load all lexicons
        self.load_lexicons()
        
    def load_lexicons(self):
        """Load all Indonesian lexicon resources"""
        print("Loading Indonesian lexicon resources...")
        
        for lexicon_name, url in self.lexicon_sources.items():
            try:
                print(f"Loading {lexicon_name}...")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.lexicons[lexicon_name] = self._parse_lexicon(response.text, lexicon_name)
                    print(f"✓ Loaded {lexicon_name}: {len(self.lexicons[lexicon_name])} entries")
                else:
                    print(f"✗ Failed to load {lexicon_name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"✗ Error loading {lexicon_name}: {e}")
                
        # Create fallback lexicon if no lexicons loaded
        if not self.lexicons:
            print("Creating fallback lexicon...")
            self._create_fallback_lexicon()
            
        print(f"Total lexicons loaded: {len(self.lexicons)}")
        
    def _parse_lexicon(self, content, lexicon_name):
        """Parse lexicon content based on format"""
        lexicon = {}
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                if lexicon_name in ['sentiwords_id', 'combined_lexicon', 'sentimentword']:
                    # Format: word\tscore or word score
                    parts = re.split(r'\s+', line, 1)
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        score = float(parts[1])
                        lexicon[word] = score
                        
                elif lexicon_name == 'sentistrength_valence':
                    # Format: word\tpos_score\tneg_score
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        word = parts[0].strip()
                        pos_score = float(parts[1])
                        neg_score = float(parts[2])
                        # Combine scores
                        lexicon[word] = pos_score - neg_score
                        
                elif lexicon_name in ['emoji_lexicon', 'emoticon']:
                    # Format: emoji/emoticon\tscore
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        symbol = parts[0].strip()
                        score = float(parts[1])
                        lexicon[symbol] = score
                        
                elif lexicon_name == 'boosterword':
                    # Booster words (positive multipliers)
                    word = line.strip()
                    lexicon[word] = 1.5  # Boost factor
                    
                elif lexicon_name == 'negatingword':
                    # Negation words
                    word = line.strip()
                    lexicon[word] = -1.0  # Negation factor
                    
                elif lexicon_name == 'idiom':
                    # Format: idiom\tscore
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        idiom = parts[0].strip()
                        score = float(parts[1])
                        lexicon[idiom] = score
                        
            except (ValueError, IndexError) as e:
                continue
                
        return lexicon
        
    def _create_fallback_lexicon(self):
        """Create basic fallback lexicon"""
        fallback = {
            'bagus': 1.0, 'baik': 0.8, 'hebat': 1.2, 'mantap': 1.0,
            'keren': 0.8, 'suka': 0.6, 'senang': 0.8, 'gembira': 1.0,
            'cinta': 1.2, 'sayang': 1.0, 'indah': 0.8, 'cantik': 0.8,
            'jelek': -1.0, 'buruk': -1.0, 'parah': -1.2, 'benci': -1.2,
            'marah': -1.0, 'sedih': -0.8, 'kecewa': -0.8, 'bodoh': -1.0,
            'tidak': -1.0, 'bukan': -1.0, 'jangan': -0.8,
            'sangat': 1.5, 'banget': 1.3, 'sekali': 1.2
        }
        self.lexicons['fallback'] = fallback
        print(f"✓ Created fallback lexicon: {len(fallback)} entries")
        
    def extract_lexicon_features(self, text):
        """Extract lexicon-based features from text"""
        if pd.isna(text) or not text:
            return {
                'positive_score': 0.0,
                'negative_score': 0.0,
                'sentiment_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'booster_count': 0,
                'negation_count': 0,
                'emoji_score': 0.0
            }
            
        text = str(text).lower()
        words = re.findall(r'\b\w+\b', text)
        
        features = {
            'positive_score': 0.0,
            'negative_score': 0.0,
            'sentiment_score': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'booster_count': 0,
            'negation_count': 0,
            'emoji_score': 0.0
        }
        
        # Process each lexicon
        for lexicon_name, lexicon in self.lexicons.items():
            for word in words:
                if word in lexicon:
                    score = lexicon[word]
                    
                    if lexicon_name == 'boosterword':
                        features['booster_count'] += 1
                    elif lexicon_name == 'negatingword':
                        features['negation_count'] += 1
                    elif lexicon_name in ['emoji_lexicon', 'emoticon']:
                        features['emoji_score'] += score
                    else:
                        if score > 0:
                            features['positive_score'] += score
                            features['positive_count'] += 1
                        elif score < 0:
                            features['negative_score'] += abs(score)
                            features['negative_count'] += 1
                        else:
                            features['neutral_count'] += 1
                            
        # Calculate overall sentiment score
        features['sentiment_score'] = features['positive_score'] - features['negative_score']
        
        # Apply booster and negation effects
        if features['booster_count'] > 0:
            features['sentiment_score'] *= (1 + 0.3 * features['booster_count'])
            
        if features['negation_count'] > 0:
            features['sentiment_score'] *= -1
            
        # Add emoji contribution
        features['sentiment_score'] += features['emoji_score']
        
        return features
        
    def prepare_features(self, texts, labels=None):
        """Prepare features for training or prediction"""
        print("Extracting lexicon features...")
        
        # Extract lexicon features
        lexicon_features = []
        for text in texts:
            features = self.extract_lexicon_features(text)
            lexicon_features.append([
                features['positive_score'],
                features['negative_score'], 
                features['sentiment_score'],
                features['positive_count'],
                features['negative_count'],
                features['booster_count'],
                features['negation_count'],
                features['emoji_score']
            ])
            
        lexicon_features = np.array(lexicon_features)
        
        # Extract TF-IDF features
        print("Extracting TF-IDF features...")
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None,
                lowercase=True
            )
            tfidf_features = self.vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.vectorizer.transform(texts)
            
        # Combine features
        tfidf_dense = tfidf_features.toarray()
        combined_features = np.hstack([lexicon_features, tfidf_dense])
        
        return combined_features

    def train(self, texts, labels, test_size=0.2, random_state=42):
        """Train the Naive Bayes model"""
        print("Training Indonesian Naive Bayes Analyzer...")

        # Prepare features
        X = self.prepare_features(texts, labels)
        y = np.array(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train model
        print("Training Naive Bayes model...")
        self.nb_model = MultinomialNB(alpha=1.0)

        # Handle negative values for MultinomialNB
        X_train_positive = X_train - X_train.min() + 1
        X_test_positive = X_test - X_test.min() + 1

        self.nb_model.fit(X_train_positive, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.nb_model.predict(X_test_positive)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Training completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    def predict(self, texts):
        """Predict sentiment for texts"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X = self.prepare_features(texts)
        X_positive = X - X.min() + 1

        predictions = self.nb_model.predict(X_positive)
        probabilities = self.nb_model.predict_proba(X_positive)

        return predictions, probabilities

    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        predictions, probabilities = self.predict([text])

        sentiment = predictions[0]
        confidence = max(probabilities[0])

        # Get lexicon-based features for additional insight
        lexicon_features = self.extract_lexicon_features(text)

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'lexicon_score': lexicon_features['sentiment_score'],
            'positive_words': lexicon_features['positive_count'],
            'negative_words': lexicon_features['negative_count'],
            'booster_words': lexicon_features['booster_count'],
            'negation_words': lexicon_features['negation_count']
        }

    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'vectorizer': self.vectorizer,
            'nb_model': self.nb_model,
            'lexicons': self.lexicons
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.nb_model = model_data['nb_model']
        self.lexicons = model_data['lexicons']
        self.is_trained = True

        print(f"Model loaded from {filepath}")

    def analyze_lexicon_coverage(self, texts):
        """Analyze lexicon coverage in the dataset"""
        print("Analyzing lexicon coverage...")

        total_words = 0
        covered_words = 0
        lexicon_hits = defaultdict(int)

        all_lexicon_words = set()
        for lexicon in self.lexicons.values():
            all_lexicon_words.update(lexicon.keys())

        for text in texts:
            if pd.isna(text):
                continue

            words = re.findall(r'\b\w+\b', str(text).lower())
            total_words += len(words)

            for word in words:
                if word in all_lexicon_words:
                    covered_words += 1

                # Count hits per lexicon
                for lexicon_name, lexicon in self.lexicons.items():
                    if word in lexicon:
                        lexicon_hits[lexicon_name] += 1

        coverage = covered_words / total_words if total_words > 0 else 0

        print(f"Lexicon Coverage Analysis:")
        print(f"Total words: {total_words}")
        print(f"Covered words: {covered_words}")
        print(f"Coverage: {coverage:.2%}")
        print(f"\nHits per lexicon:")
        for lexicon_name, hits in lexicon_hits.items():
            print(f"  {lexicon_name}: {hits}")

        return {
            'total_words': total_words,
            'covered_words': covered_words,
            'coverage': coverage,
            'lexicon_hits': dict(lexicon_hits)
        }


def create_sample_dataset():
    """Create a sample Indonesian sentiment dataset for testing"""
    sample_data = [
        ("Film ini sangat bagus dan menghibur", "positive"),
        ("Saya suka sekali dengan ceritanya", "positive"),
        ("Aktingnya luar biasa hebat", "positive"),
        ("Sangat memuaskan dan tidak mengecewakan", "positive"),
        ("Film yang indah dan menyentuh hati", "positive"),

        ("Film ini buruk sekali", "negative"),
        ("Saya tidak suka dengan jalan ceritanya", "negative"),
        ("Aktingnya sangat jelek dan mengecewakan", "negative"),
        ("Sangat membosankan dan tidak menarik", "negative"),
        ("Film yang parah dan membuang waktu", "negative"),

        ("Film ini biasa saja", "neutral"),
        ("Tidak terlalu bagus tapi juga tidak jelek", "neutral"),
        ("Lumayan untuk ditonton", "neutral"),
        ("Film standar dengan cerita yang cukup", "neutral"),
        ("Tidak ada yang istimewa dari film ini", "neutral")
    ]

    texts, labels = zip(*sample_data)
    return list(texts), list(labels)


if __name__ == "__main__":
    # Example usage
    print("Indonesian Naive Bayes Sentiment Analyzer")
    print("=" * 50)

    # Initialize analyzer
    analyzer = IndonesianNaiveBayesAnalyzer()

    # Create sample dataset
    texts, labels = create_sample_dataset()

    # Analyze lexicon coverage
    coverage = analyzer.analyze_lexicon_coverage(texts)

    # Train model
    results = analyzer.train(texts, labels)

    # Test predictions
    test_texts = [
        "Film ini sangat bagus dan menghibur sekali",
        "Saya tidak suka dengan film ini, sangat buruk",
        "Film yang biasa saja, tidak terlalu istimewa"
    ]

    print("\nTesting predictions:")
    for text in test_texts:
        result = analyzer.predict_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Lexicon Score: {result['lexicon_score']:.3f}")
        print("-" * 50)
