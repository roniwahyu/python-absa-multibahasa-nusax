"""
VADER Indonesia - Indonesian Sentiment Analysis using VADER approach
Adapted from the original VADER sentiment analyzer for Indonesian language
"""

import pandas as pd
import requests
from io import StringIO
import re
from collections import defaultdict
import math

class VaderIndonesia:
    """
    VADER sentiment analyzer adapted for Indonesian language using InSet lexicon
    """
    
    def __init__(self):
        self.lexicon = {}
        self.load_indonesian_lexicon()
        
        # VADER constants
        self.B_INCR = 0.293
        self.B_DECR = -0.293
        self.C_INCR = 0.733
        self.N_SCALAR = -0.74
        
        # Booster words (Indonesian)
        self.BOOSTER_DICT = {
            'sangat': self.B_INCR, 'amat': self.B_INCR, 'sekali': self.B_INCR,
            'banget': self.B_INCR, 'bgt': self.B_INCR, 'benar': self.B_INCR,
            'sungguh': self.B_INCR, 'betul': self.B_INCR, 'paling': self.B_INCR,
            'super': self.B_INCR, 'luar biasa': self.B_INCR, 'ekstrem': self.B_INCR,
            'agak': self.B_DECR, 'sedikit': self.B_DECR, 'kurang': self.B_DECR,
            'lumayan': self.B_DECR, 'cukup': self.B_DECR, 'rada': self.B_DECR
        }
        
        # Negation words (Indonesian)
        self.NEGATE = {
            'tidak', 'tak', 'bukan', 'bukanlah', 'tidaklah', 'jangan',
            'gak', 'ga', 'nggak', 'ngga', 'gk', 'tdk', 'nope', 'no'
        }
    
    def load_indonesian_lexicon(self):
        """
        Load Indonesian sentiment lexicon from InSet dataset
        """
        print("Loading Indonesian sentiment lexicon (InSet)...")
        
        try:
            # Load positive words
            pos_url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
            pos_response = requests.get(pos_url)
            pos_data = StringIO(pos_response.text)
            
            for line in pos_data:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[0].strip()
                    try:
                        score = float(parts[1]) / 5.0  # Normalize to [-1, 1] range
                        self.lexicon[word] = score
                    except ValueError:
                        continue
            
            # Load negative words
            neg_url = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
            neg_response = requests.get(neg_url)
            neg_data = StringIO(neg_response.text)
            
            for line in neg_data:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[0].strip()
                    try:
                        score = -float(parts[1]) / 5.0  # Negative and normalize
                        self.lexicon[word] = score
                    except ValueError:
                        continue
            
            print(f"Loaded {len(self.lexicon)} Indonesian sentiment words")
            
        except Exception as e:
            print(f"Error loading Indonesian lexicon: {e}")
            print("Using fallback lexicon...")
            self._create_fallback_lexicon()
    
    def _create_fallback_lexicon(self):
        """
        Create a basic Indonesian sentiment lexicon as fallback
        """
        # Basic positive words
        positive_words = {
            'bagus': 0.8, 'baik': 0.7, 'hebat': 0.9, 'luar biasa': 1.0,
            'mantap': 0.8, 'keren': 0.7, 'oke': 0.5, 'suka': 0.6,
            'senang': 0.7, 'gembira': 0.8, 'cinta': 0.9, 'sayang': 0.8,
            'indah': 0.7, 'cantik': 0.7, 'ganteng': 0.7, 'pintar': 0.8,
            'sukses': 0.9, 'berhasil': 0.8, 'menang': 0.8, 'juara': 0.9,
            'sempurna': 1.0, 'terbaik': 0.9, 'amazing': 0.9, 'wow': 0.7
        }
        
        # Basic negative words
        negative_words = {
            'jelek': -0.8, 'buruk': -0.8, 'parah': -0.9, 'rusak': -0.7,
            'benci': -0.9, 'marah': -0.8, 'sedih': -0.7, 'kecewa': -0.7,
            'bodoh': -0.8, 'gila': -0.6, 'sial': -0.8, 'anjing': -0.9,
            'bangsat': -0.9, 'sampah': -0.8, 'gagal': -0.8, 'kalah': -0.7,
            'lemot': -0.6, 'lambat': -0.5, 'error': -0.7, 'masalah': -0.6,
            'susah': -0.6, 'sulit': -0.5, 'ribet': -0.6, 'males': -0.5
        }
        
        self.lexicon.update(positive_words)
        self.lexicon.update(negative_words)
        print(f"Created fallback lexicon with {len(self.lexicon)} words")
    
    def _normalize_text(self, text):
        """
        Normalize Indonesian text for processing
        """
        if pd.isna(text) or not text:
            return []
        
        # Convert to lowercase and split into words
        words = str(text).lower().split()
        
        # Remove punctuation and clean words
        cleaned_words = []
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)
            if word:
                cleaned_words.append(word)
        
        return cleaned_words
    
    def _sentiment_valence(self, valence, text, is_cap_diff):
        """
        Calculate sentiment valence with modifiers
        """
        # Check for capitalization emphasis
        if is_cap_diff:
            if valence > 0:
                valence += self.C_INCR
            else:
                valence -= self.C_INCR
        
        return valence
    
    def _amplify_ep(self, text):
        """
        Check for exclamation points and amplify accordingly
        """
        ep_count = text.count('!')
        if ep_count > 4:
            ep_count = 4
        ep_amplifier = ep_count * 0.292
        return ep_amplifier
    
    def _amplify_qm(self, text):
        """
        Check for question marks
        """
        qm_count = text.count('?')
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                qm_amplifier = qm_count * 0.18
        return qm_amplifier
    
    def _sift_sentiment_scores(self, sentiments):
        """
        Separate positive and negative sentiment scores
        """
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += sentiment_score + 1
            elif sentiment_score < 0:
                neg_sum += sentiment_score - 1
            else:
                neu_count += 1
        
        return pos_sum, neg_sum, neu_count

    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative valence.
        """
        if not text:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

        sentiments = []
        words_and_emoticons = self._normalize_text(text)

        for i, item in enumerate(words_and_emoticons):
            valence = 0
            item_lowercase = item.lower()

            if item_lowercase in self.lexicon:
                valence = self.lexicon[item_lowercase]

                # Check for booster/dampener bi-grams or tri-grams
                if i > 0:
                    prev_word = words_and_emoticons[i-1].lower()
                    if prev_word in self.BOOSTER_DICT:
                        valence = valence + (valence * self.BOOSTER_DICT[prev_word])

                # Check for negation
                if i > 0:
                    prev_word = words_and_emoticons[i-1].lower()
                    if prev_word in self.NEGATE:
                        valence = valence * self.N_SCALAR
                elif i > 1:
                    prev_prev_word = words_and_emoticons[i-2].lower()
                    if prev_prev_word in self.NEGATE:
                        valence = valence * self.N_SCALAR

                # Check for capitalization emphasis
                is_cap_diff = item.isupper() and len(item) > 1
                valence = self._sentiment_valence(valence, item, is_cap_diff)

            sentiments.append(valence)

        # Check for exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)

        # Apply amplifiers
        if sentiments:
            sum_s = sum(sentiments)
            if sum_s > 0:
                sum_s += ep_amplifier
            elif sum_s < 0:
                sum_s -= ep_amplifier

            sum_s += qm_amplifier
        else:
            sum_s = 0

        # Calculate compound score
        compound = self._normalize_score(sum_s)

        # Calculate positive, negative, and neutral scores
        pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

        if pos_sum > math.fabs(neg_sum):
            pos_sum += ep_amplifier
        elif pos_sum < math.fabs(neg_sum):
            neg_sum -= ep_amplifier

        total = pos_sum + math.fabs(neg_sum) + neu_count
        if total > 0:
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)
        else:
            pos = neg = neu = 0.0

        return {
            'neg': round(neg, 3),
            'neu': round(neu, 3),
            'pos': round(pos, 3),
            'compound': round(compound, 4)
        }

    def _normalize_score(self, score, alpha=15):
        """
        Normalize the score to be between -1 and 1 using an alpha that
        approximates the max expected value
        """
        norm_score = score / math.sqrt((score * score) + alpha)
        if norm_score < -1.0:
            return -1.0
        elif norm_score > 1.0:
            return 1.0
        else:
            return norm_score

def vader_sentiment(text, vader_analyzer=None):
    """
    Analyze sentiment using VADER Indonesia
    Returns compound score and sentiment label
    """
    if vader_analyzer is None:
        vader_analyzer = VaderIndonesia()

    if pd.isna(text) or text.strip() == '':
        return 0.0, 'neutral'

    try:
        scores = vader_analyzer.polarity_scores(str(text))
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return compound, sentiment
    except Exception as e:
        print(f"Error in VADER analysis: {e}")
        return 0.0, 'neutral'
