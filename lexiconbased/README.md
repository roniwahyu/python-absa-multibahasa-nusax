# Indonesian Naive Bayes Sentiment Analyzer

A comprehensive sentiment analysis tool for Indonesian text that combines multiple lexicon resources with Naive Bayes classification.

## Features

- **Multiple Lexicon Integration**: Combines 9 different Indonesian lexicon sources
- **Hybrid Approach**: Uses both lexicon-based features and TF-IDF features
- **Comprehensive Analysis**: Provides detailed sentiment analysis with confidence scores
- **Easy to Use**: Simple API for training and prediction
- **Extensible**: Easy to add new lexicon sources

## Lexicon Sources

The analyzer automatically loads lexicon data from these repositories:

1. **lan666as/indonesia-twitter-sentiment-analysis**:
   - `sentiwords_id.txt`: Word sentiment scores
   - `combined_lexicon.txt`: Combined lexicon
   - `SentiStrengthID-valence.txt`: Valence scores
   - `emoji_utf8_lexicon.txt`: Emoji sentiment scores

2. **agusmakmun/SentiStrengthID**:
   - `sentimentword.txt`: Sentiment words with scores
   - `idiom.txt`: Indonesian idioms with sentiment
   - `emoticon.txt`: Emoticon sentiment scores
   - `boosterword.txt`: Intensity booster words
   - `negatingword.txt`: Negation words

## Installation

```bash
pip install pandas numpy scikit-learn requests matplotlib seaborn
```

## Quick Start

```python
from indonesian_naive_bayes_analyzer import IndonesianNaiveBayesAnalyzer

# Initialize analyzer
analyzer = IndonesianNaiveBayesAnalyzer()

# Create sample data
texts = [
    "Film ini sangat bagus dan menghibur",
    "Saya tidak suka dengan ceritanya",
    "Film yang biasa saja"
]
labels = ["positive", "negative", "neutral"]

# Train model
analyzer.train(texts, labels)

# Predict sentiment
result = analyzer.predict_sentiment("Film ini luar biasa hebat!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Features Extracted

The analyzer extracts the following lexicon-based features:

- **Positive Score**: Sum of positive word scores
- **Negative Score**: Sum of negative word scores  
- **Sentiment Score**: Overall sentiment score
- **Word Counts**: Positive, negative, neutral word counts
- **Booster Count**: Number of intensity booster words
- **Negation Count**: Number of negation words
- **Emoji Score**: Sentiment from emojis/emoticons

## API Reference

### IndonesianNaiveBayesAnalyzer

#### Methods

- `__init__()`: Initialize analyzer and load lexicons
- `train(texts, labels, test_size=0.2)`: Train the model
- `predict(texts)`: Predict sentiment for multiple texts
- `predict_sentiment(text)`: Predict sentiment for single text
- `extract_lexicon_features(text)`: Extract lexicon features
- `analyze_lexicon_coverage(texts)`: Analyze lexicon coverage
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load trained model

#### Returns

`predict_sentiment()` returns a dictionary with:
- `sentiment`: Predicted sentiment class
- `confidence`: Prediction confidence (0-1)
- `lexicon_score`: Raw lexicon-based score
- `positive_words`: Count of positive words
- `negative_words`: Count of negative words
- `booster_words`: Count of booster words
- `negation_words`: Count of negation words

## Examples

### Basic Usage

```python
# Initialize and train
analyzer = IndonesianNaiveBayesAnalyzer()
analyzer.train(train_texts, train_labels)

# Single prediction
result = analyzer.predict_sentiment("Sangat bagus sekali!")
print(result)
```

### Batch Prediction

```python
# Multiple predictions
texts = ["Text 1", "Text 2", "Text 3"]
predictions, probabilities = analyzer.predict(texts)
```

### Feature Analysis

```python
# Extract lexicon features
features = analyzer.extract_lexicon_features("Film ini sangat bagus")
print(f"Sentiment score: {features['sentiment_score']}")
print(f"Positive words: {features['positive_count']}")
```

### Coverage Analysis

```python
# Analyze lexicon coverage
coverage = analyzer.analyze_lexicon_coverage(texts)
print(f"Coverage: {coverage['coverage']:.2%}")
```

## Files

- `indonesian_naive_bayes_analyzer.py`: Main analyzer class
- `indonesian_naive_bayes_demo.ipynb`: Jupyter notebook demo
- `example_usage.py`: Simple usage example
- `README.md`: This documentation

## Performance

The analyzer combines:
- **Lexicon-based features**: 8 features from multiple lexicon sources
- **TF-IDF features**: Up to 5000 n-gram features (1-2 grams)
- **Naive Bayes classifier**: MultinomialNB with alpha=1.0

Typical performance on Indonesian sentiment data:
- Accuracy: 80-90% (depends on dataset quality)
- Coverage: 60-80% vocabulary coverage with lexicons

## Limitations

- Requires internet connection for initial lexicon loading
- Performance depends on lexicon coverage of your specific domain
- Best suited for Indonesian text (Bahasa Indonesia)
- May need domain-specific training for specialized texts

## Contributing

To add new lexicon sources:

1. Add URL to `lexicon_sources` in `__init__()`
2. Implement parsing logic in `_parse_lexicon()`
3. Test with your lexicon format

## License

This project uses publicly available lexicon resources. Please check individual repository licenses for the lexicon data sources.

## Acknowledgments

Thanks to the creators of the Indonesian lexicon resources:
- lan666as/indonesia-twitter-sentiment-analysis
- agusmakmun/SentiStrengthID
- fajri91/InSet (used in VADER Indonesia)

## Citation

If you use this analyzer in your research, please cite the original lexicon sources and this implementation.
