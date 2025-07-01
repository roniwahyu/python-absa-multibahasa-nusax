# Instagram Reviews Sentiment Analyzer with NusaX

This application scrapes Google Play Store reviews for the Instagram app and performs sentiment analysis using NusaX models for Indonesian, Javanese, and Sundanese languages.

## Features

- **Google Play Store Scraping**: Scrapes up to 2000 reviews from Instagram app
- **Multi-language Support**: Handles Indonesian, Javanese, and Sundanese text
- **Sentiment Analysis**: Uses NusaX-compatible models for sentiment classification
- **Mixed-code Detection**: Identifies and processes mixed-language reviews
- **Comprehensive Reporting**: Generates detailed analysis reports with visualizations
- **Data Export**: Saves results to CSV format for further analysis

## Installation

1. **Clone or download this repository**

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import google_play_scraper, transformers, pandas; print('All dependencies installed successfully!')"
   ```

## Usage

### Quick Start

Run the analysis with default settings (Instagram app, 2000 reviews, Indonesian market):

```bash
python run_analysis.py
```

### Custom Parameters

```bash
python run_analysis.py --app-id com.instagram.android --country id --lang id --count 2000 --output-dir ./results
```

### Parameters

- `--app-id`: Google Play Store app ID (default: com.instagram.android)
- `--country`: Country code for the Play Store (default: id for Indonesia)
- `--lang`: Language code (default: id for Indonesian)
- `--count`: Number of reviews to scrape (default: 2000)
- `--output-dir`: Output directory for results (default: ./results)

### Direct Python Usage

```python
from playstore_sentiment_analyzer import ReviewAnalyzer

# Initialize analyzer
analyzer = ReviewAnalyzer()

# Setup for Instagram app in Indonesia
analyzer.setup('com.instagram.android', country='id', lang='id')

# Scrape and analyze 2000 reviews
df = analyzer.scrape_and_analyze(2000)

# Save results
analyzer.save_results('instagram_reviews.csv')

# Generate report
analyzer.generate_report()
```

## Output Files

The application generates several output files:

1. **CSV File**: `instagram_reviews_sentiment_YYYYMMDD_HHMMSS.csv`
   - Contains all scraped reviews with sentiment analysis results
   - Columns include: content, score, sentiment, sentiment_confidence, detected_language, etc.

2. **Visualization**: `sentiment_analysis_report_YYYYMMDD_HHMMSS.png`
   - Charts showing sentiment distribution, rating correlations, language distribution

3. **Console Report**: Detailed text report with statistics and insights

## Data Schema

The output CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `content` | Review text content |
| `score` | User rating (1-5 stars) |
| `at` | Review timestamp |
| `reviewId` | Unique review identifier |
| `userName` | Reviewer username |
| `userImage` | Reviewer profile image URL |
| `sentiment` | Predicted sentiment (positive/negative/neutral) |
| `sentiment_confidence` | Confidence score (0-1) |
| `detected_language` | Detected language (indonesian/javanese/sundanese) |
| `review_length` | Character count of review |
| `word_count` | Word count of review |

## NusaX Integration

This application is designed to work with the NusaX dataset for Indonesian regional languages:

- **Indonesian**: Standard Indonesian language processing
- **Javanese**: Regional language support for Javanese text
- **Sundanese**: Regional language support for Sundanese text
- **Mixed-code**: Handles reviews containing multiple languages

### Language Detection

The application includes sophisticated language detection for:
- Pure Indonesian text
- Pure regional language text (Javanese/Sundanese)
- Mixed-code text (combination of languages)
- Code-switching patterns common in Indonesian social media

## Configuration

Edit `config.py` to customize:

```python
APP_CONFIG = {
    'app_id': 'com.instagram.android',
    'country': 'id',
    'lang': 'id',
    'review_count': 2000,
    'sort_order': 'NEWEST'
}

NUSAX_CONFIG = {
    'languages': ['indonesian', 'javanese', 'sundanese'],
    'model_base': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
    'max_text_length': 512
}
```

## Troubleshooting

### Common Issues

1. **Rate Limiting**: If you encounter rate limiting, reduce the batch size or increase delays in `config.py`

2. **Memory Issues**: For large datasets, consider processing in smaller batches

3. **Model Loading**: If models fail to load, check your internet connection and available disk space

4. **No Reviews Found**: Verify the app ID and ensure the app is available in the specified country

### Error Messages

- `No reviews scraped`: Check app ID and country/language settings
- `Model loading failed`: Ensure transformers library is properly installed
- `Rate limit exceeded`: Wait and retry, or adjust scraping parameters

## Dependencies

- `google-play-scraper`: For scraping Play Store reviews
- `transformers`: For sentiment analysis models
- `pandas`: For data manipulation
- `torch`: For deep learning models
- `matplotlib/seaborn`: For visualizations
- `langdetect`: For language detection
- `tqdm`: For progress bars

## Performance Notes

- **Scraping Speed**: ~200 reviews per minute (with rate limiting)
- **Analysis Speed**: ~100 reviews per minute (depends on model and hardware)
- **Memory Usage**: ~2-4GB RAM for 2000 reviews with transformer models
- **Storage**: ~10-50MB per 2000 reviews (depending on review length)

## Examples

### Sample Output

```
SENTIMENT ANALYSIS REPORT
==================================================

Total Reviews Analyzed: 2000
Date Range: 2024-01-01 to 2024-01-15

Sentiment Distribution:
  Positive: 1200 (60.0%)
  Negative: 600 (30.0%)
  Neutral: 200 (10.0%)

Language Distribution:
  Indonesian: 1800 (90.0%)
  Javanese: 150 (7.5%)
  Sundanese: 50 (2.5%)

Average Confidence by Sentiment:
  Positive: 0.856
  Negative: 0.823
  Neutral: 0.645
```

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **NusaX Team**: For providing multilingual Indonesian datasets
- **Google Play Scraper**: For the excellent scraping library
- **Hugging Face**: For transformer models and tools
- **Indonesian NLP Community**: For language processing resources