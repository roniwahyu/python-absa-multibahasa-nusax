# Instagram Sentiment Analysis - Jupyter Notebook

This interactive Jupyter notebook scrapes 2000 Instagram app reviews from the Indonesian Google Play Store and performs sentiment analysis using NusaX-compatible models for Indonesian, Javanese, and Sundanese languages.

## 🚀 Quick Start

### Option 1: Windows Batch File (Easiest)
```bash
# Double-click or run in command prompt
run_jupyter.bat
```

### Option 2: Python Script
```bash
python run_jupyter.py
```

### Option 3: Manual Launch
```bash
# Install Jupyter if not installed
pip install jupyter

# Launch notebook
jupyter notebook instagram_sentiment_analysis.ipynb
```

## 📋 Prerequisites

- **Python 3.8+** installed
- **Internet connection** for downloading models and scraping reviews
- **~4GB RAM** recommended for model loading
- **~2GB disk space** for models and data

## 📦 Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements_jupyter.txt
   ```

2. **Verify installation**:
   ```bash
   python -c "import pandas, transformers, google_play_scraper; print('✅ All dependencies installed!')"
   ```

## 📓 Notebook Structure

The notebook is organized into 13 sections:

1. **Setup and Installation** - Install and import required libraries
2. **Configuration** - Set parameters for scraping and analysis
3. **NusaX Language Detection** - Initialize language processing tools
4. **Test Language Detection** - Verify language detection works
5. **Load Sentiment Models** - Download and initialize AI models
6. **Test Sentiment Analysis** - Verify sentiment analysis works
7. **Scrape Instagram Reviews** - Collect 2000 reviews from Play Store
8. **Process Reviews** - Apply sentiment analysis to all reviews
9. **Data Analysis** - Generate comprehensive statistics
10. **Visualizations** - Create charts and graphs
11. **Export Results** - Save data to CSV file
12. **Mixed Language Analysis** - Analyze code-switching patterns
13. **Summary** - Final insights and conclusions

## 🎯 Key Features

### Multi-Language Support
- **Indonesian**: Standard Indonesian language processing
- **Javanese**: Regional language detection and analysis
- **Sundanese**: Regional language detection and analysis
- **Mixed-Code**: Handles reviews with multiple languages

### Advanced Text Processing
- Abbreviation expansion (gk → tidak, udah → sudah, etc.)
- Social media text normalization
- Language-specific preprocessing
- Emoticon detection and processing

### Comprehensive Analysis
- Sentiment classification (positive/negative/neutral)
- Confidence scoring for predictions
- Language distribution analysis
- Rating correlation with sentiment
- Review length and word count statistics

### Interactive Visualizations
- Sentiment distribution pie charts
- Rating vs sentiment correlation plots
- Language distribution bar charts
- Confidence score histograms
- Review length box plots
- Cross-language sentiment analysis

## 📊 Expected Output

### Data Files
- `instagram_reviews_sentiment_YYYYMMDD_HHMMSS.csv` - Complete dataset with all analysis results

### Visualizations
- 6 interactive charts showing various aspects of the sentiment analysis
- Real-time progress bars during processing
- Detailed statistics and insights

### Sample Results
```
📊 KEY INSIGHTS:
   • Analyzed 2,000 Instagram reviews from Indonesian Play Store
   • Overall sentiment: 65.2% positive, 25.8% negative, 9.0% neutral
   • Average rating: 4.12/5 stars
   • Average sentiment confidence: 0.847
   • Language distribution: 92.3% Indonesian, 7.7% regional languages
```

## ⚙️ Configuration Options

You can modify the configuration in Section 2:

```python
CONFIG = {
    'app_id': 'com.instagram.android',  # App to analyze
    'country': 'id',                    # Indonesia
    'lang': 'id',                       # Indonesian
    'review_count': 2000,               # Number of reviews
    'sort_order': Sort.NEWEST,          # Sort by newest
    'batch_size': 200,                  # Reviews per batch
    'delay_between_batches': 1          # Rate limiting
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Try running the cell again
   - Models are cached after first download

2. **Rate Limiting During Scraping**
   - Increase `delay_between_batches` in config
   - Reduce `batch_size` in config
   - Wait and retry if blocked

3. **Memory Issues**
   - Close other applications
   - Reduce `review_count` in config
   - Restart Jupyter kernel

4. **Jupyter Won't Start**
   - Install Jupyter: `pip install jupyter`
   - Check Python installation
   - Try running from command line

### Error Messages

- **"No reviews scraped"**: Check app ID and country settings
- **"Model loading failed"**: Check internet connection and disk space
- **"CUDA not available"**: Normal on most systems, will use CPU
- **"Rate limit exceeded"**: Wait 5-10 minutes and retry

## 📈 Performance Notes

- **Scraping Speed**: ~200 reviews per minute (with rate limiting)
- **Analysis Speed**: ~100 reviews per minute (depends on hardware)
- **Memory Usage**: ~2-4GB RAM for 2000 reviews
- **Total Runtime**: ~15-30 minutes for complete analysis

## 🎓 Educational Value

This notebook demonstrates:
- Real-world web scraping techniques
- Multi-language NLP processing
- Sentiment analysis with transformer models
- Data visualization with matplotlib/seaborn
- Indonesian regional language processing
- Code-switching detection in social media text

## 📝 Data Schema

The output CSV contains these columns:

| Column | Description |
|--------|-------------|
| `content` | Review text content |
| `score` | User rating (1-5 stars) |
| `at` | Review timestamp |
| `reviewId` | Unique review identifier |
| `userName` | Reviewer username |
| `sentiment` | Predicted sentiment (positive/negative/neutral) |
| `sentiment_confidence` | Confidence score (0-1) |
| `detected_language` | Detected language (indonesian/javanese/sundanese) |
| `review_length` | Character count of review |
| `word_count` | Word count of review |

## 🤝 Contributing

To improve this notebook:
1. Add more regional languages
2. Improve language detection accuracy
3. Add more visualization types
4. Optimize performance
5. Add more text preprocessing steps

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **NusaX Team**: For multilingual Indonesian datasets
- **Google Play Scraper**: For the excellent scraping library
- **Hugging Face**: For transformer models and tools
- **Indonesian NLP Community**: For language processing resources
