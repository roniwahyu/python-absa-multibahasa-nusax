# Google Play Store Multi-App Scraper

This project provides tools to scrape app information and user reviews from Google Play Store for multiple apps simultaneously, then save the data to CSV and XLSX formats.

## Features

- 🔍 **Multi-app scraping**: Scrape multiple apps in one run
- 📱 **App information**: Get detailed app metadata (title, developer, rating, installs, etc.)
- 💬 **User reviews**: Scrape user reviews with ratings and timestamps
- 📊 **Data export**: Save to both CSV and XLSX formats
- 📈 **Summary reports**: Automatic generation of summary statistics
- 🎯 **Rate limiting**: Built-in delays to avoid being blocked
- 📓 **Jupyter support**: Interactive notebook for data exploration

## Target Apps

The scraper is configured to scrape these popular apps:
- WhatsApp (`com.whatsapp`)
- Facebook (`com.facebook.katana`)
- Instagram (`com.instagram.android`)
- Snapchat (`com.snapchat.android`)
- Spotify (`com.spotify.music`)

## Installation

### Option 1: Quick Setup
```bash
python setup_scraper.py
```

### Option 2: Manual Installation
```bash
pip install -r requirements.txt
```

### Option 3: Individual Packages
```bash
pip install google-play-scraper pandas numpy tqdm openpyxl
```

## Usage

### Method 1: Python Script (Recommended)
```bash
python google_play_multi_scraper.py
```

### Method 2: Jupyter Notebook (Interactive)
```bash
jupyter notebook google_play_scraper_multi_apps.ipynb
```

## Configuration

You can modify the following parameters in the scripts:

```python
# App IDs to scrape
app_ids = [
    'com.whatsapp',
    'com.facebook.katana',
    'com.instagram.android',
    'com.snapchat.android',
    'com.spotify.music'
]

# Scraping configuration
COUNTRY = 'id'  # Indonesia
LANG = 'id'     # Indonesian
REVIEWS_PER_APP = 1000  # Number of reviews per app
```

## Output Files

The scraper generates several files with timestamps:

### CSV Files
- `google_play_apps_info_YYYYMMDD_HHMMSS.csv` - App information
- `google_play_reviews_YYYYMMDD_HHMMSS.csv` - User reviews

### Excel Files
- `google_play_apps_info_YYYYMMDD_HHMMSS.xlsx` - App information
- `google_play_reviews_YYYYMMDD_HHMMSS.xlsx` - User reviews
- `google_play_complete_data_YYYYMMDD_HHMMSS.xlsx` - Combined file with multiple sheets

### Combined Excel Structure
- **App_Information** sheet: App metadata
- **Reviews** sheet: All user reviews
- **Summary** sheet: Statistics and analysis

## Data Fields

### App Information
- `app_id`: Package name
- `title`: App name
- `developer`: Developer name
- `category`: App category
- `rating`: Average rating (1-5)
- `rating_count`: Number of ratings
- `installs`: Install count range
- `price`: App price
- `size`: App size
- `description`: App description
- `updated`: Last update date
- `version`: Current version

### Reviews
- `reviewId`: Unique review ID
- `userName`: Reviewer name
- `content`: Review text
- `score`: Rating (1-5)
- `thumbsUpCount`: Helpful votes
- `at`: Review date
- `app_id`: Associated app
- `review_length`: Character count
- `word_count`: Word count

## Rate Limiting

The scraper includes built-in rate limiting to avoid being blocked:
- 1 second delay between review batches
- 3 seconds delay between apps
- 200 reviews per batch

## Error Handling

- Graceful handling of network errors
- Automatic retry for failed requests
- Detailed error logging
- Partial data preservation

## Troubleshooting

### Common Issues

1. **Import Error: google_play_scraper**
   ```bash
   pip install google-play-scraper
   ```

2. **Import Error: openpyxl**
   ```bash
   pip install openpyxl
   ```

3. **Rate Limiting**
   - Increase delays between requests
   - Reduce batch sizes
   - Use VPN if blocked

4. **No Reviews Found**
   - Check app ID is correct
   - Try different country/language codes
   - Some apps may have limited reviews

### Performance Tips

- Start with fewer reviews per app for testing
- Use the Jupyter notebook for interactive exploration
- Monitor network connection during long scraping sessions
- Consider running during off-peak hours

## Legal Considerations

- Respect Google Play Store's terms of service
- Use scraped data responsibly
- Don't overload the servers with too many requests
- Consider the privacy of user reviews

## Next Steps

After scraping the data, you can:
1. **Sentiment Analysis**: Analyze review sentiment using NLP models
2. **Data Visualization**: Create charts and graphs
3. **Trend Analysis**: Track rating changes over time
4. **Competitive Analysis**: Compare apps in the same category
5. **Machine Learning**: Build predictive models

## Support

For issues or questions:
1. Check the error messages in the console
2. Verify your internet connection
3. Ensure all dependencies are installed
4. Try with a smaller dataset first

## Files in This Project

- `google_play_multi_scraper.py` - Main Python script
- `google_play_scraper_multi_apps.ipynb` - Jupyter notebook
- `setup_scraper.py` - Setup and dependency installer
- `requirements.txt` - Python dependencies
- `requirements_jupyter.txt` - Jupyter-specific dependencies
- `README_SCRAPER.md` - This documentation

Happy scraping! 🚀
