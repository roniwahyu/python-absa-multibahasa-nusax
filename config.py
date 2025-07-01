"""
Configuration file for the Play Store sentiment analyzer.
"""

# App configuration
APP_CONFIG = {
    'app_id': 'com.instagram.android',
    'country': 'id',  # Indonesia
    'lang': 'id',     # Indonesian
    'review_count': 2000,
    'sort_order': 'NEWEST'  # NEWEST, MOST_RELEVANT, RATING
}

# NusaX model configuration
NUSAX_CONFIG = {
    'languages': ['indonesian', 'javanese', 'sundanese'],
    'model_base': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
    'fallback_model': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'max_text_length': 512,
    'batch_size': 32
}

# Output configuration
OUTPUT_CONFIG = {
    'save_csv': True,
    'save_plots': True,
    'generate_report': True,
    'output_dir': './results'
}

# Scraping configuration
SCRAPING_CONFIG = {
    'batch_size': 200,
    'delay_between_batches': 1,  # seconds
    'max_retries': 3,
    'timeout': 30
}