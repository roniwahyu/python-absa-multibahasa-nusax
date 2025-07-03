# Comprehensive Multilingual Stopwords for Social Media Domain

## Overview

This project provides a comprehensive multilingual stopword dataset specifically designed for social media text processing and sentiment analysis. The dataset covers **English**, **Indonesian** (formal and colloquial), **Javanese**, and **Sundanese** languages with over **1600 entries**.

## Dataset Details

### Generated Files
- **`multilingual_stopwords_comprehensive.csv`** - Main comprehensive dataset (1617 entries)
- **`generate_comprehensive_stopwords.py`** - Generator script

### Dataset Statistics
- **Total entries**: 1617
- **English entries**: 245
- **Indonesian entries**: 1469
- **Javanese entries**: 273
- **Sundanese entries**: 272
- **Formal Indonesian entries**: 1477

### Column Structure
The CSV file contains the following columns:
- `en` - English stopwords
- `id` - Indonesian stopwords (colloquial/slang)
- `jv` - Javanese stopwords
- `su` - Sundanese stopwords
- `formal_id` - Formal Indonesian stopwords

## Features

### Comprehensive Language Coverage
1. **English**: Standard NLTK stopwords plus social media abbreviations
2. **Indonesian Slang**: Internet slang, social media abbreviations, and colloquial terms
3. **Formal Indonesian**: Standard Indonesian stopwords and formal vocabulary
4. **Javanese**: Ngoko, alus, and krama levels with Indonesian equivalents
5. **Sundanese**: Kasar/loma and lemes/hormat levels with Indonesian equivalents

### Social Media Specific Terms
- Internet abbreviations (LOL, BRB, LMAO, etc.)
- Indonesian social media slang (wkwk, anjay, mantul, etc.)
- Emotional expressions and interjections
- Common particles and fillers
- Technology and digital terms

### Multilingual Mappings
- Cross-language equivalents where applicable
- Regional language variations
- Formal and informal register distinctions
- Cultural and contextual adaptations

## Usage

### Python Example
```python
import pandas as pd

# Load the stopwords dataset
stopwords_df = pd.read_csv('multilingual_stopwords_comprehensive.csv')

# Get English stopwords
english_stopwords = set(stopwords_df['en'].dropna().str.strip())

# Get Indonesian stopwords (both formal and colloquial)
indonesian_stopwords = set(
    stopwords_df['id'].dropna().str.strip().tolist() + 
    stopwords_df['formal_id'].dropna().str.strip().tolist()
)

# Get Javanese stopwords
javanese_stopwords = set(stopwords_df['jv'].dropna().str.strip())

# Get Sundanese stopwords
sundanese_stopwords = set(stopwords_df['su'].dropna().str.strip())

# Function to remove stopwords from text
def remove_stopwords(text, language='id'):
    words = text.lower().split()
    if language == 'en':
        filtered_words = [word for word in words if word not in english_stopwords]
    elif language == 'id':
        filtered_words = [word for word in words if word not in indonesian_stopwords]
    elif language == 'jv':
        filtered_words = [word for word in words if word not in javanese_stopwords]
    elif language == 'su':
        filtered_words = [word for word in words if word not in sundanese_stopwords]
    else:
        filtered_words = words
    
    return ' '.join(filtered_words)

# Example usage
text = "saya sangat suka dengan aplikasi ini"
cleaned_text = remove_stopwords(text, 'id')
print(cleaned_text)  # Output: "suka aplikasi"
```

### For Sentiment Analysis
This stopword list is particularly useful for:
- Social media sentiment analysis
- Indonesian language processing
- Regional language text mining
- Cross-lingual information retrieval
- Multilingual chatbot development

## Technical Details

### Generation Process
1. **English Base**: Started with NLTK English stopwords
2. **Indonesian Expansion**: Added comprehensive slang and formal terms
3. **Regional Languages**: Mapped Indonesian terms to Javanese and Sundanese
4. **Social Media Terms**: Added platform-specific abbreviations and expressions
5. **Deduplication**: Removed duplicate entries while preserving language variants

### Quality Assurance
- Manual curation of language mappings
- Cultural context consideration
- Social media usage validation
- Cross-linguistic consistency checks

## Applications

### Recommended Use Cases
- **Social Media Analytics**: Twitter, Instagram, Facebook text analysis
- **Sentiment Analysis**: Product reviews, social media sentiment
- **Information Retrieval**: Search engines, document classification
- **Natural Language Processing**: Text preprocessing, feature extraction
- **Chatbot Development**: Intent recognition, response generation

### Domain Suitability
- Social media platforms
- E-commerce reviews
- News article processing
- Academic research
- Commercial text analytics

## File Structure
```
├── multilingual_stopwords_comprehensive.csv    # Main dataset
├── generate_comprehensive_stopwords.py         # Generator script
└── README_Multilingual_Stopwords.md           # This documentation
```

## Requirements
- Python 3.7+
- pandas
- nltk

## License
This dataset is provided for research and commercial use. Please cite appropriately when used in academic work.

## Contributing
To contribute additional terms or improvements:
1. Review the existing mappings in the generator script
2. Add new terms following the established patterns
3. Ensure cultural and linguistic accuracy
4. Test with representative social media text samples

## Contact
For questions, suggestions, or collaboration opportunities, please refer to the project documentation.

---

**Generated on**: 2025-07-03  
**Version**: 1.0  
**Total Entries**: 1617  
**Languages**: English, Indonesian, Javanese, Sundanese
