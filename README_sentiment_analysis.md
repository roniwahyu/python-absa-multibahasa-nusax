# Sentiment Analysis with 4 Methods

This project performs sentiment analysis on the `stemmed_text` column from the preprocessed Google Play reviews dataset using 4 different methods.

## Methods Used

1. **Score-based Sentiment Labeling**
   - Score 1-2: Negative
   - Score 3: Neutral  
   - Score 4-5: Positive

2. **TextBlob Sentiment Analysis**
   - Uses TextBlob's polarity score
   - Polarity > 0.1: Positive
   - Polarity < -0.1: Negative
   - Otherwise: Neutral

3. **VADER Sentiment Analysis**
   - Uses VADER's compound score
   - Compound >= 0.05: Positive
   - Compound <= -0.05: Negative
   - Otherwise: Neutral

4. **Ensemble Weighted Voting**
   - Combines all three methods with weights:
     - Score-based: 40%
     - TextBlob: 30%
     - VADER: 30%
   - Final sentiment is determined by highest weighted vote

## Installation

1. Install required packages:
```bash
pip install -r requirements_sentiment.txt
```

2. For TextBlob, you may need to download corpora:
```python
import nltk
nltk.download('punkt')
nltk.download('brown')
```

## Usage

1. Ensure your preprocessed CSV file `google_play_reviews_DigitalBank_preprocessed.csv` is in the same directory
2. Open and run the Jupyter notebook: `sentiment_analysis_4_methods.ipynb`
3. The notebook will:
   - Load and analyze the data
   - Apply all 4 sentiment analysis methods
   - Compare results between methods
   - Generate visualizations
   - Save results to `google_play_reviews_DigitalBank_sentiment_analysis.csv`

## Output

The analysis produces:
- Sentiment labels for each method
- Polarity/compound scores from TextBlob and VADER
- Agreement analysis between methods
- Confusion matrices comparing methods
- Sample examples for each sentiment category
- Final summary statistics

## Output Columns

The final CSV file contains:
- `reviewId`: Unique review identifier
- `content`: Original review text
- `score`: Original rating score (1-5)
- `stemmed_text`: Preprocessed and stemmed text
- `sentiment_score_based`: Score-based sentiment label
- `sentiment_textblob`: TextBlob sentiment label
- `sentiment_vader`: VADER sentiment label
- `sentiment_ensemble`: Ensemble weighted voting result
- `textblob_polarity`: TextBlob polarity score (-1 to 1)
- `vader_compound`: VADER compound score (-1 to 1)
