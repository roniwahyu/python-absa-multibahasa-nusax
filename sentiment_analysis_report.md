# Sentiment Analysis Report

This report summarizes the results and visualizations from the sentiment analysis notebook.

## Dataset Summary
- Total reviews analyzed: **9,756**
- Columns in dataset: ['reviewId', 'userName', 'userImage', 'content', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'appVersion', 'app_id', 'scraped_at', 'review_length', 'word_count', 'tokenized_text', 'filtered_text', 'normalized_text', 'no_stopwords_text', 'stemmed_text', 'sentiment_score_based', 'textblob_polarity', 'sentiment_textblob', 'vader_compound', 'sentiment_vader', 'sentiment_ensemble', 'text_length']

### Score Distribution
```
score
1    3236
2     526
3     536
4     574
5    4884
Name: count, dtype: int64
```

## Sentiment Distribution by Method
### Score Based
- positive: 5,458 (55.9%)
- negative: 3,762 (38.6%)
- neutral: 536 (5.5%)

### Textblob
- neutral: 8,871 (90.9%)
- positive: 732 (7.5%)
- negative: 153 (1.6%)

### Vader
- neutral: 8,430 (86.4%)
- positive: 999 (10.2%)
- negative: 327 (3.4%)

### Ensemble
- neutral: 8,355 (85.6%)
- positive: 966 (9.9%)
- negative: 435 (4.5%)

### Performance Summary Table
| Method      |   Positive_Count |   Neutral_Count |   Negative_Count |   Positive_Pct |   Neutral_Pct |   Negative_Pct |
|:------------|-----------------:|----------------:|-----------------:|---------------:|--------------:|---------------:|
| Score-based |             5458 |             536 |             3762 |           55.9 |           5.5 |           38.6 |
| TextBlob    |              732 |            8871 |              153 |            7.5 |          90.9 |            1.6 |
| VADER       |              999 |            8430 |              327 |           10.2 |          86.4 |            3.4 |
| Ensemble    |              966 |            8355 |              435 |            9.9 |          85.6 |            4.5 |

## Agreement Statistics
- Score-based vs TextBlob: 0.117 (11.7%)
- Score-based vs VADER: 0.152 (15.2%)
- TextBlob vs VADER: 0.927 (92.7%)
- All three methods: 0.108 (10.8%)
- Ensemble vs Score-based: 0.181 (18.1%)
- Ensemble vs TextBlob: 0.936 (93.6%)
- Ensemble vs VADER: 0.970 (97.0%)

## Correlation Matrix
```
                      score  textblob_polarity  vader_compound
score              1.000000           0.168711        0.179759
textblob_polarity  0.168711           1.000000        0.725724
vader_compound     0.179759           0.725724        1.000000
```

## Score-based to Ensemble Transition Matrix (%)
| sentiment_score_based   |   negative |   neutral |   positive |
|:------------------------|-----------:|----------:|-----------:|
| negative                |  11.0314   |   86.2839 |    2.68474 |
| neutral                 |   2.05224  |   94.5896 |    3.35821 |
| positive                |   0.164896 |   84.3166 |   15.5185  |

## Visualizations
Below are the main graphics generated in the notebook. Please refer to the notebook for interactive versions.

- `sentiment_distribution_methods.png` (please export this plot from the notebook)
- `agreement_matrix.png` (please export this plot from the notebook)
- `score_vs_polarity.png` (please export this plot from the notebook)
- `method_consistency_pie.png` (please export this plot from the notebook)
- `confusion_matrices.png` (please export this plot from the notebook)
- `score_distribution_by_sentiment.png` (please export this plot from the notebook)
- `text_length_by_sentiment.png` (please export this plot from the notebook)
## Example Reviews
### Cases Where All Methods Disagree
- **Text:** tolong saya lupa jawaban ke amanan fase 2 bank jago saya udah kirim dari via whatup,facebook,dan gma...
  - Score: 2, Score-based: negative, TextBlob: neutral (0.000), VADER: positive (0.052), Ensemble: negative
- **Text:** bener2 super lemot, mau buka aplikasi, transfer, liat saldo,, duhhh ga bisa sat set gitu...
  - Score: 2, Score-based: negative, TextBlob: neutral (0.017), VADER: positive (0.599), Ensemble: negative
- **Text:** 10x gagal transfer tapi fee 5k ttp dipotong dan totalnya 50k?tf u thinking about?syariah cuma gimmic...
  - Score: 1, Score-based: negative, TextBlob: neutral (-0.033), VADER: positive (0.273), Ensemble: negative
- **Text:** mantapp. kalo bisa jangan tunggu dibintang satuun dulu baru direspon biar ga trauma makenya...
  - Score: 5, Score-based: positive, TextBlob: neutral (0.000), VADER: negative (-0.421), Ensemble: positive
- **Text:** Update ulasan. Aplikasinya buffering ga responsive dan realtime. Terus hubungi CS nya sulit, harus l...
  - Score: 1, Score-based: negative, TextBlob: neutral (0.000), VADER: positive (0.361), Ensemble: negative

### Cases Where All Methods Agree (Neutral Example)
- **Text:** loading isi pulsa lama sekali...
  - Score: 3, All methods: neutral
- **Text:** Perangkat Terdeteksi Tidak Aman. Aplikasi Bank Jago ini terdeteksi sebagai Malware oleh aplikasi jar...
  - Score: 3, All methods: neutral

> **Note:** For full details and interactive plots, see the original Jupyter notebook.
