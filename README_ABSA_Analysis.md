# Aspect-Based Sentiment Analysis with Multi-Class SVM

This project implements a comprehensive aspect-based sentiment analysis system using Support Vector Machines (SVM) with multiple feature extraction methods and sentiment labeling approaches.

## Overview

The analysis compares different combinations of:
- **Feature Extraction**: TF-IDF and Word2Vec
- **Sentiment Methods**: Score-based, TextBlob, VADER, and Ensemble
- **SVM Kernels**: Linear, RBF, Polynomial, and Sigmoid
- **Training Splits**: 65%, 70%, and 75%

## Dataset

- **Source**: Google Play Store reviews for Digital Banking apps
- **File**: `google_play_reviews_DigitalBank_sentiment_analysis.csv`
- **Size**: 9,756 reviews
- **Features**: Pre-processed stemmed text with multiple sentiment labels

### Dataset Columns
- `reviewId`: Unique identifier for each review
- `content`: Original review text
- `score`: User rating (1-5 stars)
- `stemmed_text`: Preprocessed and stemmed text for analysis
- `sentiment_score_based`: Sentiment based on score (1-2=negative, 3=neutral, 4-5=positive)
- `sentiment_textblob`: TextBlob sentiment analysis result
- `sentiment_vader`: VADER sentiment analysis result
- `sentiment_ensemble`: Ensemble of all three methods
- `textblob_polarity`: TextBlob polarity score
- `vader_compound`: VADER compound score

## Implementation Files

### 1. Jupyter Notebook
- **File**: `aspect_based_sentiment_analysis.ipynb`
- **Purpose**: Interactive analysis with step-by-step execution
- **Features**: Detailed visualizations and explanations

### 2. Python Script
- **File**: `aspect_based_sentiment_analysis.py`
- **Purpose**: Automated execution of complete analysis
- **Features**: Comprehensive evaluation with progress tracking

## Feature Extraction Methods

### TF-IDF (Term Frequency-Inverse Document Frequency)
- **Parameters**:
  - Max features: 5,000
  - N-gram range: (1, 2)
  - Min document frequency: 2
  - Max document frequency: 95%
- **Output**: 5,000-dimensional sparse vectors

### Word2Vec
- **Parameters**:
  - Vector size: 100 dimensions
  - Window size: 5
  - Min count: 2
  - Epochs: 10
- **Output**: 100-dimensional dense vectors (document-level averaging)

## Sentiment Labeling Methods

### 1. Score-Based Sentiment
- **Logic**: Based on user ratings
  - 1-2 stars → Negative
  - 3 stars → Neutral
  - 4-5 stars → Positive
- **Distribution**: 55.9% positive, 38.6% negative, 5.5% neutral

### 2. TextBlob Sentiment
- **Method**: Rule-based sentiment analysis
- **Distribution**: 90.9% neutral, 7.5% positive, 1.6% negative

### 3. VADER Sentiment
- **Method**: Valence Aware Dictionary and sEntiment Reasoner
- **Distribution**: 93.4% positive, 6.6% neutral

### 4. Ensemble Sentiment
- **Method**: Weighted voting of all three methods
- **Distribution**: 53.3% positive, 35.8% negative, 10.9% neutral

## SVM Configuration

### Kernels Tested
1. **Linear**: Good for high-dimensional data
2. **RBF (Radial Basis Function)**: Non-linear, good for complex patterns
3. **Polynomial**: Non-linear with polynomial decision boundaries
4. **Sigmoid**: Neural network-like activation

### Training Scenarios
- **65% Training / 35% Testing**: More test data for robust evaluation
- **70% Training / 30% Testing**: Balanced approach
- **75% Training / 25% Testing**: More training data for complex models

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall classification accuracy
- **ROC AUC**: Area Under the Receiver Operating Characteristic curve
- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Precision, recall, and F1-score per class

### Visualization Outputs
1. **Sentiment Distribution**: Bar plots showing label distributions
2. **Performance Comparison**: Box plots comparing different configurations
3. **Confusion Matrices**: Heatmaps for best-performing models
4. **ROC Curves**: Performance curves for multiclass classification

## Expected Results

The analysis will generate:
- **96 total experiments** (4 methods × 2 features × 4 kernels × 3 splits)
- **Performance comparison** across all configurations
- **Best model identification** for each sentiment method
- **Comprehensive summary** with recommendations

## Usage Instructions

### Running the Jupyter Notebook
```bash
jupyter notebook aspect_based_sentiment_analysis.ipynb
```

### Running the Python Script
```bash
python aspect_based_sentiment_analysis.py
```

### Output Files
- `svm_results_YYYYMMDD_HHMMSS.csv`: Detailed results for all experiments
- `sentiment_distribution.png`: Sentiment distribution visualization
- `svm_performance_comparison.png`: Performance comparison plots
- `confusion_matrices_best_models.png`: Confusion matrices for best models

## Dependencies

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
gensim>=4.0.0
```

## Key Research Questions

1. **Which feature extraction method performs better?** TF-IDF vs Word2Vec
2. **Which sentiment labeling is most reliable?** Score-based vs TextBlob vs VADER vs Ensemble
3. **What's the optimal SVM kernel?** Linear vs RBF vs Polynomial vs Sigmoid
4. **How does training size affect performance?** 65% vs 70% vs 75%
5. **Which combinations work best together?** Feature × Method × Kernel interactions

## Expected Insights

- **Feature Performance**: TF-IDF typically performs well for sentiment analysis
- **Kernel Selection**: RBF and Linear kernels often show good performance
- **Training Size**: Larger training sets generally improve performance
- **Sentiment Method**: Ensemble methods may provide more balanced results
- **Class Imbalance**: Some sentiment methods show significant class imbalance

## Future Enhancements

1. **Deep Learning Models**: LSTM, BERT, or transformer-based approaches
2. **Feature Engineering**: Additional linguistic features, POS tags, etc.
3. **Hyperparameter Tuning**: Grid search for optimal SVM parameters
4. **Cross-Validation**: K-fold cross-validation for robust evaluation
5. **Aspect Extraction**: Identify specific aspects being discussed in reviews

## Contact and Support

For questions or issues with this analysis, please refer to the documentation or create an issue in the project repository.
