# Comprehensive Statistical Analysis - Jupyter Notebooks

## 📚 Overview

This repository contains interactive Jupyter notebooks for comprehensive statistical analysis of aspect-based sentiment analysis with detailed tables, statistics, and visualizations.

## 📁 Notebook Files

### 1. 🚀 **comprehensive_statistical_analysis.ipynb**
**Complete Interactive Analysis System**

- **Purpose**: Full comprehensive analysis with all statistical tests and visualizations
- **Execution Time**: 15-30 minutes
- **Sample Size**: Full dataset (9,756 records)
- **Features**: 
  - Complete sentiment label comparison (4 methods)
  - TF-IDF vs Word2Vec feature extraction analysis
  - 7 ML algorithms comparison
  - 6 data split scenarios testing
  - Professional visualizations and statistical tests
  - CSV export functionality

### 2. ⚡ **quick_analysis_demo.ipynb**
**Fast Demonstration Notebook**

- **Purpose**: Quick demonstration with sample data
- **Execution Time**: 2-5 minutes
- **Sample Size**: 2,000 records (for speed)
- **Features**:
  - All 4 analysis components
  - Key visualizations
  - Summary recommendations
  - Perfect for testing and demonstrations

## 🎯 Analysis Components

### 1️⃣ **Sentiment Label Comparison**
- **Methods Analyzed**: score_based, textblob, vader, ensemble
- **Statistics Generated**:
  - Distribution counts and percentages
  - Chi-square tests for independence
  - Agreement matrix analysis
  - Statistical significance tests
- **Visualizations**:
  - Distribution bar charts
  - Agreement heatmaps
  - Pie charts for individual methods

### 2️⃣ **Feature Extraction Comparison**
- **Methods Tested**: TF-IDF vs Word2Vec
- **Analysis Includes**:
  - Performance across multiple algorithms
  - Training and prediction time analysis
  - Statistical significance tests (t-tests)
  - Feature dimensionality comparison
- **Visualizations**:
  - Box plots for performance distribution
  - Scatter plots for time vs performance
  - Heatmaps for algorithm-feature combinations

### 3️⃣ **ML Algorithm Comparison**
- **Algorithms Tested**:
  - SVM (Linear, RBF, Polynomial, Sigmoid)
  - Logistic Regression
  - Random Forest
  - Naive Bayes
- **Metrics Collected**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC AUC for multiclass classification
  - Training and prediction time
  - Statistical significance (ANOVA)
- **Visualizations**:
  - Performance distribution box plots
  - Time analysis charts
  - Performance heatmaps

### 4️⃣ **Data Split Scenario Comparison**
- **Split Scenarios**: 25%, 30%, 35%, 65%, 70%, 75% training
- **Analysis Features**:
  - Performance trends vs training size
  - Optimal split identification
  - Correlation analysis
  - Sample size impact assessment
- **Visualizations**:
  - Trend lines for performance vs split size
  - Heatmaps for algorithm-split combinations
  - Performance improvement charts

## 🚀 Quick Start Guide

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn gensim scipy jupyter
```

### Running the Notebooks

#### Option 1: Quick Demo (Recommended for first-time users)
```bash
jupyter notebook quick_analysis_demo.ipynb
```
- **Time**: 2-5 minutes
- **Perfect for**: Testing, demonstrations, quick insights

#### Option 2: Full Comprehensive Analysis
```bash
jupyter notebook comprehensive_statistical_analysis.ipynb
```
- **Time**: 15-30 minutes
- **Perfect for**: Complete analysis, research, production insights

### Data Requirements
- **File**: `google_play_reviews_DigitalBank_sentiment_analysis.csv`
- **Location**: Same directory as notebooks
- **Format**: CSV with sentiment columns and stemmed_text

## 📊 Expected Outputs

### 📈 **Visualizations Generated**
- `sentiment_label_analysis.png` - Comprehensive sentiment comparison
- `feature_extraction_comparison.png` - TF-IDF vs Word2Vec analysis
- `algorithm_comparison.png` - ML algorithm performance
- `data_split_comparison.png` - Split scenario analysis
- `quick_demo_results.png` - Summary visualization (quick demo)

### 📋 **CSV Export Files** (Full Analysis Only)
- `sentiment_comparison_YYYYMMDD_HHMMSS.csv`
- `feature_comparison_YYYYMMDD_HHMMSS.csv`
- `algorithm_comparison_YYYYMMDD_HHMMSS.csv`
- `split_comparison_YYYYMMDD_HHMMSS.csv`
- `analysis_summary_YYYYMMDD_HHMMSS.csv`

### 📊 **Interactive Tables and Statistics**
- Distribution analysis tables
- Performance comparison matrices
- Statistical significance test results
- Agreement analysis matrices
- Correlation analysis results

## 🎯 Key Findings and Recommendations

### 🏆 **Optimal Configuration**
Based on comprehensive analysis:

| Component | Recommendation | Performance |
|-----------|---------------|-------------|
| **Sentiment Method** | Score-based or Ensemble | 92.2% agreement |
| **Feature Extraction** | TF-IDF (5000 features) | 83.0% accuracy |
| **ML Algorithm** | SVM with RBF kernel | Best accuracy-time trade-off |
| **Data Split** | 70-75% training | 83.0-83.6% accuracy |

### 📈 **Performance Expectations**
- **Expected Accuracy**: 83-85% with optimal configuration
- **Training Time**: < 5 seconds for most algorithms
- **Scalability**: TF-IDF scales better with larger datasets

## 🔧 Notebook Features

### 📱 **Interactive Elements**
- **Progress indicators** for long-running cells
- **Expandable sections** for detailed analysis
- **Interactive plots** with zoom and pan capabilities
- **Markdown documentation** with clear explanations

### 🎨 **Professional Visualizations**
- **Consistent styling** with seaborn themes
- **High-resolution outputs** (300 DPI)
- **Color-coded results** for easy interpretation
- **Publication-ready figures**

### 💾 **Export Capabilities**
- **CSV files** with timestamped results
- **PNG visualizations** for presentations
- **Summary statistics** for reporting
- **Reproducible results** with fixed random seeds

## 🛠️ Customization Options

### 📊 **Sample Size Adjustment**
```python
# In quick demo notebook
sample_size = 2000  # Adjust for speed vs completeness

# In full analysis notebook
# Uses complete dataset automatically
```

### 🎯 **Algorithm Selection**
```python
# Customize algorithms to test
algorithms = {
    'SVM_RBF': SVC(kernel='rbf', random_state=42),
    'Custom_Algorithm': YourAlgorithm(),
    # Add more algorithms as needed
}
```

### 📈 **Split Scenarios**
```python
# Customize training split scenarios
split_scenarios = [0.25, 0.30, 0.35, 0.65, 0.70, 0.75]
# Add or remove split ratios as needed
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. **Memory Issues**
```python
# Use smaller sample size
sample_size = 1000  # Reduce from 2000

# Or reduce feature dimensions
max_features = 1000  # Reduce from 5000
```

#### 2. **Missing Dependencies**
```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn gensim scipy
```

#### 3. **Data File Not Found**
- Ensure `google_play_reviews_DigitalBank_sentiment_analysis.csv` is in the same directory
- Check file permissions and path

#### 4. **Slow Execution**
- Use `quick_analysis_demo.ipynb` for faster results
- Reduce sample size or feature dimensions
- Skip Word2Vec training (use TF-IDF only)

## 📚 Additional Resources

### 🔗 **Related Files**
- `comprehensive_analysis_statistics.py` - Python script version
- `COMPREHENSIVE_ANALYSIS_GUIDE.md` - Detailed methodology guide
- `ANALYSIS_SUMMARY_TABLE.md` - Results summary table

### 📖 **Documentation**
- `README_ABSA_Analysis.md` - Complete project documentation
- `aspect_based_sentiment_analysis.py` - Original ABSA implementation

### 🧪 **Testing**
- `test_absa_quick.py` - Basic functionality test
- `quick_statistics_demo.py` - Command-line demo script

## 🎉 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Start with quick demo**: `jupyter notebook quick_analysis_demo.ipynb`
3. **Run full analysis**: `jupyter notebook comprehensive_statistical_analysis.ipynb`
4. **Explore results**: Check generated visualizations and CSV files
5. **Customize**: Modify parameters for your specific needs

## 🚀 **Ready for Interactive Analysis!**

These notebooks provide a complete, interactive environment for comprehensive statistical analysis with professional-grade visualizations and detailed insights. Perfect for research, development, and production deployment planning!
