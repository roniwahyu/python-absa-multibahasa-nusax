#!/usr/bin/env python3
"""
Quick analysis of the enhanced stopwords dataset
"""

import pandas as pd

def analyze_enhanced_dataset():
    """Analyze the enhanced stopwords dataset"""
    
    # Load both datasets
    original_df = pd.read_csv('multilingual_stopwords_comprehensive.csv')
    enhanced_df = pd.read_csv('multilingual_stopwords_enhanced.csv')
    
    print("=== STOPWORDS ENHANCEMENT ANALYSIS ===")
    print(f"Original dataset: {len(original_df)} entries")
    print(f"Enhanced dataset: {len(enhanced_df)} entries")
    print(f"New entries added: {len(enhanced_df) - len(original_df)}")
    
    # Count by language for original
    print(f"\nOriginal language statistics:")
    print(f"- English entries: {original_df['en'].notna().sum()}")
    print(f"- Indonesian entries: {original_df['id'].notna().sum()}")
    print(f"- Javanese entries: {original_df['jv'].notna().sum()}")
    print(f"- Sundanese entries: {original_df['su'].notna().sum()}")
    print(f"- Formal Indonesian entries: {original_df['formal_id'].notna().sum()}")
    
    # Count by language for enhanced
    print(f"\nEnhanced language statistics:")
    print(f"- English entries: {enhanced_df['en'].notna().sum()}")
    print(f"- Indonesian entries: {enhanced_df['id'].notna().sum()}")
    print(f"- Javanese entries: {enhanced_df['jv'].notna().sum()}")
    print(f"- Sundanese entries: {enhanced_df['su'].notna().sum()}")
    print(f"- Formal Indonesian entries: {enhanced_df['formal_id'].notna().sum()}")
    
    # Show some new entries
    new_entries_start = len(original_df)
    new_entries = enhanced_df.iloc[new_entries_start:]
    
    print(f"\nSample of new Indonesian entries added:")
    sample_new = new_entries['id'].dropna().head(20).tolist()
    for i, word in enumerate(sample_new, 1):
        print(f"  {i:2d}. {word}")
    
    print(f"\n✅ Enhanced dataset successfully created!")
    print(f"📊 Total entries: {len(enhanced_df)}")
    print(f"📁 Saved as: multilingual_stopwords_enhanced.csv")

if __name__ == "__main__":
    analyze_enhanced_dataset()
