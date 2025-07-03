#!/usr/bin/env python3
"""
Script to integrate colloquial Indonesian vocabulary into our enhanced stopwords dataset.
"""

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_colloquial_candidates():
    """Load colloquial stopword candidates"""
    try:
        candidates_df = pd.read_csv('colloquial_stopword_candidates.csv')
        logging.info(f"Loaded {len(candidates_df)} colloquial candidates")
        return candidates_df
    except FileNotFoundError:
        logging.error("Colloquial candidates file not found!")
        return None

def filter_high_value_candidates(candidates_df):
    """Filter for high-value colloquial stopwords"""
    
    # Priority categories for stopwords
    high_priority_categories = ['particles', 'pronouns', 'conjunctions', 'adverbs', 'question_words', 'demonstratives']
    
    # Load category-specific files
    high_value_candidates = []
    
    for category in high_priority_categories:
        try:
            category_df = pd.read_csv(f'colloquial_{category}.csv')
            
            # Filter based on category-specific criteria
            if category == 'particles':
                # Include short particles and common function words
                filtered = category_df[
                    (category_df['length'] <= 3) | 
                    (category_df['formal'].isin(['yang', 'saja', 'juga', 'hanya', 'dengan', 'untuk', 'dari', 'pada']))
                ]
            elif category == 'pronouns':
                # Include all pronouns
                filtered = category_df
            elif category == 'conjunctions':
                # Include all conjunctions
                filtered = category_df
            elif category == 'adverbs':
                # Focus on intensity adverbs
                filtered = category_df[
                    category_df['formal'].isin(['banget', 'sangat', 'sekali', 'agak', 'cukup', 'terlalu'])
                ]
            elif category == 'question_words':
                # Include all question words
                filtered = category_df
            elif category == 'demonstratives':
                # Include all demonstratives
                filtered = category_df
            
            # Add category label
            filtered = filtered.copy()
            filtered['priority_category'] = category
            high_value_candidates.append(filtered)
            
            logging.info(f"Selected {len(filtered)} candidates from {category}")
            
        except FileNotFoundError:
            logging.warning(f"Category file for {category} not found")
    
    # Combine all high-value candidates
    if high_value_candidates:
        combined_df = pd.concat(high_value_candidates, ignore_index=True)
        logging.info(f"Total high-value candidates: {len(combined_df)}")
        return combined_df
    else:
        return pd.DataFrame()

def integrate_with_enhanced_stopwords(colloquial_df):
    """Integrate colloquial vocabulary with enhanced stopwords"""
    
    # Load existing enhanced stopwords
    try:
        existing_df = pd.read_csv('multilingual_stopwords_enhanced.csv')
        logging.info(f"Loaded existing enhanced stopwords: {len(existing_df)} entries")
    except FileNotFoundError:
        logging.error("Enhanced stopwords file not found!")
        return
    
    # Prepare new entries from colloquial data
    new_entries = []
    existing_indonesian = set(existing_df['id'].dropna().str.strip().str.lower())
    existing_formal = set(existing_df['formal_id'].dropna().str.strip().str.lower())
    
    added_count = 0
    skipped_count = 0
    
    for _, row in colloquial_df.iterrows():
        slang = str(row['slang']).strip().lower()
        formal = str(row['formal']).strip().lower()
        
        # Skip if either is invalid
        if slang == 'nan' or formal == 'nan' or len(slang) == 0 or len(formal) == 0:
            continue
        
        # Skip if slang already exists in dataset
        if slang in existing_indonesian or slang in existing_formal:
            skipped_count += 1
            continue
        
        # Add slang term as Indonesian colloquial
        new_entry = {
            'en': '',
            'id': slang,
            'jv': '',
            'su': '',
            'formal_id': formal
        }
        new_entries.append(new_entry)
        added_count += 1
    
    if new_entries:
        # Create DataFrame for new entries
        new_df = pd.DataFrame(new_entries)
        
        # Combine with existing data
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Save final dataset
        final_df.to_csv('multilingual_stopwords_final.csv', index=False)
        
        logging.info(f"Final dataset saved with {len(final_df)} total entries")
        logging.info(f"Added {added_count} new colloquial stopwords")
        logging.info(f"Skipped {skipped_count} existing words")
        
        return final_df, added_count, skipped_count
    else:
        logging.info("No new words to add")
        return existing_df, 0, 0

def create_integration_summary(final_df, added_count, skipped_count, colloquial_df):
    """Create summary of the integration process"""
    
    # Count by language
    en_count = final_df['en'].notna().sum()
    id_count = final_df['id'].notna().sum()
    jv_count = final_df['jv'].notna().sum()
    su_count = final_df['su'].notna().sum()
    formal_id_count = final_df['formal_id'].notna().sum()
    
    # Create summary
    summary = f"""
COLLOQUIAL INDONESIAN INTEGRATION SUMMARY
=========================================

Dataset Evolution:
- Original comprehensive: 1,617 entries
- Enhanced with KBBI: 1,755 entries  
- Final with colloquial: {len(final_df)} entries
- New colloquial additions: {added_count}
- Skipped (already exist): {skipped_count}

Final Language Statistics:
- English entries: {en_count}
- Indonesian entries: {id_count}
- Javanese entries: {jv_count}
- Sundanese entries: {su_count}
- Formal Indonesian entries: {formal_id_count}

Colloquial Categories Integrated:
"""
    
    # Add category breakdown
    if 'priority_category' in colloquial_df.columns:
        category_counts = colloquial_df['priority_category'].value_counts()
        for category, count in category_counts.items():
            summary += f"- {category.title()}: {count} terms\n"
    
    summary += f"""
Sample New Colloquial Additions:
"""
    
    # Show sample new entries (last added entries)
    if added_count > 0:
        sample_entries = final_df.tail(min(20, added_count))
        for i, (_, row) in enumerate(sample_entries.iterrows(), 1):
            if pd.notna(row['id']) and row['id'] != '':
                summary += f"  {i:2d}. {row['id']} -> {row['formal_id']}\n"
    
    summary += f"""
Quality Improvements:
✅ Enhanced social media text preprocessing
✅ Better coverage of Indonesian internet slang
✅ Improved colloquial language support
✅ More comprehensive particle and pronoun coverage
✅ Enhanced question word and demonstrative support

Usage Recommendations:
- Use for social media sentiment analysis
- Apply to informal Indonesian text processing
- Suitable for chat and messaging data
- Ideal for youth-oriented content analysis
"""
    
    return summary

def main():
    """Main function"""
    # Load colloquial candidates
    candidates_df = load_colloquial_candidates()
    if candidates_df is None:
        return
    
    # Filter high-value candidates
    high_value_df = filter_high_value_candidates(candidates_df)
    if high_value_df.empty:
        logging.error("No high-value candidates found")
        return
    
    # Integrate with enhanced stopwords
    final_df, added_count, skipped_count = integrate_with_enhanced_stopwords(high_value_df)
    
    # Create and save summary
    summary = create_integration_summary(final_df, added_count, skipped_count, high_value_df)
    
    with open('colloquial_integration_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Print summary
    print(summary)
    
    print(f"\n✅ Integration complete!")
    print(f"📊 Final dataset: {len(final_df)} entries")
    print(f"📁 Saved as: multilingual_stopwords_final.csv")

if __name__ == "__main__":
    main()
