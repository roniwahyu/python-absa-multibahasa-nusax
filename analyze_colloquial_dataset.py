#!/usr/bin/env python3
"""
Script to analyze the colloquial Indonesian dataset and extract useful vocabulary
for enhancing our multilingual stopwords dataset.
"""

import pandas as pd
import requests
from io import StringIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_colloquial_data():
    """Download colloquial Indonesian dataset"""
    url = "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/refs/heads/master/kamus/nasalsabila_kamus-alay/colloquial-indonesian-lexicon.csv"
    
    try:
        logging.info("Downloading colloquial Indonesian dataset...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Read CSV data
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        logging.info(f"Successfully downloaded colloquial dataset with {len(df)} entries")
        return df
        
    except Exception as e:
        logging.error(f"Error downloading colloquial dataset: {e}")
        return None

def analyze_colloquial_structure(df):
    """Analyze the structure of colloquial dataset"""
    logging.info("Analyzing colloquial dataset structure...")
    
    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    # Check categories
    if 'category1' in df.columns:
        print("\nCategory1 distribution:")
        print(df['category1'].value_counts().head(10))
    
    return df

def extract_stopword_candidates(df):
    """Extract colloquial words that could be useful as stopwords"""
    logging.info("Extracting stopword candidates...")
    
    # Focus on common function words, particles, and frequently used colloquial terms
    stopword_candidates = []
    
    # Get slang-formal pairs
    if 'slang' in df.columns and 'formal' in df.columns:
        for _, row in df.iterrows():
            slang = str(row['slang']).strip().lower()
            formal = str(row['formal']).strip().lower()
            
            # Skip if either is NaN or empty
            if slang == 'nan' or formal == 'nan' or len(slang) == 0 or len(formal) == 0:
                continue
            
            # Focus on short, common words (likely to be function words)
            if len(slang) <= 6 and len(formal) <= 10:
                # Check if formal word suggests it's a function word
                function_word_indicators = [
                    'yang', 'dengan', 'untuk', 'dari', 'pada', 'dalam', 'oleh',
                    'akan', 'sudah', 'sedang', 'masih', 'belum', 'tidak', 'bukan',
                    'saya', 'aku', 'kamu', 'dia', 'mereka', 'kita', 'kami',
                    'ini', 'itu', 'adalah', 'juga', 'hanya', 'saja', 'pula',
                    'dan', 'atau', 'tetapi', 'karena', 'jika', 'ketika',
                    'sangat', 'banget', 'sekali', 'agak', 'cukup', 'terlalu',
                    'bisa', 'dapat', 'mau', 'ingin', 'harus', 'perlu',
                    'sama', 'seperti', 'kayak', 'kaya', 'begitu', 'gitu',
                    'dimana', 'kemana', 'bagaimana', 'gimana', 'kenapa',
                    'kapan', 'siapa', 'apa', 'mana', 'berapa'
                ]
                
                # Check if formal word is a function word
                found_match = False
                for indicator in function_word_indicators:
                    if indicator in formal or formal in indicator:
                        stopword_candidates.append({
                            'slang': slang,
                            'formal': formal,
                            'category': row.get('category1', ''),
                            'length': len(slang)
                        })
                        found_match = True
                        break

                # Also include very short words (likely particles)
                if not found_match and len(slang) <= 3:
                    stopword_candidates.append({
                        'slang': slang,
                        'formal': formal,
                        'category': row.get('category1', ''),
                        'length': len(slang)
                    })
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for candidate in stopword_candidates:
        key = (candidate['slang'], candidate['formal'])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    
    logging.info(f"Extracted {len(unique_candidates)} stopword candidates")
    return unique_candidates

def categorize_candidates(candidates):
    """Categorize the stopword candidates"""
    logging.info("Categorizing stopword candidates...")
    
    categories = {
        'particles': [],
        'pronouns': [],
        'conjunctions': [],
        'adverbs': [],
        'question_words': [],
        'demonstratives': [],
        'others': []
    }
    
    for candidate in candidates:
        slang = candidate['slang']
        formal = candidate['formal']
        
        # Categorize based on formal word
        if formal in ['saya', 'aku', 'kamu', 'dia', 'mereka', 'kita', 'kami']:
            categories['pronouns'].append(candidate)
        elif formal in ['ini', 'itu', 'begitu', 'demikian']:
            categories['demonstratives'].append(candidate)
        elif formal in ['dan', 'atau', 'tetapi', 'karena', 'jika', 'ketika']:
            categories['conjunctions'].append(candidate)
        elif formal in ['sangat', 'banget', 'sekali', 'agak', 'cukup', 'terlalu']:
            categories['adverbs'].append(candidate)
        elif formal in ['dimana', 'kemana', 'bagaimana', 'kenapa', 'kapan', 'siapa', 'apa', 'mana', 'berapa']:
            categories['question_words'].append(candidate)
        elif len(slang) <= 3 or formal in ['juga', 'hanya', 'saja', 'pula']:
            categories['particles'].append(candidate)
        else:
            categories['others'].append(candidate)
    
    return categories

def save_analysis_results(candidates, categories):
    """Save analysis results to files"""
    logging.info("Saving analysis results...")
    
    # Save all candidates
    candidates_df = pd.DataFrame(candidates)
    candidates_df.to_csv('colloquial_stopword_candidates.csv', index=False)
    
    # Save by category
    for category, items in categories.items():
        if items:
            category_df = pd.DataFrame(items)
            category_df.to_csv(f'colloquial_{category}.csv', index=False)
    
    # Create summary
    with open('colloquial_analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("COLLOQUIAL INDONESIAN DATASET ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total stopword candidates: {len(candidates)}\n\n")
        
        f.write("By category:\n")
        for category, items in categories.items():
            f.write(f"- {category.title()}: {len(items)} items\n")
        
        f.write("\nSample candidates by category:\n")
        for category, items in categories.items():
            if items:
                f.write(f"\n{category.title()}:\n")
                for item in items[:10]:  # Show first 10
                    f.write(f"  {item['slang']} -> {item['formal']}\n")
    
    logging.info("Analysis results saved")

def main():
    """Main function"""
    # Download colloquial dataset
    df = download_colloquial_data()
    if df is None:
        return
    
    # Analyze structure
    df = analyze_colloquial_structure(df)
    
    # Extract stopword candidates
    candidates = extract_stopword_candidates(df)
    
    # Categorize candidates
    categories = categorize_candidates(candidates)
    
    # Save results
    save_analysis_results(candidates, categories)
    
    # Print summary
    print(f"\n=== COLLOQUIAL ANALYSIS SUMMARY ===")
    print(f"Total stopword candidates: {len(candidates)}")
    
    print(f"\nBy category:")
    for category, items in categories.items():
        print(f"- {category.title()}: {len(items)} items")
    
    print(f"\nSample high-value candidates:")
    # Show some examples
    for candidate in candidates[:20]:
        print(f"  {candidate['slang']} -> {candidate['formal']}")

if __name__ == "__main__":
    main()
