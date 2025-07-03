#!/usr/bin/env python3
"""
Script to analyze KBBI dataset and extract useful Indonesian vocabulary
for enhancing our multilingual stopwords dataset.
"""

import pandas as pd
import requests
from io import StringIO
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_kbbi_data():
    """Download KBBI dataset from GitHub"""
    url = "https://raw.githubusercontent.com/aryakdaniswara/kbbi-dataset-kbbi-v/refs/heads/main/csv/kbbi_v.csv"
    
    try:
        logging.info("Downloading KBBI dataset...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Read CSV data
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        logging.info(f"Successfully downloaded KBBI dataset with {len(df)} entries")
        return df
        
    except Exception as e:
        logging.error(f"Error downloading KBBI dataset: {e}")
        return None

def analyze_kbbi_structure(df):
    """Analyze the structure of KBBI dataset"""
    logging.info("Analyzing KBBI dataset structure...")
    
    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    # Check for key columns
    key_columns = ['nama', 'kata_dasar', 'kelas', 'contoh']
    available_columns = [col for col in key_columns if col in df.columns]
    print(f"\nKey columns available: {available_columns}")
    
    return df

def extract_common_words(df):
    """Extract common Indonesian words that could be useful as stopwords"""
    logging.info("Extracting common Indonesian words...")
    
    # Get the main word column (likely 'nama' or 'kata_dasar')
    if 'nama' in df.columns:
        words_column = 'nama'
    elif 'kata_dasar' in df.columns:
        words_column = 'kata_dasar'
    else:
        logging.error("Cannot find main word column")
        return []
    
    # Extract words
    words = df[words_column].dropna().str.strip().tolist()
    
    # Filter for common function words and particles
    common_patterns = [
        r'^(dan|atau|tetapi|namun|karena|sebab|jika|kalau|ketika|saat|waktu)$',
        r'^(di|ke|dari|untuk|dengan|pada|dalam|oleh|atas|bawah|antara)$',
        r'^(ini|itu|yang|adalah|akan|sudah|sedang|masih|belum|tidak|bukan)$',
        r'^(saya|aku|kamu|dia|mereka|kita|kami)$',
        r'^(ada|semua|setiap|beberapa|banyak|sedikit|sangat|agak|cukup)$',
        r'^(lah|kah|pun|nya|mu|ku)$',
        r'^(juga|hanya|saja|pula|bahkan|malah|justru)$'
    ]
    
    filtered_words = []
    for word in words:
        if isinstance(word, str) and len(word) > 0:
            # Check if word matches common patterns
            for pattern in common_patterns:
                if re.match(pattern, word.lower()):
                    filtered_words.append(word.lower())
                    break
    
    # Remove duplicates and sort
    filtered_words = sorted(list(set(filtered_words)))
    
    logging.info(f"Extracted {len(filtered_words)} common words")
    return filtered_words

def extract_particles_and_affixes(df):
    """Extract Indonesian particles and affixes"""
    logging.info("Extracting particles and affixes...")
    
    particles_affixes = []
    
    if 'nama' in df.columns:
        # Look for entries that start with - (affixes) or are particles
        affixes = df[df['nama'].str.startswith('-', na=False)]['nama'].tolist()
        particles_affixes.extend(affixes)
        
        # Look for particles in kelas column
        if 'kelas' in df.columns:
            particle_entries = df[df['kelas'].str.contains('Partikel', na=False)]['nama'].tolist()
            particles_affixes.extend(particle_entries)
    
    # Clean and filter
    cleaned = []
    for item in particles_affixes:
        if isinstance(item, str) and len(item.strip()) > 0:
            cleaned.append(item.strip())
    
    particles_affixes = sorted(list(set(cleaned)))
    
    logging.info(f"Extracted {len(particles_affixes)} particles and affixes")
    return particles_affixes

def extract_formal_vocabulary(df):
    """Extract formal Indonesian vocabulary"""
    logging.info("Extracting formal vocabulary...")
    
    formal_words = []
    
    if 'nama' in df.columns and 'kelas' in df.columns:
        # Look for formal words (nouns, verbs, adjectives that are commonly used)
        formal_classes = ['Nomina', 'Verba', 'Adjektiva', 'Adverbia']
        
        for word_class in formal_classes:
            class_words = df[df['kelas'].str.contains(word_class, na=False)]['nama'].tolist()
            
            # Filter for shorter, more common words
            for word in class_words:
                if isinstance(word, str) and 2 <= len(word) <= 8:
                    formal_words.append(word.lower())
    
    # Remove duplicates and sort
    formal_words = sorted(list(set(formal_words)))
    
    logging.info(f"Extracted {len(formal_words)} formal vocabulary words")
    return formal_words[:200]  # Limit to top 200

def save_extracted_words(common_words, particles, formal_words):
    """Save extracted words to files"""
    logging.info("Saving extracted words...")
    
    # Save common words
    with open('kbbi_common_words.txt', 'w', encoding='utf-8') as f:
        for word in common_words:
            f.write(f"{word}\n")
    
    # Save particles and affixes
    with open('kbbi_particles_affixes.txt', 'w', encoding='utf-8') as f:
        for item in particles:
            f.write(f"{item}\n")
    
    # Save formal vocabulary
    with open('kbbi_formal_vocabulary.txt', 'w', encoding='utf-8') as f:
        for word in formal_words:
            f.write(f"{word}\n")
    
    logging.info("Files saved successfully")

def main():
    """Main function"""
    # Download KBBI dataset
    df = download_kbbi_data()
    if df is None:
        return
    
    # Analyze structure
    df = analyze_kbbi_structure(df)
    
    # Extract different types of words
    common_words = extract_common_words(df)
    particles = extract_particles_and_affixes(df)
    formal_words = extract_formal_vocabulary(df)
    
    # Save results
    save_extracted_words(common_words, particles, formal_words)
    
    # Print summary
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Common words extracted: {len(common_words)}")
    print(f"Particles/affixes extracted: {len(particles)}")
    print(f"Formal vocabulary extracted: {len(formal_words)}")
    
    print(f"\nSample common words: {common_words[:10]}")
    print(f"Sample particles: {particles[:10]}")
    print(f"Sample formal words: {formal_words[:10]}")

if __name__ == "__main__":
    main()
