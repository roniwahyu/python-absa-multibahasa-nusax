#!/usr/bin/env python3
"""
Script to enhance our multilingual stopwords with additional Indonesian vocabulary
based on KBBI patterns and common Indonesian linguistic elements.
"""

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_additional_indonesian_stopwords():
    """Get additional Indonesian stopwords based on KBBI patterns"""
    
    # Common Indonesian function words and particles from KBBI analysis
    additional_stopwords = [
        # Pronouns and demonstratives
        'beliau', 'mereka', 'kita', 'kami', 'kalian', 'engkau', 'anda',
        'ini', 'itu', 'tersebut', 'demikian', 'begini', 'begitu',
        
        # Conjunctions and connectors
        'dan', 'atau', 'tetapi', 'namun', 'akan tetapi', 'melainkan',
        'karena', 'sebab', 'oleh karena', 'akibat', 'sehingga',
        'jika', 'kalau', 'apabila', 'bila', 'seandainya', 'andai',
        'ketika', 'saat', 'waktu', 'sewaktu', 'tatkala', 'manakala',
        'sebelum', 'sesudah', 'setelah', 'hingga', 'sampai',
        'selama', 'sambil', 'seraya', 'sementara',
        'agar', 'supaya', 'biar', 'untuk',
        'meskipun', 'walaupun', 'sekalipun', 'kendatipun',
        
        # Prepositions
        'di', 'ke', 'dari', 'pada', 'dalam', 'oleh', 'dengan',
        'atas', 'bawah', 'antara', 'diantara', 'mengenai', 'tentang',
        'terhadap', 'kepada', 'bagi', 'untuk', 'tanpa', 'kecuali',
        'selain', 'hingga', 'sampai', 'sejak', 'semenjak',
        
        # Auxiliary verbs and modals
        'adalah', 'ialah', 'yaitu', 'yakni', 'merupakan',
        'akan', 'telah', 'sudah', 'sedang', 'tengah', 'lagi',
        'masih', 'belum', 'pernah', 'tidak pernah',
        'dapat', 'bisa', 'mampu', 'sanggup', 'mau', 'ingin',
        'harus', 'mesti', 'wajib', 'perlu', 'hendak',
        
        # Negations
        'tidak', 'tak', 'bukan', 'bukanlah', 'jangan', 'janganlah',
        'tanpa', 'minus', 'kecuali', 'selain',
        
        # Quantifiers and determiners
        'ada', 'tidak ada', 'semua', 'seluruh', 'segenap', 'segala',
        'setiap', 'tiap', 'masing-masing', 'para', 'kaum',
        'beberapa', 'sebagian', 'separuh', 'setengah',
        'banyak', 'sedikit', 'sejumlah', 'sekelompok',
        'satu', 'dua', 'tiga', 'empat', 'lima',
        'pertama', 'kedua', 'ketiga', 'terakhir',
        
        # Intensifiers and adverbs
        'sangat', 'amat', 'sekali', 'banget', 'betul-betul',
        'benar-benar', 'sungguh', 'sungguh-sungguh',
        'agak', 'sedikit', 'rada', 'cukup', 'lumayan',
        'terlalu', 'kelewat', 'kebangetan',
        'paling', 'lebih', 'kurang', 'hampir', 'nyaris',
        'kira-kira', 'sekitar', 'kurang lebih', 'lebih kurang',
        
        # Particles and discourse markers
        'lah', 'kah', 'pun', 'nya', 'mu', 'ku',
        'juga', 'pula', 'pun', 'saja', 'hanya', 'cuma',
        'bahkan', 'malah', 'justru', 'sebaliknya',
        'memang', 'emang', 'indeed', 'tentu', 'pasti',
        'mungkin', 'barangkali', 'kiranya', 'agaknya',
        'rupanya', 'ternyata', 'nampaknya', 'sepertinya',
        
        # Time expressions
        'sekarang', 'kini', 'saat ini', 'dewasa ini',
        'dulu', 'dahulu', 'tempo dulu', 'masa lalu',
        'nanti', 'kelak', 'esok', 'besok', 'lusa',
        'kemarin', 'kemaren', 'tadi', 'barusan',
        'selalu', 'senantiasa', 'terus', 'terus-menerus',
        'kadang', 'kadang-kadang', 'terkadang', 'sesekali',
        'jarang', 'sering', 'kerap', 'acap',
        
        # Spatial expressions
        'sini', 'sana', 'situ', 'dimana', 'kemana', 'darimana',
        'disini', 'disana', 'disitu', 'kesini', 'kesana', 'kesitu',
        'dekat', 'jauh', 'sekitar', 'sekeliling',
        'depan', 'belakang', 'samping', 'sebelah',
        'atas', 'bawah', 'tengah', 'pinggir', 'tepi',
        
        # Question words
        'apa', 'siapa', 'mana', 'dimana', 'kemana', 'darimana',
        'kapan', 'mengapa', 'kenapa', 'bagaimana', 'gimana',
        'berapa', 'seberapa', 'yang mana', 'siapa saja',
        
        # Formal/literary words
        'adapun', 'sedangkan', 'sementara itu', 'lagi pula',
        'tambahan pula', 'selanjutnya', 'kemudian', 'lalu',
        'akhirnya', 'pada akhirnya', 'kesimpulannya',
        'dengan demikian', 'oleh karena itu', 'maka dari itu',
        'sebaliknya', 'sebaliknya', 'walaupun demikian',
        
        # Common verbs that function as auxiliaries
        'menjadi', 'menjadikan', 'membuat', 'membentuk',
        'menggunakan', 'memakai', 'memanfaatkan',
        'memberikan', 'memberi', 'menyediakan',
        'mengambil', 'memperoleh', 'mendapat', 'mendapatkan',
        'melakukan', 'menjalankan', 'menjalani',
        'memiliki', 'mempunyai', 'punya',
        
        # Courtesy and politeness markers
        'mohon', 'tolong', 'silakan', 'silahkan', 'mari',
        'coba', 'cobalah', 'harap', 'kiranya',
        'maaf', 'permisi', 'terima kasih', 'makasih',
        'sama-sama', 'kembali', 'selamat',
        
        # Emphasis and confirmation
        'kok', 'sih', 'dong', 'deh', 'nih', 'tuh',
        'kan', 'ya', 'iya', 'yah', 'nah', 'wah',
        'lho', 'lo', 'tho', 'to', 'gitu', 'gini',
        
        # Colloquial particles
        'aja', 'doang', 'kok', 'sih', 'deh', 'dong',
        'nih', 'tuh', 'gitu', 'gini', 'kayak', 'kaya',
        'udah', 'udeh', 'belom', 'blom', 'gimana', 'gmn',
        'kenapa', 'knp', 'emang', 'memang', 'bener', 'benar'
    ]
    
    return additional_stopwords

def enhance_comprehensive_stopwords():
    """Enhance our comprehensive stopwords with KBBI-based vocabulary"""
    
    # Load existing comprehensive stopwords
    try:
        existing_df = pd.read_csv('multilingual_stopwords_comprehensive.csv')
        logging.info(f"Loaded existing stopwords: {len(existing_df)} entries")
    except FileNotFoundError:
        logging.error("Comprehensive stopwords file not found!")
        return
    
    # Get additional Indonesian stopwords
    additional_words = get_additional_indonesian_stopwords()
    logging.info(f"Additional Indonesian words to add: {len(additional_words)}")
    
    # Create new entries
    new_entries = []
    existing_indonesian = set(existing_df['id'].dropna().str.strip().str.lower())
    existing_formal = set(existing_df['formal_id'].dropna().str.strip().str.lower())
    
    added_count = 0
    for word in additional_words:
        word = word.strip().lower()
        
        # Skip if already exists
        if word in existing_indonesian or word in existing_formal:
            continue
            
        # Add as both colloquial and formal Indonesian
        new_entry = {
            'en': '',
            'id': word,
            'jv': '',
            'su': '',
            'formal_id': word
        }
        new_entries.append(new_entry)
        added_count += 1
    
    if new_entries:
        # Create DataFrame for new entries
        new_df = pd.DataFrame(new_entries)
        
        # Combine with existing data
        enhanced_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Save enhanced dataset
        enhanced_df.to_csv('multilingual_stopwords_enhanced.csv', index=False)
        
        logging.info(f"Enhanced dataset saved with {len(enhanced_df)} total entries")
        logging.info(f"Added {added_count} new Indonesian stopwords")
        
        # Print statistics
        print(f"\n=== ENHANCEMENT SUMMARY ===")
        print(f"Original entries: {len(existing_df)}")
        print(f"New entries added: {added_count}")
        print(f"Total entries: {len(enhanced_df)}")
        
        # Count by language
        en_count = enhanced_df['en'].notna().sum()
        id_count = enhanced_df['id'].notna().sum()
        jv_count = enhanced_df['jv'].notna().sum()
        su_count = enhanced_df['su'].notna().sum()
        formal_id_count = enhanced_df['formal_id'].notna().sum()
        
        print(f"\nLanguage statistics:")
        print(f"- English entries: {en_count}")
        print(f"- Indonesian entries: {id_count}")
        print(f"- Javanese entries: {jv_count}")
        print(f"- Sundanese entries: {su_count}")
        print(f"- Formal Indonesian entries: {formal_id_count}")
        
        # Show sample of new entries
        print(f"\nSample new entries:")
        for i, entry in enumerate(new_entries[:10]):
            print(f"  {i+1}. {entry['id']}")
            
    else:
        logging.info("No new words to add - all words already exist in dataset")

def main():
    """Main function"""
    enhance_comprehensive_stopwords()

if __name__ == "__main__":
    main()
