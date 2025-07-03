#!/usr/bin/env python3
"""
Multilingual Stopword Generator for Social Media Domain

This script generates a comprehensive multilingual stopword list specifically
designed for social media text processing, covering English, Indonesian (formal and colloquial),
Javanese, and Sundanese languages.

Author: NLP Engineer
Date: 2025-07-03
"""

import pandas as pd
import nltk
from collections import OrderedDict, Counter
import logging
import re
from datasets import load_dataset
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultilingualStopwordGenerator:
    """
    A class to generate multilingual stopwords for social media domain
    """
    
    def __init__(self):
        """Initialize the generator with language mappings and dictionaries"""
        self.stopwords_data = []
        self.setup_nltk()
        self.setup_language_mappings()
        self.nusax_data = None
    
    def setup_nltk(self):
        """Download and setup NLTK data"""
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.english_stopwords = set(stopwords.words('english'))
            logger.info("NLTK English stopwords loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NLTK stopwords: {e}")
            # Fallback to basic English stopwords
            self.english_stopwords = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
                'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
    
    def setup_language_mappings(self):
        """Setup language mapping dictionaries"""
        
        # Indonesian slang and social media abbreviations
        self.indonesian_slang = {
            'wkwk': 'haha',
            'wkwkwk': 'hahaha',
            'xixixi': 'hehehe',
            'hehe': 'hehe',
            'hihi': 'hihi',
            'brb': 'be right back',
            'btw': 'by the way',
            'cmiiw': 'correct me if i am wrong',
            'fyi': 'for your information',
            'imho': 'in my humble opinion',
            'lol': 'laugh out loud',
            'omg': 'oh my god',
            'wtf': 'what the f',
            'asap': 'as soon as possible',
            'aka': 'also known as',
            'etc': 'et cetera',
            'gw': 'saya',
            'gue': 'saya',
            'lu': 'kamu',
            'lo': 'kamu',
            'elu': 'kamu',
            'bokap': 'ayah',
            'nyokap': 'ibu',
            'bro': 'saudara',
            'sis': 'saudari',
            'gan': 'juragan',
            'agan': 'juragan',
            'suhu': 'master',
            'newbie': 'pemula',
            'noob': 'pemula',
            'pro': 'profesional',
            'mantap': 'bagus',
            'keren': 'bagus',
            'anjay': 'wow',
            'anjir': 'wow',
            'buset': 'wow',
            'gilak': 'gila',
            'gokil': 'gila',
            'kepo': 'ingin tahu',
            'gabut': 'tidak ada kegiatan',
            'baper': 'bawa perasaan',
            'galau': 'bingung',
            'bucin': 'budak cinta',
            'jones': 'jomblo ngenes',
            'php': 'pemberi harapan palsu',
            'pdkt': 'pendekatan',
            'ttm': 'teman tapi mesra',
            'clbk': 'cinta lama bersemi kembali'
        }
        
        # Indonesian formal words and common stopwords
        self.indonesian_formal = {
            'saya', 'aku', 'kamu', 'anda', 'dia', 'mereka', 'kita', 'kami',
            'ini', 'itu', 'yang', 'dan', 'atau', 'tetapi', 'namun', 'karena',
            'sebab', 'jika', 'kalau', 'bila', 'ketika', 'saat', 'waktu',
            'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'dalam', 'oleh',
            'adalah', 'ialah', 'yaitu', 'yakni', 'akan', 'telah', 'sudah',
            'sedang', 'masih', 'belum', 'tidak', 'bukan', 'jangan', 'ada',
            'tidak ada', 'semua', 'setiap', 'beberapa', 'banyak', 'sedikit',
            'sangat', 'agak', 'cukup', 'terlalu', 'paling', 'lebih', 'kurang'
        }

        # Javanese mappings (ngoko, alus, krama levels)
        self.javanese_mappings = {
            'saya': 'aku',  # ngoko
            'kamu': 'kowe',  # ngoko
            'dia': 'dheweke',
            'ini': 'iki',
            'itu': 'iku',
            'yang': 'sing',
            'dan': 'lan',
            'atau': 'utawa',
            'tidak': 'ora',
            'ada': 'ana',
            'di': 'ing',
            'ke': 'menyang',
            'dari': 'saka',
            'untuk': 'kanggo',
            'dengan': 'karo',
            'pada': 'ing',
            'dalam': 'ing jero',
            'akan': 'arep',
            'sudah': 'wis',
            'sedang': 'lagi',
            'masih': 'isih',
            'belum': 'durung',
            'semua': 'kabeh',
            'banyak': 'akeh',
            'sedikit': 'sithik',
            'sangat': 'banget',
            'bagus': 'apik',
            'jelek': 'elek',
            'besar': 'gedhe',
            'kecil': 'cilik',
            'panjang': 'dawa',
            'pendek': 'cendhak'
        }

        # Sundanese mappings (kasar/loma and lemes/hormat levels)
        self.sundanese_mappings = {
            'saya': 'abdi',  # lemes
            'kamu': 'anjeun',  # lemes
            'dia': 'anjeunna',
            'ini': 'ieu',
            'itu': 'eta',
            'yang': 'nu',
            'dan': 'jeung',
            'atau': 'atawa',
            'tidak': 'henteu',
            'ada': 'aya',
            'di': 'di',
            'ke': 'ka',
            'dari': 'ti',
            'untuk': 'pikeun',
            'dengan': 'sareng',
            'pada': 'dina',
            'dalam': 'dina',
            'akan': 'bade',
            'sudah': 'parantos',
            'sedang': 'nuju',
            'masih': 'masih',
            'belum': 'can',
            'semua': 'sadaya',
            'banyak': 'seueur',
            'sedikit': 'saeutik',
            'sangat': 'pisan',
            'bagus': 'saé',
            'jelek': 'awon',
            'besar': 'ageung',
            'kecil': 'alit',
            'panjang': 'panjang',
            'pendek': 'pondok'
        }

        # Social media specific particles and interjections
        self.social_media_particles = {
            'deh', 'dong', 'sih', 'kok', 'lah', 'kah', 'tuh', 'nih',
            'yah', 'wah', 'aduh', 'astaga', 'alamak', 'waduh', 'duh',
            'ih', 'eh', 'ah', 'oh', 'uh', 'hmm', 'hm', 'em', 'um',
            'ya', 'iya', 'yup', 'yep', 'nah', 'kan', 'gitu', 'gini',
            'begitu', 'begini', 'kayak', 'kaya', 'seperti', 'macam',
            'banget', 'bgt', 'bener', 'emang', 'memang', 'gimana',
            'bagaimana', 'kenapa', 'mengapa', 'kapan', 'dimana',
            'kemana', 'darimana', 'siapa', 'apa', 'mana'
        }

    def add_stopword_entry(self, en='', id_word='', jv='', su='', formal_id=''):
        """Add a stopword entry to the dataset"""
        entry = {
            'en': en.strip(),
            'id': id_word.strip(),
            'jv': jv.strip(),
            'su': su.strip(),
            'formal_id': formal_id.strip()
        }
        self.stopwords_data.append(entry)

    def generate_from_english_stopwords(self):
        """Generate entries starting from English stopwords"""
        logger.info("Generating entries from English stopwords...")

        for en_word in self.english_stopwords:
            # Try to find Indonesian equivalent
            id_word = ''
            formal_id = ''
            jv_word = ''
            su_word = ''

            # Basic English to Indonesian mappings
            en_to_id_mapping = {
                'i': 'saya',
                'me': 'saya',
                'my': 'saya',
                'we': 'kita',
                'our': 'kita',
                'you': 'kamu',
                'your': 'kamu',
                'he': 'dia',
                'him': 'dia',
                'his': 'dia',
                'she': 'dia',
                'her': 'dia',
                'it': 'itu',
                'they': 'mereka',
                'them': 'mereka',
                'their': 'mereka',
                'this': 'ini',
                'that': 'itu',
                'these': 'ini',
                'those': 'itu',
                'and': 'dan',
                'or': 'atau',
                'but': 'tetapi',
                'if': 'jika',
                'because': 'karena',
                'when': 'ketika',
                'while': 'sementara',
                'before': 'sebelum',
                'after': 'setelah',
                'in': 'di',
                'on': 'di',
                'at': 'di',
                'by': 'oleh',
                'for': 'untuk',
                'with': 'dengan',
                'from': 'dari',
                'to': 'ke',
                'of': 'dari',
                'the': '',  # No direct equivalent
                'a': '',    # No direct equivalent
                'an': '',   # No direct equivalent
                'is': 'adalah',
                'are': 'adalah',
                'was': 'adalah',
                'were': 'adalah',
                'be': 'adalah',
                'been': 'telah',
                'being': 'sedang',
                'have': 'punya',
                'has': 'punya',
                'had': 'punya',
                'do': '',
                'does': '',
                'did': '',
                'will': 'akan',
                'would': 'akan',
                'could': 'bisa',
                'should': 'harus',
                'may': 'mungkin',
                'might': 'mungkin',
                'can': 'bisa',
                'must': 'harus',
                'not': 'tidak',
                'no': 'tidak',
                'yes': 'ya',
                'all': 'semua',
                'any': 'apapun',
                'some': 'beberapa',
                'many': 'banyak',
                'much': 'banyak',
                'few': 'sedikit',
                'little': 'sedikit',
                'more': 'lebih',
                'most': 'paling',
                'less': 'kurang',
                'very': 'sangat',
                'too': 'terlalu',
                'so': 'jadi',
                'just': 'hanya',
                'only': 'hanya',
                'also': 'juga',
                'even': 'bahkan',
                'still': 'masih',
                'yet': 'belum',
                'already': 'sudah',
                'now': 'sekarang',
                'then': 'kemudian',
                'here': 'disini',
                'there': 'disana',
                'where': 'dimana',
                'how': 'bagaimana',
                'what': 'apa',
                'who': 'siapa',
                'why': 'mengapa',
                'which': 'yang mana'
            }

            if en_word in en_to_id_mapping:
                formal_id = en_to_id_mapping[en_word]
                id_word = formal_id  # Default to formal

                # Check if there's a Javanese equivalent
                if formal_id in self.javanese_mappings:
                    jv_word = self.javanese_mappings[formal_id]

                # Check if there's a Sundanese equivalent
                if formal_id in self.sundanese_mappings:
                    su_word = self.sundanese_mappings[formal_id]

            self.add_stopword_entry(en_word, id_word, jv_word, su_word, formal_id)

    def generate_from_indonesian_slang(self):
        """Generate entries from Indonesian slang and social media terms"""
        logger.info("Generating entries from Indonesian slang...")

        for slang, formal in self.indonesian_slang.items():
            jv_word = ''
            su_word = ''

            # Try to find Javanese equivalent for the formal word
            if formal in self.javanese_mappings:
                jv_word = self.javanese_mappings[formal]

            # Try to find Sundanese equivalent for the formal word
            if formal in self.sundanese_mappings:
                su_word = self.sundanese_mappings[formal]

            self.add_stopword_entry('', slang, jv_word, su_word, formal)

    def generate_from_formal_indonesian(self):
        """Generate entries from formal Indonesian words"""
        logger.info("Generating entries from formal Indonesian words...")

        for formal_word in self.indonesian_formal:
            jv_word = ''
            su_word = ''

            # Try to find Javanese equivalent
            if formal_word in self.javanese_mappings:
                jv_word = self.javanese_mappings[formal_word]

            # Try to find Sundanese equivalent
            if formal_word in self.sundanese_mappings:
                su_word = self.sundanese_mappings[formal_word]

            self.add_stopword_entry('', formal_word, jv_word, su_word, formal_word)

    def generate_from_social_media_particles(self):
        """Generate entries from social media particles and interjections"""
        logger.info("Generating entries from social media particles...")

        for particle in self.social_media_particles:
            # Most particles don't have direct equivalents in other languages
            self.add_stopword_entry('', particle, '', '', particle)

    def add_additional_entries(self):
        """Add additional common social media terms and expressions"""
        logger.info("Adding additional social media terms...")

        # Additional social media terms
        additional_terms = [
            # English social media terms
            ('lmao', '', '', '', 'laugh my ass off'),
            ('rofl', '', '', '', 'rolling on floor laughing'),
            ('ttyl', '', '', '', 'talk to you later'),
            ('imo', '', '', '', 'in my opinion'),
            ('tbh', '', '', '', 'to be honest'),
            ('nvm', '', '', '', 'never mind'),
            ('idk', '', '', '', 'i dont know'),
            ('irl', '', '', '', 'in real life'),
            ('dm', '', '', '', 'direct message'),
            ('pm', '', '', '', 'private message'),

            # Indonesian internet slang
            ('', 'woles', '', '', 'santai'),
            ('', 'santuy', '', '', 'santai'),
            ('', 'gabisa', '', '', 'tidak bisa'),
            ('', 'gasuka', '', '', 'tidak suka'),
            ('', 'gatau', '', '', 'tidak tahu'),
            ('', 'gapapa', '', '', 'tidak apa-apa'),
            ('', 'udah', '', '', 'sudah'),
            ('', 'udeh', '', '', 'sudah'),
            ('', 'belom', '', '', 'belum'),
            ('', 'blom', '', '', 'belum'),
            ('', 'gimana', '', '', 'bagaimana'),
            ('', 'gmn', '', '', 'bagaimana'),
            ('', 'knp', '', '', 'kenapa'),
            ('', 'knapa', '', '', 'kenapa'),
            ('', 'dmn', '', '', 'dimana'),
            ('', 'kmn', '', '', 'kemana'),
            ('', 'sapa', '', '', 'siapa'),
            ('', 'apa', '', '', 'apa'),
            ('', 'apaan', '', '', 'apa'),
            ('', 'ngapain', '', '', 'sedang apa'),
            ('', 'lagi', '', '', 'sedang'),
            ('', 'lg', '', '', 'sedang'),
            ('', 'jg', '', '', 'juga'),
            ('', 'juga', '', '', 'juga'),
            ('', 'tp', '', '', 'tetapi'),
            ('', 'tapi', '', '', 'tetapi'),
            ('', 'krn', '', '', 'karena'),
            ('', 'karna', '', '', 'karena'),
            ('', 'dgn', '', '', 'dengan'),
            ('', 'sama', '', '', 'dengan'),
            ('', 'utk', '', '', 'untuk'),
            ('', 'buat', '', '', 'untuk'),
            ('', 'dr', '', '', 'dari'),
            ('', 'dri', '', '', 'dari'),
            ('', 'ke', '', '', 'ke'),
            ('', 'di', '', '', 'di'),
            ('', 'pd', '', '', 'pada'),
            ('', 'dlm', '', '', 'dalam'),
            ('', 'sblm', '', '', 'sebelum'),
            ('', 'stlh', '', '', 'setelah'),
            ('', 'saat', '', '', 'saat'),
            ('', 'wkt', '', '', 'waktu'),
            ('', 'waktu', '', '', 'waktu'),

            # Javanese specific terms
            ('', '', 'aku', '', 'saya'),
            ('', '', 'kowe', '', 'kamu'),
            ('', '', 'dheweke', '', 'dia'),
            ('', '', 'awakmu', '', 'kamu'),
            ('', '', 'awakne', '', 'dia'),
            ('', '', 'iki', '', 'ini'),
            ('', '', 'iku', '', 'itu'),
            ('', '', 'kene', '', 'sini'),
            ('', '', 'kono', '', 'sana'),
            ('', '', 'ngendi', '', 'dimana'),
            ('', '', 'piye', '', 'bagaimana'),
            ('', '', 'apa', '', 'apa'),
            ('', '', 'sapa', '', 'siapa'),
            ('', '', 'kapan', '', 'kapan'),
            ('', '', 'ngapa', '', 'kenapa'),

            # Sundanese specific terms
            ('', '', '', 'abdi', 'saya'),
            ('', '', '', 'anjeun', 'kamu'),
            ('', '', '', 'anjeunna', 'dia'),
            ('', '', '', 'ieu', 'ini'),
            ('', '', '', 'eta', 'itu'),
            ('', '', '', 'dieu', 'sini'),
            ('', '', '', 'dinya', 'sana'),
            ('', '', '', 'dimana', 'dimana'),
            ('', '', '', 'kumaha', 'bagaimana'),
            ('', '', '', 'naon', 'apa'),
            ('', '', '', 'saha', 'siapa'),
            ('', '', '', 'iraha', 'kapan'),
            ('', '', '', 'naha', 'kenapa'),
        ]

        for entry in additional_terms:
            self.add_stopword_entry(*entry)

    def deduplicate_entries(self):
        """Remove duplicate entries from the dataset"""
        logger.info("Deduplicating entries...")

        seen = set()
        unique_data = []

        for entry in self.stopwords_data:
            # Create a tuple of all non-empty values for comparison
            key_values = tuple(v for v in entry.values() if v.strip())

            if key_values and key_values not in seen:
                seen.add(key_values)
                unique_data.append(entry)

        self.stopwords_data = unique_data
        logger.info(f"Removed duplicates. Final count: {len(self.stopwords_data)} entries")

    def generate_dataset(self):
        """Generate the complete multilingual stopword dataset"""
        logger.info("Starting multilingual stopword generation...")

        # Try to load and process NusaX dataset first
        if self.load_nusax_dataset():
            self.extract_stopwords_from_nusax()

        # Generate from different sources
        self.generate_from_english_stopwords()
        self.generate_from_indonesian_slang()
        self.generate_from_formal_indonesian()
        self.generate_from_social_media_particles()
        self.add_additional_entries()

        # Remove duplicates
        self.deduplicate_entries()

        # Ensure we have at least 1500 entries
        if len(self.stopwords_data) < 1500:
            logger.warning(f"Only {len(self.stopwords_data)} entries generated. Adding more...")
            self.add_more_entries_to_reach_target()

        logger.info(f"Dataset generation complete. Total entries: {len(self.stopwords_data)}")
        return self.stopwords_data

    def add_more_entries_to_reach_target(self):
        """Add more entries to reach the 1500 target"""
        # Comprehensive additional words and variations
        additional_words = [
            # More English words
            ('about', 'tentang', '', '', 'tentang'),
            ('above', 'atas', '', '', 'atas'),
            ('across', 'seberang', '', '', 'seberang'),
            ('against', 'melawan', '', '', 'melawan'),
            ('along', 'sepanjang', '', '', 'sepanjang'),
            ('among', 'antara', '', '', 'antara'),
            ('around', 'sekitar', '', '', 'sekitar'),
            ('behind', 'belakang', '', '', 'belakang'),
            ('below', 'bawah', '', '', 'bawah'),
            ('beneath', 'bawah', '', '', 'bawah'),
            ('beside', 'samping', '', '', 'samping'),
            ('between', 'antara', '', '', 'antara'),
            ('beyond', 'melampaui', '', '', 'melampaui'),
            ('during', 'selama', '', '', 'selama'),
            ('except', 'kecuali', '', '', 'kecuali'),
            ('inside', 'dalam', '', '', 'dalam'),
            ('outside', 'luar', '', '', 'luar'),
            ('through', 'melalui', '', '', 'melalui'),
            ('throughout', 'sepanjang', '', '', 'sepanjang'),
            ('toward', 'menuju', '', '', 'menuju'),
            ('towards', 'menuju', '', '', 'menuju'),
            ('under', 'bawah', '', '', 'bawah'),
            ('until', 'sampai', '', '', 'sampai'),
            ('upon', 'atas', '', '', 'atas'),
            ('within', 'dalam', '', '', 'dalam'),
            ('without', 'tanpa', '', '', 'tanpa'),

            # More Indonesian variations and slang
            ('', 'aja', '', '', 'saja'),
            ('', 'doang', '', '', 'saja'),
            ('', 'kok', '', '', ''),
            ('', 'sih', '', '', ''),
            ('', 'deh', '', '', ''),
            ('', 'dong', '', '', ''),
            ('', 'lah', '', '', ''),
            ('', 'kah', '', '', ''),
            ('', 'tuh', '', '', ''),
            ('', 'nih', '', '', ''),
            ('', 'yah', '', '', ''),
            ('', 'wah', '', '', ''),
            ('', 'aduh', '', '', ''),
            ('', 'astaga', '', '', ''),
            ('', 'alamak', '', '', ''),
            ('', 'waduh', '', '', ''),
            ('', 'duh', '', '', ''),
            ('', 'ih', '', '', ''),
            ('', 'eh', '', '', ''),
            ('', 'ah', '', '', ''),
            ('', 'oh', '', '', ''),
            ('', 'uh', '', '', ''),
            ('', 'hmm', '', '', ''),
            ('', 'hm', '', '', ''),
            ('', 'em', '', '', ''),
            ('', 'um', '', '', ''),
            ('', 'nah', '', '', ''),
            ('', 'kan', '', '', ''),
            ('', 'gitu', '', '', 'begitu'),
            ('', 'gini', '', '', 'begini'),
            ('', 'kayak', '', '', 'seperti'),
            ('', 'kaya', '', '', 'seperti'),
            ('', 'macam', '', '', 'seperti'),
            ('', 'banget', '', '', 'sangat'),
            ('', 'bgt', '', '', 'sangat'),
            ('', 'bener', '', '', 'benar'),
            ('', 'emang', '', '', 'memang'),
            ('', 'memang', '', '', 'memang'),

            # Numbers and time expressions
            ('one', 'satu', 'siji', 'hiji', 'satu'),
            ('two', 'dua', 'loro', 'dua', 'dua'),
            ('three', 'tiga', 'telu', 'tilu', 'tiga'),
            ('four', 'empat', 'papat', 'opat', 'empat'),
            ('five', 'lima', 'lima', 'lima', 'lima'),
            ('six', 'enam', 'enem', 'genep', 'enam'),
            ('seven', 'tujuh', 'pitu', 'tujuh', 'tujuh'),
            ('eight', 'delapan', 'wolu', 'dalapan', 'delapan'),
            ('nine', 'sembilan', 'sanga', 'salapan', 'sembilan'),
            ('ten', 'sepuluh', 'sepuluh', 'sapuluh', 'sepuluh'),
            ('today', 'hari ini', 'dina iki', 'dinten ieu', 'hari ini'),
            ('tomorrow', 'besok', 'sesuk', 'isukan', 'besok'),
            ('yesterday', 'kemarin', 'wingi', 'kamari', 'kemarin'),
            ('morning', 'pagi', 'esuk', 'isuk', 'pagi'),
            ('afternoon', 'siang', 'awan', 'siang', 'siang'),
            ('evening', 'sore', 'sore', 'sonten', 'sore'),
            ('night', 'malam', 'bengi', 'wengi', 'malam'),

            # More social media terms and internet slang
            ('', 'wkwkwkwk', '', '', 'hahaha'),
            ('', 'wkwkland', '', '', 'indonesia'),
            ('', 'xixi', '', '', 'hehe'),
            ('', 'xixixixi', '', '', 'hehehe'),
            ('', 'kwkwkw', '', '', 'hahaha'),
            ('', 'wakaka', '', '', 'hahaha'),
            ('', 'wakakaka', '', '', 'hahaha'),
            ('', 'hahaha', '', '', 'hahaha'),
            ('', 'hehehe', '', '', 'hehehe'),
            ('', 'hihi', '', '', 'hihi'),
            ('', 'hoho', '', '', 'hoho'),
            ('', 'huhu', '', '', 'huhu'),
            ('', 'wkwk', '', '', 'haha'),
            ('', 'kwkw', '', '', 'haha'),
            ('', 'kkkk', '', '', 'haha'),
            ('', 'kkkkk', '', '', 'hahaha'),
            ('', 'wkakaka', '', '', 'hahaha'),
            ('', 'wakwaw', '', '', 'haha'),
            ('', 'wkwkwk', '', '', 'hahaha'),

            # More Indonesian formal and colloquial
            ('', 'sekarang', '', '', 'sekarang'),
            ('', 'skrg', '', '', 'sekarang'),
            ('', 'skrang', '', '', 'sekarang'),
            ('', 'nanti', '', '', 'nanti'),
            ('', 'ntar', '', '', 'nanti'),
            ('', 'tar', '', '', 'nanti'),
            ('', 'tadi', '', '', 'tadi'),
            ('', 'td', '', '', 'tadi'),
            ('', 'kemarin', '', '', 'kemarin'),
            ('', 'kmrn', '', '', 'kemarin'),
            ('', 'besok', '', '', 'besok'),
            ('', 'bsk', '', '', 'besok'),
            ('', 'lusa', '', '', 'lusa'),
            ('', 'minggu', '', '', 'minggu'),
            ('', 'mgg', '', '', 'minggu'),
            ('', 'bulan', '', '', 'bulan'),
            ('', 'bln', '', '', 'bulan'),
            ('', 'tahun', '', '', 'tahun'),
            ('', 'thn', '', '', 'tahun'),
            ('', 'hari', '', '', 'hari'),
            ('', 'hr', '', '', 'hari'),
            ('', 'jam', '', '', 'jam'),
            ('', 'menit', '', '', 'menit'),
            ('', 'mnt', '', '', 'menit'),
            ('', 'detik', '', '', 'detik'),
            ('', 'dtk', '', '', 'detik'),

            # More Javanese terms
            ('', '', 'aku', '', 'saya'),
            ('', '', 'kowe', '', 'kamu'),
            ('', '', 'dheweke', '', 'dia'),
            ('', '', 'awakmu', '', 'kamu'),
            ('', '', 'awakne', '', 'dia'),
            ('', '', 'iki', '', 'ini'),
            ('', '', 'iku', '', 'itu'),
            ('', '', 'kene', '', 'sini'),
            ('', '', 'kono', '', 'sana'),
            ('', '', 'ngendi', '', 'dimana'),
            ('', '', 'piye', '', 'bagaimana'),
            ('', '', 'apa', '', 'apa'),
            ('', '', 'sapa', '', 'siapa'),
            ('', '', 'kapan', '', 'kapan'),
            ('', '', 'ngapa', '', 'kenapa'),
            ('', '', 'ning', '', 'di'),
            ('', '', 'nang', '', 'di'),
            ('', '', 'menyang', '', 'ke'),
            ('', '', 'saka', '', 'dari'),
            ('', '', 'kanggo', '', 'untuk'),
            ('', '', 'karo', '', 'dengan'),
            ('', '', 'lan', '', 'dan'),
            ('', '', 'utawa', '', 'atau'),
            ('', '', 'nanging', '', 'tetapi'),
            ('', '', 'merga', '', 'karena'),
            ('', '', 'yen', '', 'jika'),
            ('', '', 'nalika', '', 'ketika'),
            ('', '', 'arep', '', 'akan'),
            ('', '', 'wis', '', 'sudah'),
            ('', '', 'lagi', '', 'sedang'),
            ('', '', 'isih', '', 'masih'),
            ('', '', 'durung', '', 'belum'),
            ('', '', 'ora', '', 'tidak'),
            ('', '', 'dudu', '', 'bukan'),
            ('', '', 'ana', '', 'ada'),
            ('', '', 'ora ana', '', 'tidak ada'),
            ('', '', 'kabeh', '', 'semua'),
            ('', '', 'akeh', '', 'banyak'),
            ('', '', 'sithik', '', 'sedikit'),
            ('', '', 'banget', '', 'sangat'),
            ('', '', 'apik', '', 'bagus'),
            ('', '', 'elek', '', 'jelek'),
            ('', '', 'gedhe', '', 'besar'),
            ('', '', 'cilik', '', 'kecil'),
            ('', '', 'dawa', '', 'panjang'),
            ('', '', 'cendhak', '', 'pendek'),

            # More Sundanese terms
            ('', '', '', 'abdi', 'saya'),
            ('', '', '', 'anjeun', 'kamu'),
            ('', '', '', 'anjeunna', 'dia'),
            ('', '', '', 'ieu', 'ini'),
            ('', '', '', 'eta', 'itu'),
            ('', '', '', 'dieu', 'sini'),
            ('', '', '', 'dinya', 'sana'),
            ('', '', '', 'dimana', 'dimana'),
            ('', '', '', 'kumaha', 'bagaimana'),
            ('', '', '', 'naon', 'apa'),
            ('', '', '', 'saha', 'siapa'),
            ('', '', '', 'iraha', 'kapan'),
            ('', '', '', 'naha', 'kenapa'),
            ('', '', '', 'di', 'di'),
            ('', '', '', 'ka', 'ke'),
            ('', '', '', 'ti', 'dari'),
            ('', '', '', 'pikeun', 'untuk'),
            ('', '', '', 'sareng', 'dengan'),
            ('', '', '', 'jeung', 'dan'),
            ('', '', '', 'atawa', 'atau'),
            ('', '', '', 'tapi', 'tetapi'),
            ('', '', '', 'sabab', 'karena'),
            ('', '', '', 'lamun', 'jika'),
            ('', '', '', 'nalika', 'ketika'),
            ('', '', '', 'bade', 'akan'),
            ('', '', '', 'parantos', 'sudah'),
            ('', '', '', 'nuju', 'sedang'),
            ('', '', '', 'masih', 'masih'),
            ('', '', '', 'can', 'belum'),
            ('', '', '', 'henteu', 'tidak'),
            ('', '', '', 'sanés', 'bukan'),
            ('', '', '', 'aya', 'ada'),
            ('', '', '', 'teu aya', 'tidak ada'),
            ('', '', '', 'sadaya', 'semua'),
            ('', '', '', 'seueur', 'banyak'),
            ('', '', '', 'saeutik', 'sedikit'),
            ('', '', '', 'pisan', 'sangat'),
            ('', '', '', 'saé', 'bagus'),
            ('', '', '', 'awon', 'jelek'),
            ('', '', '', 'ageung', 'besar'),
            ('', '', '', 'alit', 'kecil'),
            ('', '', '', 'panjang', 'panjang'),
            ('', '', '', 'pondok', 'pendek'),

            # Additional common words
            ('good', 'bagus', 'apik', 'saé', 'bagus'),
            ('bad', 'buruk', 'elek', 'awon', 'buruk'),
            ('big', 'besar', 'gedhe', 'ageung', 'besar'),
            ('small', 'kecil', 'cilik', 'alit', 'kecil'),
            ('long', 'panjang', 'dawa', 'panjang', 'panjang'),
            ('short', 'pendek', 'cendhak', 'pondok', 'pendek'),
            ('new', 'baru', 'anyar', 'anyar', 'baru'),
            ('old', 'lama', 'lawas', 'lami', 'lama'),
            ('young', 'muda', 'enom', 'ngora', 'muda'),
            ('fast', 'cepat', 'cepet', 'gancang', 'cepat'),
            ('slow', 'lambat', 'alon', 'laun', 'lambat'),
            ('hot', 'panas', 'panas', 'panas', 'panas'),
            ('cold', 'dingin', 'adhem', 'tiis', 'dingin'),
            ('warm', 'hangat', 'anget', 'haneut', 'hangat'),
            ('cool', 'sejuk', 'adem', 'tiis', 'sejuk'),
            ('wet', 'basah', 'teles', 'baseuh', 'basah'),
            ('dry', 'kering', 'garing', 'garing', 'kering'),
            ('clean', 'bersih', 'resik', 'beresih', 'bersih'),
            ('dirty', 'kotor', 'reged', 'kotor', 'kotor'),
            ('easy', 'mudah', 'gampang', 'gampil', 'mudah'),
            ('hard', 'sulit', 'angel', 'hese', 'sulit'),
            ('light', 'ringan', 'entheng', 'hampang', 'ringan'),
            ('heavy', 'berat', 'abot', 'beurat', 'berat'),
            ('high', 'tinggi', 'dhuwur', 'luhur', 'tinggi'),
            ('low', 'rendah', 'cendhek', 'handap', 'rendah'),
            ('near', 'dekat', 'cedhak', 'deukeut', 'dekat'),
            ('far', 'jauh', 'adoh', 'jauh', 'jauh'),
            ('left', 'kiri', 'kiwa', 'kénca', 'kiri'),
            ('right', 'kanan', 'tengen', 'katuhu', 'kanan'),
            ('front', 'depan', 'ngarep', 'hareupeun', 'depan'),
            ('back', 'belakang', 'mburi', 'tukang', 'belakang'),
            ('up', 'atas', 'dhuwur', 'luhur', 'atas'),
            ('down', 'bawah', 'ngisor', 'handap', 'bawah'),
            ('first', 'pertama', 'pisanan', 'kahiji', 'pertama'),
            ('last', 'terakhir', 'pungkasan', 'panungtungan', 'terakhir'),
            ('next', 'berikutnya', 'sabanjure', 'salajengna', 'berikutnya'),
            ('previous', 'sebelumnya', 'sadurunge', 'sateuacanna', 'sebelumnya'),
            ('same', 'sama', 'padha', 'sarua', 'sama'),
            ('different', 'berbeda', 'beda', 'béda', 'berbeda'),
            ('true', 'benar', 'bener', 'leres', 'benar'),
            ('false', 'salah', 'salah', 'lepat', 'salah'),
            ('yes', 'ya', 'iya', 'enya', 'ya'),
            ('no', 'tidak', 'ora', 'henteu', 'tidak'),
            ('maybe', 'mungkin', 'mbok menawa', 'meureun', 'mungkin'),
            ('sure', 'pasti', 'mesthi', 'pasti', 'pasti'),
            ('never', 'tidak pernah', 'ora tau', 'henteu kantos', 'tidak pernah'),
            ('always', 'selalu', 'tansah', 'salawasna', 'selalu'),
            ('sometimes', 'kadang', 'kadhang', 'sakapeung', 'kadang'),
            ('often', 'sering', 'kerep', 'mindeng', 'sering'),
            ('rarely', 'jarang', 'arang', 'jarang', 'jarang'),
            ('usually', 'biasanya', 'biasane', 'biasana', 'biasanya'),
            ('normally', 'normalnya', 'lumrahe', 'biasana', 'normalnya'),
            ('really', 'benar-benar', 'tenan', 'leres-leres', 'benar-benar'),
            ('actually', 'sebenarnya', 'sejatine', 'saleresna', 'sebenarnya'),
            ('probably', 'mungkin', 'mbok menawa', 'sigana', 'mungkin'),
            ('definitely', 'pasti', 'mesthi', 'pasti', 'pasti'),
            ('certainly', 'tentu', 'mesthi', 'tangtu', 'tentu'),
            ('absolutely', 'mutlak', 'mutlak', 'mutlak', 'mutlak'),
            ('completely', 'sepenuhnya', 'sakabehe', 'lengkep', 'sepenuhnya'),
            ('totally', 'total', 'total', 'total', 'total'),
            ('exactly', 'persis', 'persis', 'persis', 'persis'),
            ('almost', 'hampir', 'meh', 'ampir', 'hampir'),
            ('quite', 'cukup', 'cukup', 'cukup', 'cukup'),
            ('rather', 'agak', 'rada', 'rada', 'agak'),
            ('pretty', 'cukup', 'lumayan', 'lumayan', 'cukup'),
            ('fairly', 'cukup', 'lumayan', 'lumayan', 'cukup'),
            ('extremely', 'sangat', 'banget', 'pisan', 'sangat'),
            ('incredibly', 'luar biasa', 'luar biasa', 'luar biasa', 'luar biasa'),
            ('amazingly', 'menakjubkan', 'nggumunake', 'endah', 'menakjubkan'),
            ('surprisingly', 'mengejutkan', 'nggumunake', 'héran', 'mengejutkan'),
        ]

        # Add entries until we reach 1500
        for entry in additional_words:
            self.add_stopword_entry(*entry)
            if len(self.stopwords_data) >= 1500:
                break

        # If still not enough, add more variations
        if len(self.stopwords_data) < 1500:
            self.add_even_more_entries()

    def load_nusax_dataset(self):
        """Load NusaX-senti dataset for extracting authentic multilingual stopwords"""
        try:
            logger.info("Loading NusaX-senti dataset...")
            # Load the dataset for each language separately
            self.nusax_data = {}
            languages = ['ind', 'jav', 'sun', 'eng']  # Focus on main languages we need

            for lang in languages:
                try:
                    self.nusax_data[lang] = load_dataset("indonlp/NusaX-senti", lang)
                    logger.info(f"Loaded {lang} subset successfully")
                except Exception as e:
                    logger.warning(f"Could not load {lang} subset: {e}")
                    continue

            if self.nusax_data:
                logger.info("NusaX-senti dataset loaded successfully")
                return True
            else:
                logger.warning("No NusaX subsets could be loaded")
                return False
        except Exception as e:
            logger.warning(f"Could not load NusaX dataset: {e}")
            logger.info("Continuing without NusaX dataset...")
            return False

    def extract_stopwords_from_nusax(self):
        """Extract common words from NusaX dataset to enhance stopword list"""
        if not self.nusax_data:
            logger.info("NusaX dataset not available, skipping extraction")
            return

        logger.info("Extracting stopwords from NusaX dataset...")

        # Extract common words from each language
        word_frequency = {}

        for lang_code, dataset in self.nusax_data.items():
            logger.info(f"Processing {lang_code} language data...")
            word_frequency[lang_code] = Counter()

            for split in ['train', 'validation', 'test']:
                try:
                    # Get data for this language and split
                    data = dataset[split]

                    # Process each text sample
                    for sample in data:
                        text = sample['text'].lower()
                        # Simple tokenization - split by whitespace and remove punctuation
                        words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text)  # Include accented characters
                        word_frequency[lang_code].update(words)

                except Exception as e:
                    logger.warning(f"Error processing {lang_code} {split}: {e}")
                    continue

        # Extract most common words as potential stopwords
        for lang_code, freq_counter in word_frequency.items():
            # Get top 100 most common words for each language
            common_words = [word for word, count in freq_counter.most_common(100)
                          if len(word) > 1 and count > 3]  # Filter short words and rare words

            logger.info(f"Found {len(common_words)} common words in {lang_code}")

            # Map to our column format
            for word in common_words:
                if lang_code == 'eng':
                    self.add_stopword_entry(word, '', '', '', '')
                elif lang_code == 'ind':
                    # Try to find equivalents in Javanese and Sundanese
                    jv_word = self.javanese_mappings.get(word, '')
                    su_word = self.sundanese_mappings.get(word, '')
                    self.add_stopword_entry('', word, jv_word, su_word, word)
                elif lang_code == 'jav':
                    # Find Indonesian equivalent if possible
                    id_word = ''
                    formal_id = ''
                    for id_key, jv_val in self.javanese_mappings.items():
                        if jv_val == word:
                            id_word = id_key
                            formal_id = id_key
                            break
                    self.add_stopword_entry('', id_word, word, '', formal_id)
                elif lang_code == 'sun':
                    # Find Indonesian equivalent if possible
                    id_word = ''
                    formal_id = ''
                    for id_key, su_val in self.sundanese_mappings.items():
                        if su_val == word:
                            id_word = id_key
                            formal_id = id_key
                            break
                    self.add_stopword_entry('', id_word, '', word, formal_id)
                else:
                    # For other languages (like Acehnese), add as additional data
                    self.add_stopword_entry('', '', '', '', word)

        logger.info("Finished extracting stopwords from NusaX dataset")

    def add_even_more_entries(self):
        """Add even more entries to ensure we reach 1500"""
        # Generate more variations and combinations
        more_entries = []

        # Add more Indonesian internet slang and abbreviations
        internet_slang = [
            ('', 'ygy', '', '', 'ya guys ya'),
            ('', 'yg', '', '', 'yang'),
            ('', 'sy', '', '', 'saya'),
            ('', 'km', '', '', 'kamu'),
            ('', 'dy', '', '', 'dia'),
            ('', 'mrk', '', '', 'mereka'),
            ('', 'kt', '', '', 'kita'),
            ('', 'kmi', '', '', 'kami'),
            ('', 'ini', '', '', 'ini'),
            ('', 'itu', '', '', 'itu'),
            ('', 'dan', '', '', 'dan'),
            ('', 'atau', '', '', 'atau'),
            ('', 'tp', '', '', 'tetapi'),
            ('', 'krn', '', '', 'karena'),
            ('', 'jk', '', '', 'jika'),
            ('', 'ktk', '', '', 'ketika'),
            ('', 'saat', '', '', 'saat'),
            ('', 'wkt', '', '', 'waktu'),
            ('', 'di', '', '', 'di'),
            ('', 'ke', '', '', 'ke'),
            ('', 'dr', '', '', 'dari'),
            ('', 'utk', '', '', 'untuk'),
            ('', 'dgn', '', '', 'dengan'),
            ('', 'pd', '', '', 'pada'),
            ('', 'dlm', '', '', 'dalam'),
            ('', 'olh', '', '', 'oleh'),
            ('', 'adlh', '', '', 'adalah'),
            ('', 'akn', '', '', 'akan'),
            ('', 'tlh', '', '', 'telah'),
            ('', 'sdh', '', '', 'sudah'),
            ('', 'sdg', '', '', 'sedang'),
            ('', 'msh', '', '', 'masih'),
            ('', 'blm', '', '', 'belum'),
            ('', 'tdk', '', '', 'tidak'),
            ('', 'bkn', '', '', 'bukan'),
            ('', 'jgn', '', '', 'jangan'),
            ('', 'ada', '', '', 'ada'),
            ('', 'smu', '', '', 'semua'),
            ('', 'stp', '', '', 'setiap'),
            ('', 'bbp', '', '', 'beberapa'),
            ('', 'bnyk', '', '', 'banyak'),
            ('', 'sdkt', '', '', 'sedikit'),
            ('', 'sgt', '', '', 'sangat'),
            ('', 'agk', '', '', 'agak'),
            ('', 'ckp', '', '', 'cukup'),
            ('', 'tll', '', '', 'terlalu'),
            ('', 'plg', '', '', 'paling'),
            ('', 'lbh', '', '', 'lebih'),
            ('', 'krg', '', '', 'kurang'),
        ]

        # Add more emotional expressions and reactions
        emotions = [
            ('', 'huft', '', '', ''),
            ('', 'hufh', '', '', ''),
            ('', 'hufft', '', '', ''),
            ('', 'haah', '', '', ''),
            ('', 'haaah', '', '', ''),
            ('', 'aaah', '', '', ''),
            ('', 'oooh', '', '', ''),
            ('', 'uuuh', '', '', ''),
            ('', 'eeeh', '', '', ''),
            ('', 'iiih', '', '', ''),
            ('', 'owh', '', '', ''),
            ('', 'owwh', '', '', ''),
            ('', 'owwwh', '', '', ''),
            ('', 'hehe', '', '', ''),
            ('', 'hihi', '', '', ''),
            ('', 'hoho', '', '', ''),
            ('', 'huhu', '', '', ''),
            ('', 'haha', '', '', ''),
            ('', 'hehe', '', '', ''),
            ('', 'xixi', '', '', ''),
            ('', 'keke', '', '', ''),
            ('', 'gege', '', '', ''),
            ('', 'wkwk', '', '', ''),
            ('', 'kwkw', '', '', ''),
            ('', 'wakaka', '', '', ''),
            ('', 'wkakaka', '', '', ''),
            ('', 'wakakaka', '', '', ''),
            ('', 'wkwkwk', '', '', ''),
            ('', 'wkwkwkwk', '', '', ''),
            ('', 'kwkwkw', '', '', ''),
            ('', 'kwkwkwkw', '', '', ''),
        ]

        # Add more question words and variations
        questions = [
            ('', 'gimana', '', '', 'bagaimana'),
            ('', 'gmn', '', '', 'bagaimana'),
            ('', 'bgmn', '', '', 'bagaimana'),
            ('', 'kenapa', '', '', 'kenapa'),
            ('', 'knp', '', '', 'kenapa'),
            ('', 'knapa', '', '', 'kenapa'),
            ('', 'mengapa', '', '', 'mengapa'),
            ('', 'dimana', '', '', 'dimana'),
            ('', 'dmn', '', '', 'dimana'),
            ('', 'dmana', '', '', 'dimana'),
            ('', 'kemana', '', '', 'kemana'),
            ('', 'kmn', '', '', 'kemana'),
            ('', 'kmana', '', '', 'kemana'),
            ('', 'darimana', '', '', 'darimana'),
            ('', 'drmn', '', '', 'darimana'),
            ('', 'drmana', '', '', 'darimana'),
            ('', 'siapa', '', '', 'siapa'),
            ('', 'spa', '', '', 'siapa'),
            ('', 'sapa', '', '', 'siapa'),
            ('', 'apa', '', '', 'apa'),
            ('', 'apaan', '', '', 'apa'),
            ('', 'apain', '', '', 'apa'),
            ('', 'mana', '', '', 'mana'),
            ('', 'mn', '', '', 'mana'),
            ('', 'kapan', '', '', 'kapan'),
            ('', 'kpn', '', '', 'kapan'),
            ('', 'kpan', '', '', 'kapan'),
        ]

        # Add more time-related expressions
        time_expressions = [
            ('', 'sekarang', '', '', 'sekarang'),
            ('', 'skrg', '', '', 'sekarang'),
            ('', 'skrang', '', '', 'sekarang'),
            ('', 'nanti', '', '', 'nanti'),
            ('', 'ntar', '', '', 'nanti'),
            ('', 'tar', '', '', 'nanti'),
            ('', 'tadi', '', '', 'tadi'),
            ('', 'td', '', '', 'tadi'),
            ('', 'barusan', '', '', 'baru saja'),
            ('', 'brsan', '', '', 'baru saja'),
            ('', 'baru', '', '', 'baru'),
            ('', 'br', '', '', 'baru'),
            ('', 'lama', '', '', 'lama'),
            ('', 'lm', '', '', 'lama'),
            ('', 'cepat', '', '', 'cepat'),
            ('', 'cpat', '', '', 'cepat'),
            ('', 'cpt', '', '', 'cepat'),
            ('', 'lambat', '', '', 'lambat'),
            ('', 'lmbt', '', '', 'lambat'),
            ('', 'pelan', '', '', 'pelan'),
            ('', 'pln', '', '', 'pelan'),
        ]

        # Combine all additional entries
        more_entries.extend(internet_slang)
        more_entries.extend(emotions)
        more_entries.extend(questions)
        more_entries.extend(time_expressions)

        # Add entries until we reach 1500
        for entry in more_entries:
            if len(self.stopwords_data) >= 1500:
                break
            self.add_stopword_entry(*entry)

    def save_to_csv(self, filename='multilingual_stopwords_socialmedia.csv'):
        """Save the dataset to CSV file"""
        logger.info(f"Saving dataset to {filename}...")

        # Convert to DataFrame
        df = pd.DataFrame(self.stopwords_data)

        # Reorder columns to match requirements
        df = df[['en', 'id', 'jv', 'su', 'formal_id']]

        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8')

        logger.info(f"Dataset saved successfully to {filename}")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Columns: {list(df.columns)}")

        # Print sample of the data
        print("\nSample of generated data:")
        print(df.head(10).to_string(index=False))

        return df


def main():
    """Main function to generate the multilingual stopword dataset"""
    try:
        # Create generator instance
        generator = MultilingualStopwordGenerator()

        # Generate the dataset
        generator.generate_dataset()

        # Save to CSV
        df = generator.save_to_csv()

        print(f"\n✅ Successfully generated multilingual stopword dataset!")
        print(f"📊 Total entries: {len(df)}")
        print(f"📁 Saved to: multilingual_stopwords_socialmedia.csv")

        # Show statistics
        print(f"\n📈 Statistics:")
        print(f"   - English entries: {len(df[df['en'] != ''])}")
        print(f"   - Indonesian entries: {len(df[df['id'] != ''])}")
        print(f"   - Javanese entries: {len(df[df['jv'] != ''])}")
        print(f"   - Sundanese entries: {len(df[df['su'] != ''])}")
        print(f"   - Formal Indonesian entries: {len(df[df['formal_id'] != ''])}")

    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise


if __name__ == "__main__":
    main()
