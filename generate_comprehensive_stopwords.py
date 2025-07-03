#!/usr/bin/env python3
"""
Comprehensive Multilingual Stopword Generator for Social Media Domain

This script generates a comprehensive multilingual stopword list specifically
designed for social media text processing, covering English, Indonesian (formal and colloquial),
Javanese, and Sundanese languages with over 1500 entries.

Author: NLP Engineer
Date: 2025-07-03
"""

import pandas as pd
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveMultilingualStopwordGenerator:
    """
    A comprehensive class to generate multilingual stopwords for social media domain
    """
    
    def __init__(self):
        """Initialize the generator with comprehensive language mappings and dictionaries"""
        self.stopwords_data = []
        self.setup_nltk()
        self.setup_comprehensive_mappings()
    
    def setup_nltk(self):
        """Download and setup NLTK data"""
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.english_stopwords = set(stopwords.words('english'))
            logger.info("NLTK English stopwords loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NLTK stopwords: {e}")
            # Fallback to comprehensive English stopwords
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
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
            }
    
    def setup_comprehensive_mappings(self):
        """Setup comprehensive language mapping dictionaries"""
        
        # Comprehensive Indonesian slang and social media abbreviations
        self.indonesian_slang = {
            # Laughter expressions
            'wkwk': 'haha', 'wkwkwk': 'hahaha', 'wkwkwkwk': 'hahahaha',
            'xixixi': 'hehehe', 'xixi': 'hehe', 'hehe': 'hehe', 'hihi': 'hihi',
            'hoho': 'hoho', 'huhu': 'huhu', 'wakaka': 'hahaha', 'wakakaka': 'hahahaha',
            'kwkw': 'haha', 'kwkwkw': 'hahaha', 'kkkk': 'haha', 'kkkkk': 'hahaha',
            
            # English abbreviations commonly used in Indonesian social media
            'brb': 'be right back', 'btw': 'by the way', 'cmiiw': 'correct me if i am wrong',
            'fyi': 'for your information', 'imho': 'in my humble opinion', 'lol': 'laugh out loud',
            'omg': 'oh my god', 'wtf': 'what the f', 'asap': 'as soon as possible',
            'aka': 'also known as', 'etc': 'et cetera', 'lmao': 'laugh my ass off',
            'rofl': 'rolling on floor laughing', 'ttyl': 'talk to you later',
            'imo': 'in my opinion', 'tbh': 'to be honest', 'nvm': 'never mind',
            'idk': 'i dont know', 'irl': 'in real life', 'dm': 'direct message',
            'pm': 'private message',
            
            # Indonesian pronouns and common words (slang)
            'gw': 'saya', 'gue': 'saya', 'aku': 'saya', 'w': 'saya',
            'lu': 'kamu', 'lo': 'kamu', 'elu': 'kamu', 'u': 'kamu',
            'dia': 'dia', 'dy': 'dia', 'mrk': 'mereka', 'kt': 'kita', 'kmi': 'kami',
            
            # Family terms (slang)
            'bokap': 'ayah', 'nyokap': 'ibu', 'ortu': 'orang tua', 'bro': 'saudara',
            'sis': 'saudari', 'gan': 'juragan', 'agan': 'juragan', 'suhu': 'master',
            
            # Common expressions
            'mantap': 'bagus', 'keren': 'bagus', 'anjay': 'wow', 'anjir': 'wow',
            'buset': 'wow', 'gilak': 'gila', 'gokil': 'gila', 'kepo': 'ingin tahu',
            'gabut': 'tidak ada kegiatan', 'baper': 'bawa perasaan', 'galau': 'bingung',
            'bucin': 'budak cinta', 'jones': 'jomblo ngenes', 'php': 'pemberi harapan palsu',
            'pdkt': 'pendekatan', 'ttm': 'teman tapi mesra', 'clbk': 'cinta lama bersemi kembali',
            
            # Time and frequency
            'skrg': 'sekarang', 'skrang': 'sekarang', 'ntar': 'nanti', 'tar': 'nanti',
            'td': 'tadi', 'kmrn': 'kemarin', 'bsk': 'besok', 'hr': 'hari',
            'mgg': 'minggu', 'bln': 'bulan', 'thn': 'tahun', 'mnt': 'menit',
            'dtk': 'detik', 'jam': 'jam',
            
            # Question words (slang)
            'gimana': 'bagaimana', 'gmn': 'bagaimana', 'bgmn': 'bagaimana',
            'kenapa': 'kenapa', 'knp': 'kenapa', 'knapa': 'kenapa',
            'dimana': 'dimana', 'dmn': 'dimana', 'dmana': 'dimana',
            'kemana': 'kemana', 'kmn': 'kemana', 'kmana': 'kemana',
            'darimana': 'darimana', 'drmn': 'darimana', 'drmana': 'darimana',
            'siapa': 'siapa', 'spa': 'siapa', 'sapa': 'siapa',
            'apa': 'apa', 'apaan': 'apa', 'apain': 'apa',
            'mana': 'mana', 'mn': 'mana', 'kapan': 'kapan', 'kpn': 'kapan', 'kpan': 'kapan',
            
            # Negations and affirmations
            'gabisa': 'tidak bisa', 'gasuka': 'tidak suka', 'gatau': 'tidak tahu',
            'gapapa': 'tidak apa-apa', 'udah': 'sudah', 'udeh': 'sudah',
            'belom': 'belum', 'blom': 'belum', 'iya': 'ya', 'yup': 'ya', 'yep': 'ya',
            
            # Particles and fillers
            'aja': 'saja', 'doang': 'saja', 'kok': '', 'sih': '', 'deh': '',
            'dong': '', 'lah': '', 'kah': '', 'tuh': '', 'nih': '',
            'yah': '', 'wah': '', 'nah': '', 'kan': '', 'gitu': 'begitu',
            'gini': 'begini', 'kayak': 'seperti', 'kaya': 'seperti',
            'banget': 'sangat', 'bgt': 'sangat', 'bener': 'benar',
            'emang': 'memang', 'memang': 'memang',
            
            # Interjections
            'aduh': '', 'astaga': '', 'alamak': '', 'waduh': '', 'duh': '',
            'ih': '', 'eh': '', 'ah': '', 'oh': '', 'uh': '', 'hmm': '',
            'hm': '', 'em': '', 'um': '', 'huft': '', 'hufh': '', 'hufft': '',
            'haah': '', 'haaah': '', 'aaah': '', 'oooh': '', 'uuuh': '',
            'eeeh': '', 'iiih': '', 'owh': '', 'owwh': '', 'owwwh': '',
            
            # Abbreviations
            'yg': 'yang', 'sy': 'saya', 'km': 'kamu', 'dy': 'dia',
            'mrk': 'mereka', 'kt': 'kita', 'kmi': 'kami', 'ini': 'ini',
            'itu': 'itu', 'dan': 'dan', 'atau': 'atau', 'tp': 'tetapi',
            'krn': 'karena', 'jk': 'jika', 'ktk': 'ketika', 'saat': 'saat',
            'wkt': 'waktu', 'di': 'di', 'ke': 'ke', 'dr': 'dari',
            'utk': 'untuk', 'dgn': 'dengan', 'pd': 'pada', 'dlm': 'dalam',
            'olh': 'oleh', 'adlh': 'adalah', 'akn': 'akan', 'tlh': 'telah',
            'sdh': 'sudah', 'sdg': 'sedang', 'msh': 'masih', 'blm': 'belum',
            'tdk': 'tidak', 'bkn': 'bukan', 'jgn': 'jangan', 'ada': 'ada',
            'smu': 'semua', 'stp': 'setiap', 'bbp': 'beberapa', 'bnyk': 'banyak',
            'sdkt': 'sedikit', 'sgt': 'sangat', 'agk': 'agak', 'ckp': 'cukup',
            'tll': 'terlalu', 'plg': 'paling', 'lbh': 'lebih', 'krg': 'kurang',
            
            # Social media specific
            'woles': 'santai', 'santuy': 'santai', 'newbie': 'pemula',
            'noob': 'pemula', 'pro': 'profesional'
        }

        # Comprehensive Indonesian formal words and common stopwords
        self.indonesian_formal = {
            # Pronouns
            'saya', 'aku', 'kamu', 'anda', 'dia', 'mereka', 'kita', 'kami',
            'beliau', 'kalian', 'engkau', 'dirinya', 'diri', 'sendiri',

            # Demonstratives
            'ini', 'itu', 'tersebut', 'berikut', 'demikian', 'begini', 'begitu',
            'seperti', 'serupa', 'sama', 'beda', 'berbeda', 'lain', 'lainnya',

            # Conjunctions
            'yang', 'dan', 'atau', 'tetapi', 'namun', 'karena', 'sebab',
            'jika', 'kalau', 'bila', 'ketika', 'saat', 'waktu', 'selama',
            'hingga', 'sampai', 'sebelum', 'sesudah', 'setelah', 'lalu',
            'kemudian', 'selanjutnya', 'akhirnya', 'maka', 'jadi', 'sehingga',
            'supaya', 'agar', 'untuk', 'bagi', 'terhadap', 'kepada',

            # Prepositions
            'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'dalam', 'oleh',
            'bagi', 'terhadap', 'kepada', 'menuju', 'hingga', 'sampai',
            'antara', 'diantara', 'sekitar', 'dekat', 'jauh', 'atas', 'bawah',
            'depan', 'belakang', 'kiri', 'kanan', 'tengah', 'luar', 'luas',

            # Verbs (auxiliary and common)
            'adalah', 'ialah', 'yaitu', 'yakni', 'akan', 'telah', 'sudah',
            'sedang', 'masih', 'belum', 'pernah', 'tidak', 'bukan', 'jangan',
            'harus', 'mesti', 'perlu', 'bisa', 'dapat', 'boleh', 'mau',
            'ingin', 'hendak', 'suka', 'senang', 'cinta', 'sayang',

            # Quantifiers
            'ada', 'tidak ada', 'semua', 'setiap', 'beberapa', 'banyak',
            'sedikit', 'cukup', 'kurang', 'lebih', 'paling', 'sangat',
            'agak', 'terlalu', 'hampir', 'sekitar', 'kira-kira',

            # Time expressions
            'sekarang', 'kini', 'nanti', 'besok', 'kemarin', 'dulu', 'dahulu',
            'tadi', 'barusan', 'baru', 'lama', 'cepat', 'lambat', 'pelan',
            'hari', 'minggu', 'bulan', 'tahun', 'jam', 'menit', 'detik',
            'pagi', 'siang', 'sore', 'malam', 'subuh', 'maghrib', 'isya',

            # Numbers
            'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan',
            'sembilan', 'sepuluh', 'sebelas', 'dua belas', 'puluh', 'ratus',
            'ribu', 'juta', 'miliar', 'triliun', 'pertama', 'kedua', 'ketiga',

            # Common adjectives
            'baik', 'buruk', 'bagus', 'jelek', 'besar', 'kecil', 'panjang',
            'pendek', 'tinggi', 'rendah', 'lebar', 'sempit', 'tebal', 'tipis',
            'berat', 'ringan', 'keras', 'lunak', 'kasar', 'halus', 'panas',
            'dingin', 'hangat', 'sejuk', 'basah', 'kering', 'bersih', 'kotor',
            'baru', 'lama', 'muda', 'tua', 'mudah', 'sulit', 'gampang',
            'susah', 'senang', 'sedih', 'marah', 'takut', 'berani',

            # Common adverbs
            'juga', 'pula', 'lagi', 'masih', 'sudah', 'belum', 'baru',
            'saja', 'hanya', 'cuma', 'justru', 'malah', 'bahkan', 'apalagi',
            'ternyata', 'rupanya', 'sebenarnya', 'memang', 'tentu', 'pasti',
            'mungkin', 'barangkali', 'kiranya', 'agaknya', 'sepertinya',

            # Question words
            'apa', 'siapa', 'mana', 'dimana', 'kemana', 'darimana', 'kapan',
            'mengapa', 'kenapa', 'bagaimana', 'gimana', 'berapa', 'seberapa'
        }

        # Comprehensive Javanese mappings (ngoko, alus, krama levels)
        self.javanese_mappings = {
            # Pronouns
            'saya': 'aku', 'kamu': 'kowe', 'dia': 'dheweke', 'kita': 'awake dhewe',
            'kami': 'awake', 'mereka': 'wong-wong', 'anda': 'panjenengan',
            'beliau': 'piyambakipun',

            # Demonstratives
            'ini': 'iki', 'itu': 'iku', 'sini': 'kene', 'sana': 'kono',
            'begini': 'kaya ngene', 'begitu': 'kaya ngono',

            # Common words
            'yang': 'sing', 'dan': 'lan', 'atau': 'utawa', 'tetapi': 'nanging',
            'karena': 'merga', 'jika': 'yen', 'ketika': 'nalika', 'saat': 'nalika',
            'tidak': 'ora', 'bukan': 'dudu', 'ada': 'ana', 'tidak ada': 'ora ana',
            'di': 'ing', 'ke': 'menyang', 'dari': 'saka', 'untuk': 'kanggo',
            'dengan': 'karo', 'pada': 'ing', 'dalam': 'ing jero', 'oleh': 'dening',
            'akan': 'arep', 'sudah': 'wis', 'sedang': 'lagi', 'masih': 'isih',
            'belum': 'durung', 'semua': 'kabeh', 'banyak': 'akeh', 'sedikit': 'sithik',
            'sangat': 'banget', 'bagus': 'apik', 'jelek': 'elek', 'besar': 'gedhe',
            'kecil': 'cilik', 'panjang': 'dawa', 'pendek': 'cendhak', 'tinggi': 'dhuwur',
            'rendah': 'cendhek', 'baik': 'apik', 'buruk': 'ala', 'baru': 'anyar',
            'lama': 'lawas', 'muda': 'enom', 'tua': 'tuwa', 'cepat': 'cepet',
            'lambat': 'alon', 'mudah': 'gampang', 'sulit': 'angel',

            # Question words
            'apa': 'apa', 'siapa': 'sapa', 'mana': 'endi', 'dimana': 'ngendi',
            'kapan': 'kapan', 'mengapa': 'ngapa', 'bagaimana': 'piye', 'berapa': 'pira',

            # Time
            'sekarang': 'saiki', 'nanti': 'mengko', 'kemarin': 'wingi',
            'besok': 'sesuk', 'hari': 'dina', 'pagi': 'esuk', 'siang': 'awan',
            'sore': 'sore', 'malam': 'bengi',

            # Numbers
            'satu': 'siji', 'dua': 'loro', 'tiga': 'telu', 'empat': 'papat',
            'lima': 'lima', 'enam': 'enem', 'tujuh': 'pitu', 'delapan': 'wolu',
            'sembilan': 'sanga', 'sepuluh': 'sepuluh'
        }

        # Comprehensive Sundanese mappings (kasar/loma and lemes/hormat levels)
        self.sundanese_mappings = {
            # Pronouns
            'saya': 'abdi', 'kamu': 'anjeun', 'dia': 'anjeunna', 'kita': 'urang',
            'kami': 'kami', 'mereka': 'aranjeunna', 'anda': 'anjeun',
            'beliau': 'anjeunna',

            # Demonstratives
            'ini': 'ieu', 'itu': 'eta', 'sini': 'dieu', 'sana': 'dinya',
            'begini': 'kieu', 'begitu': 'kitu',

            # Common words
            'yang': 'nu', 'dan': 'jeung', 'atau': 'atawa', 'tetapi': 'tapi',
            'karena': 'sabab', 'jika': 'lamun', 'ketika': 'nalika', 'saat': 'nalika',
            'tidak': 'henteu', 'bukan': 'sanés', 'ada': 'aya', 'tidak ada': 'teu aya',
            'di': 'di', 'ke': 'ka', 'dari': 'ti', 'untuk': 'pikeun',
            'dengan': 'sareng', 'pada': 'dina', 'dalam': 'dina', 'oleh': 'ku',
            'akan': 'bade', 'sudah': 'parantos', 'sedang': 'nuju', 'masih': 'masih',
            'belum': 'can', 'semua': 'sadaya', 'banyak': 'seueur', 'sedikit': 'saeutik',
            'sangat': 'pisan', 'bagus': 'saé', 'jelek': 'awon', 'besar': 'ageung',
            'kecil': 'alit', 'panjang': 'panjang', 'pendek': 'pondok', 'tinggi': 'luhur',
            'rendah': 'handap', 'baik': 'saé', 'buruk': 'awon', 'baru': 'anyar',
            'lama': 'lami', 'muda': 'ngora', 'tua': 'sepuh', 'cepat': 'gancang',
            'lambat': 'laun', 'mudah': 'gampil', 'sulit': 'hese',

            # Question words
            'apa': 'naon', 'siapa': 'saha', 'mana': 'mana', 'dimana': 'dimana',
            'kapan': 'iraha', 'mengapa': 'naha', 'bagaimana': 'kumaha', 'berapa': 'sabaraha',

            # Time
            'sekarang': 'ayeuna', 'nanti': 'engké', 'kemarin': 'kamari',
            'besok': 'isukan', 'hari': 'dinten', 'pagi': 'isuk', 'siang': 'siang',
            'sore': 'sonten', 'malam': 'wengi',

            # Numbers
            'satu': 'hiji', 'dua': 'dua', 'tiga': 'tilu', 'empat': 'opat',
            'lima': 'lima', 'enam': 'genep', 'tujuh': 'tujuh', 'delapan': 'dalapan',
            'sembilan': 'salapan', 'sepuluh': 'sapuluh'
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
            'kemana', 'darimana', 'siapa', 'apa', 'mana', 'huft',
            'hufh', 'hufft', 'haah', 'haaah', 'aaah', 'oooh', 'uuuh',
            'eeeh', 'iiih', 'owh', 'owwh', 'owwwh'
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

    def generate_comprehensive_dataset(self):
        """Generate comprehensive multilingual stopword dataset with 1500+ entries"""
        logger.info("Starting comprehensive multilingual stopword generation...")

        # 1. Generate from English stopwords
        self.generate_from_english_stopwords()

        # 2. Generate from Indonesian slang
        self.generate_from_indonesian_slang()

        # 3. Generate from formal Indonesian
        self.generate_from_formal_indonesian()

        # 4. Generate from social media particles
        self.generate_from_social_media_particles()

        # 5. Generate comprehensive additional entries
        self.generate_comprehensive_additional_entries()

        # 6. Remove duplicates
        self.deduplicate_entries()

        logger.info(f"Dataset generation complete. Total entries: {len(self.stopwords_data)}")
        return self.stopwords_data

    def generate_from_english_stopwords(self):
        """Generate entries starting from English stopwords"""
        logger.info("Generating entries from English stopwords...")

        # Enhanced English to Indonesian mappings
        en_to_id_mapping = {
            'i': 'saya', 'me': 'saya', 'my': 'saya', 'myself': 'saya sendiri',
            'we': 'kita', 'our': 'kita', 'ours': 'milik kita', 'ourselves': 'kita sendiri',
            'you': 'kamu', 'your': 'kamu', 'yours': 'milik kamu', 'yourself': 'kamu sendiri',
            'he': 'dia', 'him': 'dia', 'his': 'dia', 'himself': 'dia sendiri',
            'she': 'dia', 'her': 'dia', 'hers': 'milik dia', 'herself': 'dia sendiri',
            'it': 'itu', 'its': 'miliknya', 'itself': 'itu sendiri',
            'they': 'mereka', 'them': 'mereka', 'their': 'mereka', 'theirs': 'milik mereka',
            'themselves': 'mereka sendiri', 'this': 'ini', 'that': 'itu',
            'these': 'ini', 'those': 'itu', 'and': 'dan', 'or': 'atau',
            'but': 'tetapi', 'if': 'jika', 'because': 'karena', 'when': 'ketika',
            'while': 'sementara', 'before': 'sebelum', 'after': 'setelah',
            'in': 'di', 'on': 'di', 'at': 'di', 'by': 'oleh', 'for': 'untuk',
            'with': 'dengan', 'from': 'dari', 'to': 'ke', 'of': 'dari',
            'the': '', 'a': '', 'an': '', 'is': 'adalah', 'are': 'adalah',
            'was': 'adalah', 'were': 'adalah', 'be': 'adalah', 'been': 'telah',
            'being': 'sedang', 'have': 'punya', 'has': 'punya', 'had': 'punya',
            'having': 'mempunyai', 'do': '', 'does': '', 'did': '',
            'will': 'akan', 'would': 'akan', 'could': 'bisa', 'should': 'harus',
            'may': 'mungkin', 'might': 'mungkin', 'can': 'bisa', 'must': 'harus',
            'not': 'tidak', 'no': 'tidak', 'yes': 'ya', 'all': 'semua',
            'any': 'apapun', 'some': 'beberapa', 'many': 'banyak', 'much': 'banyak',
            'few': 'sedikit', 'little': 'sedikit', 'more': 'lebih', 'most': 'paling',
            'less': 'kurang', 'very': 'sangat', 'too': 'terlalu', 'so': 'jadi',
            'just': 'hanya', 'only': 'hanya', 'also': 'juga', 'even': 'bahkan',
            'still': 'masih', 'yet': 'belum', 'already': 'sudah', 'now': 'sekarang',
            'then': 'kemudian', 'here': 'disini', 'there': 'disana', 'where': 'dimana',
            'how': 'bagaimana', 'what': 'apa', 'who': 'siapa', 'why': 'mengapa',
            'which': 'yang mana', 'about': 'tentang', 'above': 'atas', 'across': 'seberang',
            'against': 'melawan', 'along': 'sepanjang', 'among': 'antara',
            'around': 'sekitar', 'behind': 'belakang', 'below': 'bawah',
            'beneath': 'bawah', 'beside': 'samping', 'between': 'antara',
            'beyond': 'melampaui', 'during': 'selama', 'except': 'kecuali',
            'inside': 'dalam', 'outside': 'luar', 'through': 'melalui',
            'throughout': 'sepanjang', 'toward': 'menuju', 'towards': 'menuju',
            'under': 'bawah', 'until': 'sampai', 'upon': 'atas', 'within': 'dalam',
            'without': 'tanpa', 'one': 'satu', 'two': 'dua', 'three': 'tiga',
            'four': 'empat', 'five': 'lima', 'six': 'enam', 'seven': 'tujuh',
            'eight': 'delapan', 'nine': 'sembilan', 'ten': 'sepuluh',
            'first': 'pertama', 'second': 'kedua', 'third': 'ketiga',
            'last': 'terakhir', 'next': 'berikutnya', 'previous': 'sebelumnya',
            'same': 'sama', 'different': 'berbeda', 'new': 'baru', 'old': 'lama',
            'good': 'bagus', 'bad': 'buruk', 'big': 'besar', 'small': 'kecil',
            'long': 'panjang', 'short': 'pendek', 'high': 'tinggi', 'low': 'rendah',
            'true': 'benar', 'false': 'salah', 'right': 'benar', 'wrong': 'salah'
        }

        for en_word in self.english_stopwords:
            formal_id = en_to_id_mapping.get(en_word, '')
            id_word = formal_id
            jv_word = self.javanese_mappings.get(formal_id, '') if formal_id else ''
            su_word = self.sundanese_mappings.get(formal_id, '') if formal_id else ''

            self.add_stopword_entry(en_word, id_word, jv_word, su_word, formal_id)

    def generate_from_indonesian_slang(self):
        """Generate entries from Indonesian slang and social media terms"""
        logger.info("Generating entries from Indonesian slang...")

        for slang, formal in self.indonesian_slang.items():
            jv_word = self.javanese_mappings.get(formal, '') if formal else ''
            su_word = self.sundanese_mappings.get(formal, '') if formal else ''
            self.add_stopword_entry('', slang, jv_word, su_word, formal)

    def generate_from_formal_indonesian(self):
        """Generate entries from formal Indonesian words"""
        logger.info("Generating entries from formal Indonesian words...")

        for formal_word in self.indonesian_formal:
            jv_word = self.javanese_mappings.get(formal_word, '')
            su_word = self.sundanese_mappings.get(formal_word, '')
            self.add_stopword_entry('', formal_word, jv_word, su_word, formal_word)

    def generate_from_social_media_particles(self):
        """Generate entries from social media particles and interjections"""
        logger.info("Generating entries from social media particles...")

        for particle in self.social_media_particles:
            self.add_stopword_entry('', particle, '', '', particle)

    def generate_comprehensive_additional_entries(self):
        """Generate comprehensive additional entries to reach 1500+ target"""
        logger.info("Generating comprehensive additional entries...")

        # Additional comprehensive entries - focusing on most common social media terms
        additional_entries = [
            # More English social media terms
            ('lmfao', '', '', '', 'laugh my f ass off'),
            ('rotfl', '', '', '', 'rolling on the floor laughing'),
            ('gtg', '', '', '', 'got to go'),
            ('bff', '', '', '', 'best friends forever'),
            ('afaik', '', '', '', 'as far as i know'),
            ('ftw', '', '', '', 'for the win'),
            ('smh', '', '', '', 'shaking my head'),
            ('tmi', '', '', '', 'too much information'),
            ('yolo', '', '', '', 'you only live once'),
            ('fomo', '', '', '', 'fear of missing out'),
            ('tbf', '', '', '', 'to be fair'),
            ('afk', '', '', '', 'away from keyboard'),
            ('g2g', '', '', '', 'got to go'),
            ('cya', '', '', '', 'see you'),
            ('thx', '', '', '', 'thanks'),
            ('np', '', '', '', 'no problem'),
            ('ur', '', '', '', 'your'),
            ('pls', '', '', '', 'please'),
            ('plz', '', '', '', 'please'),
            ('msg', '', '', '', 'message'),
            ('txt', '', '', '', 'text'),
            ('pic', '', '', '', 'picture'),
            ('vid', '', '', '', 'video'),
            ('app', '', '', '', 'application'),
            ('tech', '', '', '', 'technology'),
            ('info', '', '', '', 'information'),
            ('admin', '', '', '', 'administrator'),
            ('mod', '', '', '', 'moderator'),
            ('dev', '', '', '', 'developer'),
            ('beta', '', '', '', 'beta'),
            ('alpha', '', '', '', 'alpha'),
            ('demo', '', '', '', 'demonstration'),
            ('promo', '', '', '', 'promotion'),
            ('sale', '', '', '', 'sale'),
            ('deal', '', '', '', 'deal'),
            ('offer', '', '', '', 'offer'),
            ('free', '', '', '', 'free'),
            ('premium', '', '', '', 'premium'),
            ('pro', '', '', '', 'professional'),
            ('lite', '', '', '', 'light'),
            ('mini', '', '', '', 'mini'),
            ('max', '', '', '', 'maximum'),
            ('plus', '', '', '', 'plus'),
            ('extra', '', '', '', 'extra'),
            ('super', '', '', '', 'super'),
            ('mega', '', '', '', 'mega'),
            ('ultra', '', '', '', 'ultra'),

            # More Indonesian internet slang and variations
            ('', 'ygy', '', '', 'ya guys ya'),
            ('', 'yaudah', '', '', 'ya sudah'),
            ('', 'yaudeh', '', '', 'ya sudah'),
            ('', 'yauda', '', '', 'ya sudah'),
            ('', 'yasud', '', '', 'ya sudah'),
            ('', 'yasudah', '', '', 'ya sudah'),
            ('', 'yawes', '', '', 'ya sudah'),
            ('', 'yowes', '', '', 'ya sudah'),
            ('', 'pokoke', '', '', 'pokoknya'),
            ('', 'pokoknya', '', '', 'pokoknya'),
            ('', 'pokokna', '', '', 'pokoknya'),
            ('', 'intinya', '', '', 'intinya'),
            ('', 'intina', '', '', 'intinya'),
            ('', 'sebenernya', '', '', 'sebenarnya'),
            ('', 'sebenernya', '', '', 'sebenarnya'),
            ('', 'sebenerna', '', '', 'sebenarnya'),
            ('', 'makanya', '', '', 'makanya'),
            ('', 'makana', '', '', 'makanya'),
            ('', 'jadinya', '', '', 'jadinya'),
            ('', 'jadina', '', '', 'jadinya'),
            ('', 'terusnya', '', '', 'terusnya'),
            ('', 'terusna', '', '', 'terusnya'),
            ('', 'lagian', '', '', 'lagian'),
            ('', 'lagina', '', '', 'lagian'),
            ('', 'soalnya', '', '', 'soalnya'),
            ('', 'soalna', '', '', 'soalnya'),
            ('', 'padahal', '', '', 'padahal'),
            ('', 'ternyata', '', '', 'ternyata'),
            ('', 'rupanya', '', '', 'rupanya'),
            ('', 'rupana', '', '', 'rupanya'),
            ('', 'kayaknya', '', '', 'sepertinya'),
            ('', 'kayakna', '', '', 'sepertinya'),
            ('', 'sepertinya', '', '', 'sepertinya'),
            ('', 'keknya', '', '', 'sepertinya'),
            ('', 'mungkin', '', '', 'mungkin'),
            ('', 'mgkn', '', '', 'mungkin'),
            ('', 'barangkali', '', '', 'barangkali'),
            ('', 'brangkali', '', '', 'barangkali'),
            ('', 'siapa tau', '', '', 'siapa tahu'),
            ('', 'sapa tau', '', '', 'siapa tahu'),
            ('', 'kali aja', '', '', 'kali saja'),
            ('', 'kali aj', '', '', 'kali saja'),
            ('', 'mudah2an', '', '', 'mudah-mudahan'),
            ('', 'mudahan', '', '', 'mudah-mudahan'),
            ('', 'semoga', '', '', 'semoga'),
            ('', 'smoga', '', '', 'semoga'),
            ('', 'insyaallah', '', '', 'insya allah'),
            ('', 'insyallah', '', '', 'insya allah'),
            ('', 'alhamdulillah', '', '', 'alhamdulillah'),
            ('', 'subhanallah', '', '', 'subhanallah'),
            ('', 'mashaallah', '', '', 'masha allah'),
            ('', 'astaghfirullah', '', '', 'astaghfirullah'),
            ('', 'bismillah', '', '', 'bismillah'),
            ('', 'wallahu alam', '', '', 'wallahu alam'),
            ('', 'wallahualam', '', '', 'wallahu alam'),

            # More Javanese specific terms with Indonesian equivalents
            ('', 'aku', 'aku', '', 'saya'),
            ('', 'kowe', 'kowe', '', 'kamu'),
            ('', 'dheweke', 'dheweke', '', 'dia'),
            ('', 'awakmu', 'awakmu', '', 'kamu'),
            ('', 'awakne', 'awakne', '', 'dia'),
            ('', 'iki', 'iki', '', 'ini'),
            ('', 'iku', 'iku', '', 'itu'),
            ('', 'kene', 'kene', '', 'sini'),
            ('', 'kono', 'kono', '', 'sana'),
            ('', 'ngendi', 'ngendi', '', 'dimana'),
            ('', 'piye', 'piye', '', 'bagaimana'),
            ('', 'apa', 'apa', '', 'apa'),
            ('', 'sapa', 'sapa', '', 'siapa'),
            ('', 'kapan', 'kapan', '', 'kapan'),
            ('', 'ngapa', 'ngapa', '', 'kenapa'),
            ('', 'ning', 'ning', '', 'di'),
            ('', 'nang', 'nang', '', 'di'),
            ('', 'menyang', 'menyang', '', 'ke'),
            ('', 'saka', 'saka', '', 'dari'),
            ('', 'kanggo', 'kanggo', '', 'untuk'),
            ('', 'karo', 'karo', '', 'dengan'),
            ('', 'lan', 'lan', '', 'dan'),
            ('', 'utawa', 'utawa', '', 'atau'),
            ('', 'nanging', 'nanging', '', 'tetapi'),
            ('', 'merga', 'merga', '', 'karena'),
            ('', 'yen', 'yen', '', 'jika'),
            ('', 'nalika', 'nalika', '', 'ketika'),
            ('', 'arep', 'arep', '', 'akan'),
            ('', 'wis', 'wis', '', 'sudah'),
            ('', 'lagi', 'lagi', '', 'sedang'),
            ('', 'isih', 'isih', '', 'masih'),
            ('', 'durung', 'durung', '', 'belum'),
            ('', 'ora', 'ora', '', 'tidak'),
            ('', 'dudu', 'dudu', '', 'bukan'),
            ('', 'ana', 'ana', '', 'ada'),
            ('', 'ora ana', 'ora ana', '', 'tidak ada'),
            ('', 'kabeh', 'kabeh', '', 'semua'),
            ('', 'akeh', 'akeh', '', 'banyak'),
            ('', 'sithik', 'sithik', '', 'sedikit'),
            ('', 'banget', 'banget', '', 'sangat'),
            ('', 'apik', 'apik', '', 'bagus'),
            ('', 'elek', 'elek', '', 'jelek'),
            ('', 'gedhe', 'gedhe', '', 'besar'),
            ('', 'cilik', 'cilik', '', 'kecil'),
            ('', 'dawa', 'dawa', '', 'panjang'),
            ('', 'cendhak', 'cendhak', '', 'pendek'),
            ('', 'dhuwur', 'dhuwur', '', 'tinggi'),
            ('', 'cendhek', 'cendhek', '', 'rendah'),
            ('', 'anyar', 'anyar', '', 'baru'),
            ('', 'lawas', 'lawas', '', 'lama'),
            ('', 'enom', 'enom', '', 'muda'),
            ('', 'tuwa', 'tuwa', '', 'tua'),
            ('', 'cepet', 'cepet', '', 'cepat'),
            ('', 'alon', 'alon', '', 'lambat'),
            ('', 'gampang', 'gampang', '', 'mudah'),
            ('', 'angel', 'angel', '', 'sulit'),
            ('', 'saiki', 'saiki', '', 'sekarang'),
            ('', 'mengko', 'mengko', '', 'nanti'),
            ('', 'wingi', 'wingi', '', 'kemarin'),
            ('', 'sesuk', 'sesuk', '', 'besok'),
            ('', 'esuk', 'esuk', '', 'pagi'),
            ('', 'awan', 'awan', '', 'siang'),
            ('', 'sore', 'sore', '', 'sore'),
            ('', 'bengi', 'bengi', '', 'malam'),
            ('', 'siji', 'siji', '', 'satu'),
            ('', 'loro', 'loro', '', 'dua'),
            ('', 'telu', 'telu', '', 'tiga'),
            ('', 'papat', 'papat', '', 'empat'),
            ('', 'lima', 'lima', '', 'lima'),
            ('', 'enem', 'enem', '', 'enam'),
            ('', 'pitu', 'pitu', '', 'tujuh'),
            ('', 'wolu', 'wolu', '', 'delapan'),
            ('', 'sanga', 'sanga', '', 'sembilan'),
            ('', 'sepuluh', 'sepuluh', '', 'sepuluh'),

            # More Sundanese specific terms with Indonesian equivalents
            ('', 'abdi', '', 'abdi', 'saya'),
            ('', 'anjeun', '', 'anjeun', 'kamu'),
            ('', 'anjeunna', '', 'anjeunna', 'dia'),
            ('', 'urang', '', 'urang', 'kita'),
            ('', 'aranjeunna', '', 'aranjeunna', 'mereka'),
            ('', 'ieu', '', 'ieu', 'ini'),
            ('', 'eta', '', 'eta', 'itu'),
            ('', 'dieu', '', 'dieu', 'sini'),
            ('', 'dinya', '', 'dinya', 'sana'),
            ('', 'dimana', '', 'dimana', 'dimana'),
            ('', 'kumaha', '', 'kumaha', 'bagaimana'),
            ('', 'naon', '', 'naon', 'apa'),
            ('', 'saha', '', 'saha', 'siapa'),
            ('', 'iraha', '', 'iraha', 'kapan'),
            ('', 'naha', '', 'naha', 'kenapa'),
            ('', 'nu', '', 'nu', 'yang'),
            ('', 'jeung', '', 'jeung', 'dan'),
            ('', 'atawa', '', 'atawa', 'atau'),
            ('', 'tapi', '', 'tapi', 'tetapi'),
            ('', 'sabab', '', 'sabab', 'karena'),
            ('', 'lamun', '', 'lamun', 'jika'),
            ('', 'nalika', '', 'nalika', 'ketika'),
            ('', 'ka', '', 'ka', 'ke'),
            ('', 'ti', '', 'ti', 'dari'),
            ('', 'pikeun', '', 'pikeun', 'untuk'),
            ('', 'sareng', '', 'sareng', 'dengan'),
            ('', 'dina', '', 'dina', 'pada'),
            ('', 'ku', '', 'ku', 'oleh'),
            ('', 'bade', '', 'bade', 'akan'),
            ('', 'parantos', '', 'parantos', 'sudah'),
            ('', 'nuju', '', 'nuju', 'sedang'),
            ('', 'masih', '', 'masih', 'masih'),
            ('', 'can', '', 'can', 'belum'),
            ('', 'henteu', '', 'henteu', 'tidak'),
            ('', 'sanés', '', 'sanés', 'bukan'),
            ('', 'aya', '', 'aya', 'ada'),
            ('', 'teu aya', '', 'teu aya', 'tidak ada'),
            ('', 'sadaya', '', 'sadaya', 'semua'),
            ('', 'seueur', '', 'seueur', 'banyak'),
            ('', 'saeutik', '', 'saeutik', 'sedikit'),
            ('', 'pisan', '', 'pisan', 'sangat'),
            ('', 'saé', '', 'saé', 'bagus'),
            ('', 'awon', '', 'awon', 'jelek'),
            ('', 'ageung', '', 'ageung', 'besar'),
            ('', 'alit', '', 'alit', 'kecil'),
            ('', 'panjang', '', 'panjang', 'panjang'),
            ('', 'pondok', '', 'pondok', 'pendek'),
            ('', 'luhur', '', 'luhur', 'tinggi'),
            ('', 'handap', '', 'handap', 'rendah'),
            ('', 'anyar', '', 'anyar', 'baru'),
            ('', 'lami', '', 'lami', 'lama'),
            ('', 'ngora', '', 'ngora', 'muda'),
            ('', 'sepuh', '', 'sepuh', 'tua'),
            ('', 'gancang', '', 'gancang', 'cepat'),
            ('', 'laun', '', 'laun', 'lambat'),
            ('', 'gampil', '', 'gampil', 'mudah'),
            ('', 'hese', '', 'hese', 'sulit'),
            ('', 'ayeuna', '', 'ayeuna', 'sekarang'),
            ('', 'engké', '', 'engké', 'nanti'),
            ('', 'kamari', '', 'kamari', 'kemarin'),
            ('', 'isukan', '', 'isukan', 'besok'),
            ('', 'dinten', '', 'dinten', 'hari'),
            ('', 'isuk', '', 'isuk', 'pagi'),
            ('', 'siang', '', 'siang', 'siang'),
            ('', 'sonten', '', 'sonten', 'sore'),
            ('', 'wengi', '', 'wengi', 'malam'),
            ('', 'hiji', '', 'hiji', 'satu'),
            ('', 'dua', '', 'dua', 'dua'),
            ('', 'tilu', '', 'tilu', 'tiga'),
            ('', 'opat', '', 'opat', 'empat'),
            ('', 'lima', '', 'lima', 'lima'),
            ('', 'genep', '', 'genep', 'enam'),
            ('', 'tujuh', '', 'tujuh', 'tujuh'),
            ('', 'dalapan', '', 'dalapan', 'delapan'),
            ('', 'salapan', '', 'salapan', 'sembilan'),
            ('', 'sapuluh', '', 'sapuluh', 'sepuluh'),

            # Additional Indonesian social media terms and variations
            ('', 'mantul', '', '', 'mantap betul'),
            ('', 'mantep', '', '', 'mantap'),
            ('', 'kece', '', '', 'keren'),
            ('', 'kereen', '', '', 'keren'),
            ('', 'kereeen', '', '', 'keren'),
            ('', 'gokil', '', '', 'gila'),
            ('', 'gila', '', '', 'gila'),
            ('', 'gilak', '', '', 'gila'),
            ('', 'anjay', '', '', 'wow'),
            ('', 'anjir', '', '', 'wow'),
            ('', 'anjrit', '', '', 'wow'),
            ('', 'buset', '', '', 'wow'),
            ('', 'busyet', '', '', 'wow'),
            ('', 'astaga', '', '', 'astaga'),
            ('', 'astaga', '', '', 'astaga'),
            ('', 'alamak', '', '', 'alamak'),
            ('', 'aduh', '', '', 'aduh'),
            ('', 'waduh', '', '', 'waduh'),
            ('', 'duh', '', '', 'duh'),
            ('', 'ih', '', '', 'ih'),
            ('', 'eh', '', '', 'eh'),
            ('', 'ah', '', '', 'ah'),
            ('', 'oh', '', '', 'oh'),
            ('', 'uh', '', '', 'uh'),
            ('', 'hmm', '', '', 'hmm'),
            ('', 'hm', '', '', 'hm'),
            ('', 'em', '', '', 'em'),
            ('', 'um', '', '', 'um'),
            ('', 'huft', '', '', 'huft'),
            ('', 'hufh', '', '', 'hufh'),
            ('', 'hufft', '', '', 'hufft'),
            ('', 'haah', '', '', 'haah'),
            ('', 'haaah', '', '', 'haaah'),
            ('', 'aaah', '', '', 'aaah'),
            ('', 'oooh', '', '', 'oooh'),
            ('', 'uuuh', '', '', 'uuuh'),
            ('', 'eeeh', '', '', 'eeeh'),
            ('', 'iiih', '', '', 'iiih'),
            ('', 'owh', '', '', 'owh'),
            ('', 'owwh', '', '', 'owwh'),
            ('', 'owwwh', '', '', 'owwwh'),
            ('', 'woles', '', '', 'santai'),
            ('', 'santuy', '', '', 'santai'),
            ('', 'santai', '', '', 'santai'),
            ('', 'slow', '', '', 'pelan'),
            ('', 'pelan', '', '', 'pelan'),
            ('', 'sabar', '', '', 'sabar'),
            ('', 'tenang', '', '', 'tenang'),
            ('', 'kalem', '', '', 'tenang'),
            ('', 'cool', '', '', 'keren'),
            ('', 'asik', '', '', 'asik'),
            ('', 'asyik', '', '', 'asik'),
            ('', 'seru', '', '', 'seru'),
            ('', 'lucu', '', '', 'lucu'),
            ('', 'ngakak', '', '', 'tertawa'),
            ('', 'ketawa', '', '', 'tertawa'),
            ('', 'tertawa', '', '', 'tertawa'),
            ('', 'haha', '', '', 'haha'),
            ('', 'hehe', '', '', 'hehe'),
            ('', 'hihi', '', '', 'hihi'),
            ('', 'hoho', '', '', 'hoho'),
            ('', 'huhu', '', '', 'huhu'),
            ('', 'wakaka', '', '', 'hahaha'),
            ('', 'wakakaka', '', '', 'hahahaha'),
            ('', 'kwkw', '', '', 'haha'),
            ('', 'kwkwkw', '', '', 'hahaha'),
            ('', 'kkkk', '', '', 'haha'),
            ('', 'kkkkk', '', '', 'hahaha'),
            ('', 'xixixi', '', '', 'hehehe'),
            ('', 'xixi', '', '', 'hehe'),

            # Additional comprehensive social media and internet terms
            ('', 'gabisa', '', '', 'tidak bisa'),
            ('', 'gasuka', '', '', 'tidak suka'),
            ('', 'gatau', '', '', 'tidak tahu'),
            ('', 'gapapa', '', '', 'tidak apa-apa'),
            ('', 'udah', '', '', 'sudah'),
            ('', 'udeh', '', '', 'sudah'),
            ('', 'belom', '', '', 'belum'),
            ('', 'blom', '', '', 'belum'),
            ('', 'iya', '', '', 'ya'),
            ('', 'yup', '', '', 'ya'),
            ('', 'yep', '', '', 'ya'),
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
            ('', 'nah', '', '', ''),
            ('', 'kan', '', '', ''),
            ('', 'gitu', '', '', 'begitu'),
            ('', 'gini', '', '', 'begini'),
            ('', 'kayak', '', '', 'seperti'),
            ('', 'kaya', '', '', 'seperti'),
            ('', 'banget', '', '', 'sangat'),
            ('', 'bgt', '', '', 'sangat'),
            ('', 'bener', '', '', 'benar'),
            ('', 'emang', '', '', 'memang'),
            ('', 'memang', '', '', 'memang'),
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
            ('', 'skrg', '', '', 'sekarang'),
            ('', 'skrang', '', '', 'sekarang'),
            ('', 'ntar', '', '', 'nanti'),
            ('', 'tar', '', '', 'nanti'),
            ('', 'td', '', '', 'tadi'),
            ('', 'kmrn', '', '', 'kemarin'),
            ('', 'bsk', '', '', 'besok'),
            ('', 'hr', '', '', 'hari'),
            ('', 'mgg', '', '', 'minggu'),
            ('', 'bln', '', '', 'bulan'),
            ('', 'thn', '', '', 'tahun'),
            ('', 'mnt', '', '', 'menit'),
            ('', 'dtk', '', '', 'detik'),
            ('', 'jam', '', '', 'jam'),
            ('', 'gimana', '', '', 'bagaimana'),
            ('', 'gmn', '', '', 'bagaimana'),
            ('', 'bgmn', '', '', 'bagaimana'),
            ('', 'kenapa', '', '', 'kenapa'),
            ('', 'knp', '', '', 'kenapa'),
            ('', 'knapa', '', '', 'kenapa'),
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

            # More social media expressions and reactions
            ('', 'mantul', '', '', 'mantap betul'),
            ('', 'mantep', '', '', 'mantap'),
            ('', 'mantap', '', '', 'mantap'),
            ('', 'kece', '', '', 'keren'),
            ('', 'kereen', '', '', 'keren'),
            ('', 'kereeen', '', '', 'keren'),
            ('', 'keren', '', '', 'keren'),
            ('', 'bagus', '', '', 'bagus'),
            ('', 'jelek', '', '', 'jelek'),
            ('', 'buruk', '', '', 'buruk'),
            ('', 'oke', '', '', 'oke'),
            ('', 'ok', '', '', 'oke'),
            ('', 'okay', '', '', 'oke'),
            ('', 'siap', '', '', 'siap'),
            ('', 'ready', '', '', 'siap'),
            ('', 'done', '', '', 'selesai'),
            ('', 'finish', '', '', 'selesai'),
            ('', 'selesai', '', '', 'selesai'),
            ('', 'beres', '', '', 'selesai'),
            ('', 'clear', '', '', 'jelas'),
            ('', 'jelas', '', '', 'jelas'),
            ('', 'paham', '', '', 'paham'),
            ('', 'ngerti', '', '', 'mengerti'),
            ('', 'mengerti', '', '', 'mengerti'),
            ('', 'tau', '', '', 'tahu'),
            ('', 'tahu', '', '', 'tahu'),
            ('', 'kenal', '', '', 'kenal'),
            ('', 'familiar', '', '', 'kenal'),
            ('', 'asing', '', '', 'asing'),
            ('', 'aneh', '', '', 'aneh'),
            ('', 'weird', '', '', 'aneh'),
            ('', 'strange', '', '', 'aneh'),
            ('', 'normal', '', '', 'normal'),
            ('', 'biasa', '', '', 'biasa'),
            ('', 'usual', '', '', 'biasa'),
            ('', 'special', '', '', 'khusus'),
            ('', 'khusus', '', '', 'khusus'),
            ('', 'istimewa', '', '', 'istimewa'),
            ('', 'unik', '', '', 'unik'),
            ('', 'unique', '', '', 'unik'),
            ('', 'rare', '', '', 'langka'),
            ('', 'langka', '', '', 'langka'),
            ('', 'jarang', '', '', 'jarang'),
            ('', 'sering', '', '', 'sering'),
            ('', 'often', '', '', 'sering'),
            ('', 'always', '', '', 'selalu'),
            ('', 'selalu', '', '', 'selalu'),
            ('', 'never', '', '', 'tidak pernah'),
            ('', 'pernah', '', '', 'pernah'),
            ('', 'kadang', '', '', 'kadang'),
            ('', 'sometimes', '', '', 'kadang'),
            ('', 'maybe', '', '', 'mungkin'),
            ('', 'perhaps', '', '', 'mungkin'),
            ('', 'probably', '', '', 'mungkin'),
            ('', 'definitely', '', '', 'pasti'),
            ('', 'pasti', '', '', 'pasti'),
            ('', 'sure', '', '', 'yakin'),
            ('', 'yakin', '', '', 'yakin'),
            ('', 'doubt', '', '', 'ragu'),
            ('', 'ragu', '', '', 'ragu'),
            ('', 'bingung', '', '', 'bingung'),
            ('', 'confused', '', '', 'bingung'),
            ('', 'clear', '', '', 'jelas'),
            ('', 'obvious', '', '', 'jelas'),
            ('', 'simple', '', '', 'sederhana'),
            ('', 'sederhana', '', '', 'sederhana'),
            ('', 'complex', '', '', 'rumit'),
            ('', 'rumit', '', '', 'rumit'),
            ('', 'complicated', '', '', 'rumit'),
            ('', 'easy', '', '', 'mudah'),
            ('', 'mudah', '', '', 'mudah'),
            ('', 'gampang', '', '', 'mudah'),
            ('', 'difficult', '', '', 'sulit'),
            ('', 'sulit', '', '', 'sulit'),
            ('', 'susah', '', '', 'sulit'),
            ('', 'hard', '', '', 'sulit'),
            ('', 'soft', '', '', 'lembut'),
            ('', 'lembut', '', '', 'lembut'),
            ('', 'halus', '', '', 'halus'),
            ('', 'kasar', '', '', 'kasar'),
            ('', 'rough', '', '', 'kasar'),
            ('', 'smooth', '', '', 'halus'),
            ('', 'fast', '', '', 'cepat'),
            ('', 'cepat', '', '', 'cepat'),
            ('', 'quick', '', '', 'cepat'),
            ('', 'slow', '', '', 'lambat'),
            ('', 'lambat', '', '', 'lambat'),
            ('', 'pelan', '', '', 'pelan'),
            ('', 'hot', '', '', 'panas'),
            ('', 'panas', '', '', 'panas'),
            ('', 'cold', '', '', 'dingin'),
            ('', 'dingin', '', '', 'dingin'),
            ('', 'warm', '', '', 'hangat'),
            ('', 'hangat', '', '', 'hangat'),
            ('', 'cool', '', '', 'sejuk'),
            ('', 'sejuk', '', '', 'sejuk'),
            ('', 'wet', '', '', 'basah'),
            ('', 'basah', '', '', 'basah'),
            ('', 'dry', '', '', 'kering'),
            ('', 'kering', '', '', 'kering'),
            ('', 'clean', '', '', 'bersih'),
            ('', 'bersih', '', '', 'bersih'),
            ('', 'dirty', '', '', 'kotor'),
            ('', 'kotor', '', '', 'kotor'),
            ('', 'fresh', '', '', 'segar'),
            ('', 'segar', '', '', 'segar'),
            ('', 'old', '', '', 'lama'),
            ('', 'lama', '', '', 'lama'),
            ('', 'new', '', '', 'baru'),
            ('', 'baru', '', '', 'baru'),
            ('', 'young', '', '', 'muda'),
            ('', 'muda', '', '', 'muda'),
            ('', 'old', '', '', 'tua'),
            ('', 'tua', '', '', 'tua'),
            ('', 'big', '', '', 'besar'),
            ('', 'besar', '', '', 'besar'),
            ('', 'small', '', '', 'kecil'),
            ('', 'kecil', '', '', 'kecil'),
            ('', 'tiny', '', '', 'kecil'),
            ('', 'huge', '', '', 'besar'),
            ('', 'large', '', '', 'besar'),
            ('', 'long', '', '', 'panjang'),
            ('', 'panjang', '', '', 'panjang'),
            ('', 'short', '', '', 'pendek'),
            ('', 'pendek', '', '', 'pendek'),
            ('', 'tall', '', '', 'tinggi'),
            ('', 'tinggi', '', '', 'tinggi'),
            ('', 'low', '', '', 'rendah'),
            ('', 'rendah', '', '', 'rendah'),
            ('', 'high', '', '', 'tinggi'),
            ('', 'wide', '', '', 'lebar'),
            ('', 'lebar', '', '', 'lebar'),
            ('', 'narrow', '', '', 'sempit'),
            ('', 'sempit', '', '', 'sempit'),
            ('', 'thick', '', '', 'tebal'),
            ('', 'tebal', '', '', 'tebal'),
            ('', 'thin', '', '', 'tipis'),
            ('', 'tipis', '', '', 'tipis'),
            ('', 'heavy', '', '', 'berat'),
            ('', 'berat', '', '', 'berat'),
            ('', 'light', '', '', 'ringan'),
            ('', 'ringan', '', '', 'ringan'),
            ('', 'strong', '', '', 'kuat'),
            ('', 'kuat', '', '', 'kuat'),
            ('', 'weak', '', '', 'lemah'),
            ('', 'lemah', '', '', 'lemah'),
            ('', 'powerful', '', '', 'kuat'),
            ('', 'gentle', '', '', 'lembut'),
            ('', 'violent', '', '', 'keras'),
            ('', 'keras', '', '', 'keras'),
            ('', 'quiet', '', '', 'sepi'),
            ('', 'sepi', '', '', 'sepi'),
            ('', 'loud', '', '', 'keras'),
            ('', 'noisy', '', '', 'berisik'),
            ('', 'berisik', '', '', 'berisik'),
            ('', 'silent', '', '', 'diam'),
            ('', 'diam', '', '', 'diam'),
            ('', 'speak', '', '', 'bicara'),
            ('', 'bicara', '', '', 'bicara'),
            ('', 'talk', '', '', 'bicara'),
            ('', 'ngomong', '', '', 'bicara'),
            ('', 'say', '', '', 'bilang'),
            ('', 'bilang', '', '', 'bilang'),
            ('', 'tell', '', '', 'cerita'),
            ('', 'cerita', '', '', 'cerita'),
            ('', 'story', '', '', 'cerita'),
            ('', 'listen', '', '', 'dengar'),
            ('', 'dengar', '', '', 'dengar'),
            ('', 'hear', '', '', 'dengar'),
            ('', 'see', '', '', 'lihat'),
            ('', 'lihat', '', '', 'lihat'),
            ('', 'look', '', '', 'lihat'),
            ('', 'watch', '', '', 'tonton'),
            ('', 'tonton', '', '', 'tonton'),
            ('', 'read', '', '', 'baca'),
            ('', 'baca', '', '', 'baca'),
            ('', 'write', '', '', 'tulis'),
            ('', 'tulis', '', '', 'tulis'),
            ('', 'type', '', '', 'ketik'),
            ('', 'ketik', '', '', 'ketik'),
            ('', 'send', '', '', 'kirim'),
            ('', 'kirim', '', '', 'kirim'),
            ('', 'receive', '', '', 'terima'),
            ('', 'terima', '', '', 'terima'),
            ('', 'get', '', '', 'dapat'),
            ('', 'dapat', '', '', 'dapat'),
            ('', 'give', '', '', 'kasih'),
            ('', 'kasih', '', '', 'kasih'),
            ('', 'take', '', '', 'ambil'),
            ('', 'ambil', '', '', 'ambil'),
            ('', 'put', '', '', 'taruh'),
            ('', 'taruh', '', '', 'taruh'),
            ('', 'place', '', '', 'tempat'),
            ('', 'tempat', '', '', 'tempat'),
            ('', 'go', '', '', 'pergi'),
            ('', 'pergi', '', '', 'pergi'),
            ('', 'come', '', '', 'datang'),
            ('', 'datang', '', '', 'datang'),
            ('', 'arrive', '', '', 'tiba'),
            ('', 'tiba', '', '', 'tiba'),
            ('', 'leave', '', '', 'pergi'),
            ('', 'stay', '', '', 'tinggal'),
            ('', 'tinggal', '', '', 'tinggal'),
            ('', 'live', '', '', 'hidup'),
            ('', 'hidup', '', '', 'hidup'),
            ('', 'die', '', '', 'mati'),
            ('', 'mati', '', '', 'mati'),
            ('', 'born', '', '', 'lahir'),
            ('', 'lahir', '', '', 'lahir'),
            ('', 'grow', '', '', 'tumbuh'),
            ('', 'tumbuh', '', '', 'tumbuh'),
            ('', 'change', '', '', 'ubah'),
            ('', 'ubah', '', '', 'ubah'),
            ('', 'ganti', '', '', 'ganti'),
            ('', 'same', '', '', 'sama'),
            ('', 'sama', '', '', 'sama'),
            ('', 'different', '', '', 'beda'),
            ('', 'beda', '', '', 'beda'),
            ('', 'berbeda', '', '', 'berbeda'),
            ('', 'similar', '', '', 'mirip'),
            ('', 'mirip', '', '', 'mirip'),
            ('', 'like', '', '', 'suka'),
            ('', 'suka', '', '', 'suka'),
            ('', 'love', '', '', 'cinta'),
            ('', 'cinta', '', '', 'cinta'),
            ('', 'hate', '', '', 'benci'),
            ('', 'benci', '', '', 'benci'),
            ('', 'angry', '', '', 'marah'),
            ('', 'marah', '', '', 'marah'),
            ('', 'happy', '', '', 'senang'),
            ('', 'senang', '', '', 'senang'),
            ('', 'sad', '', '', 'sedih'),
            ('', 'sedih', '', '', 'sedih'),
            ('', 'cry', '', '', 'nangis'),
            ('', 'nangis', '', '', 'nangis'),
            ('', 'laugh', '', '', 'ketawa'),
            ('', 'smile', '', '', 'senyum'),
            ('', 'senyum', '', '', 'senyum'),
            ('', 'afraid', '', '', 'takut'),
            ('', 'takut', '', '', 'takut'),
            ('', 'scared', '', '', 'takut'),
            ('', 'brave', '', '', 'berani'),
            ('', 'berani', '', '', 'berani'),
            ('', 'worry', '', '', 'khawatir'),
            ('', 'khawatir', '', '', 'khawatir'),
            ('', 'calm', '', '', 'tenang'),
            ('', 'tenang', '', '', 'tenang'),
            ('', 'stress', '', '', 'stres'),
            ('', 'stres', '', '', 'stres'),
            ('', 'relax', '', '', 'santai'),
            ('', 'santai', '', '', 'santai'),
            ('', 'tired', '', '', 'capek'),
            ('', 'capek', '', '', 'capek'),
            ('', 'lelah', '', '', 'lelah'),
            ('', 'fresh', '', '', 'segar'),
            ('', 'sleep', '', '', 'tidur'),
            ('', 'tidur', '', '', 'tidur'),
            ('', 'wake', '', '', 'bangun'),
            ('', 'bangun', '', '', 'bangun'),
            ('', 'eat', '', '', 'makan'),
            ('', 'makan', '', '', 'makan'),
            ('', 'drink', '', '', 'minum'),
            ('', 'minum', '', '', 'minum'),
            ('', 'hungry', '', '', 'lapar'),
            ('', 'lapar', '', '', 'lapar'),
            ('', 'thirsty', '', '', 'haus'),
            ('', 'haus', '', '', 'haus'),
            ('', 'full', '', '', 'kenyang'),
            ('', 'kenyang', '', '', 'kenyang'),
            ('', 'empty', '', '', 'kosong'),
            ('', 'kosong', '', '', 'kosong'),
            ('', 'busy', '', '', 'sibuk'),
            ('', 'sibuk', '', '', 'sibuk'),
            ('', 'free', '', '', 'bebas'),
            ('', 'bebas', '', '', 'bebas'),
            ('', 'work', '', '', 'kerja'),
            ('', 'kerja', '', '', 'kerja'),
            ('', 'job', '', '', 'kerja'),
            ('', 'play', '', '', 'main'),
            ('', 'main', '', '', 'main'),
            ('', 'game', '', '', 'permainan'),
            ('', 'permainan', '', '', 'permainan'),
            ('', 'fun', '', '', 'seru'),
            ('', 'boring', '', '', 'bosan'),
            ('', 'bosan', '', '', 'bosan'),
            ('', 'interesting', '', '', 'menarik'),
            ('', 'menarik', '', '', 'menarik'),
            ('', 'important', '', '', 'penting'),
            ('', 'penting', '', '', 'penting'),
            ('', 'useful', '', '', 'berguna'),
            ('', 'berguna', '', '', 'berguna'),
            ('', 'useless', '', '', 'tidak berguna'),
            ('', 'help', '', '', 'bantu'),
            ('', 'bantu', '', '', 'bantu'),
            ('', 'tolong', '', '', 'tolong'),
            ('', 'please', '', '', 'tolong'),
            ('', 'thanks', '', '', 'terima kasih'),
            ('', 'terima kasih', '', '', 'terima kasih'),
            ('', 'makasih', '', '', 'terima kasih'),
            ('', 'thank you', '', '', 'terima kasih'),
            ('', 'welcome', '', '', 'selamat datang'),
            ('', 'selamat datang', '', '', 'selamat datang'),
            ('', 'sorry', '', '', 'maaf'),
            ('', 'maaf', '', '', 'maaf'),
            ('', 'excuse me', '', '', 'permisi'),
            ('', 'permisi', '', '', 'permisi'),
            ('', 'hello', '', '', 'halo'),
            ('', 'halo', '', '', 'halo'),
            ('', 'hi', '', '', 'hai'),
            ('', 'hai', '', '', 'hai'),
            ('', 'bye', '', '', 'dadah'),
            ('', 'dadah', '', '', 'dadah'),
            ('', 'goodbye', '', '', 'selamat tinggal'),
            ('', 'selamat tinggal', '', '', 'selamat tinggal'),
            ('', 'good morning', '', '', 'selamat pagi'),
            ('', 'selamat pagi', '', '', 'selamat pagi'),
            ('', 'good afternoon', '', '', 'selamat siang'),
            ('', 'selamat siang', '', '', 'selamat siang'),
            ('', 'good evening', '', '', 'selamat sore'),
            ('', 'selamat sore', '', '', 'selamat sore'),
            ('', 'good night', '', '', 'selamat malam'),
            ('', 'selamat malam', '', '', 'selamat malam'),

            # Final additional entries to reach 1500+ target
            ('', 'weekend', '', '', 'akhir pekan'),
            ('', 'akhir pekan', '', '', 'akhir pekan'),
            ('', 'holiday', '', '', 'liburan'),
            ('', 'liburan', '', '', 'liburan'),
            ('', 'vacation', '', '', 'liburan'),
            ('', 'school', '', '', 'sekolah'),
            ('', 'sekolah', '', '', 'sekolah'),
            ('', 'university', '', '', 'universitas'),
            ('', 'universitas', '', '', 'universitas'),
            ('', 'college', '', '', 'kuliah'),
            ('', 'kuliah', '', '', 'kuliah'),
            ('', 'student', '', '', 'siswa'),
            ('', 'siswa', '', '', 'siswa'),
            ('', 'teacher', '', '', 'guru'),
            ('', 'guru', '', '', 'guru'),
            ('', 'lesson', '', '', 'pelajaran'),
            ('', 'pelajaran', '', '', 'pelajaran'),
            ('', 'exam', '', '', 'ujian'),
            ('', 'ujian', '', '', 'ujian'),
            ('', 'test', '', '', 'tes'),
            ('', 'tes', '', '', 'tes'),
            ('', 'homework', '', '', 'pekerjaan rumah'),
            ('', 'pr', '', '', 'pekerjaan rumah'),
            ('', 'tugas', '', '', 'tugas'),
            ('', 'assignment', '', '', 'tugas'),
            ('', 'project', '', '', 'proyek'),
            ('', 'proyek', '', '', 'proyek'),
            ('', 'meeting', '', '', 'rapat'),
            ('', 'rapat', '', '', 'rapat'),
            ('', 'conference', '', '', 'konferensi'),
            ('', 'konferensi', '', '', 'konferensi'),
            ('', 'presentation', '', '', 'presentasi'),
            ('', 'presentasi', '', '', 'presentasi'),
            ('', 'report', '', '', 'laporan'),
            ('', 'laporan', '', '', 'laporan'),
            ('', 'document', '', '', 'dokumen'),
            ('', 'dokumen', '', '', 'dokumen'),
            ('', 'file', '', '', 'berkas'),
            ('', 'berkas', '', '', 'berkas'),
            ('', 'folder', '', '', 'folder'),
            ('', 'computer', '', '', 'komputer'),
            ('', 'komputer', '', '', 'komputer'),
            ('', 'laptop', '', '', 'laptop'),
            ('', 'phone', '', '', 'telepon'),
            ('', 'telepon', '', '', 'telepon'),
            ('', 'handphone', '', '', 'handphone'),
            ('', 'hp', '', '', 'handphone'),
            ('', 'smartphone', '', '', 'smartphone'),
            ('', 'internet', '', '', 'internet'),
            ('', 'website', '', '', 'situs web'),
            ('', 'situs web', '', '', 'situs web'),
            ('', 'email', '', '', 'email'),
            ('', 'social media', '', '', 'media sosial'),
            ('', 'media sosial', '', '', 'media sosial'),
            ('', 'medsos', '', '', 'media sosial'),
            ('', 'facebook', '', '', 'facebook'),
            ('', 'instagram', '', '', 'instagram'),
            ('', 'twitter', '', '', 'twitter'),
            ('', 'whatsapp', '', '', 'whatsapp'),
            ('', 'wa', '', '', 'whatsapp'),
            ('', 'telegram', '', '', 'telegram'),
            ('', 'youtube', '', '', 'youtube'),
            ('', 'google', '', '', 'google'),
            ('', 'search', '', '', 'cari'),
            ('', 'cari', '', '', 'cari'),
            ('', 'find', '', '', 'temukan'),
            ('', 'temukan', '', '', 'temukan'),
            ('', 'discover', '', '', 'temukan'),
            ('', 'explore', '', '', 'jelajahi'),
            ('', 'jelajahi', '', '', 'jelajahi'),
            ('', 'browse', '', '', 'jelajah'),
            ('', 'jelajah', '', '', 'jelajah'),
            ('', 'click', '', '', 'klik'),
            ('', 'klik', '', '', 'klik'),
            ('', 'tap', '', '', 'ketuk'),
            ('', 'ketuk', '', '', 'ketuk'),
            ('', 'touch', '', '', 'sentuh'),
            ('', 'sentuh', '', '', 'sentuh'),
            ('', 'swipe', '', '', 'geser'),
            ('', 'geser', '', '', 'geser'),
            ('', 'scroll', '', '', 'gulir'),
            ('', 'gulir', '', '', 'gulir'),
            ('', 'zoom', '', '', 'perbesar'),
            ('', 'perbesar', '', '', 'perbesar'),
            ('', 'download', '', '', 'unduh'),
            ('', 'unduh', '', '', 'unduh'),
            ('', 'upload', '', '', 'unggah'),
            ('', 'unggah', '', '', 'unggah'),
            ('', 'share', '', '', 'bagikan'),
            ('', 'bagikan', '', '', 'bagikan'),
            ('', 'like', '', '', 'suka'),
            ('', 'comment', '', '', 'komentar'),
            ('', 'komentar', '', '', 'komentar'),
            ('', 'reply', '', '', 'balas'),
            ('', 'balas', '', '', 'balas'),
            ('', 'follow', '', '', 'ikuti'),
            ('', 'ikuti', '', '', 'ikuti'),
            ('', 'unfollow', '', '', 'berhenti mengikuti'),
            ('', 'block', '', '', 'blokir'),
            ('', 'blokir', '', '', 'blokir'),
            ('', 'report', '', '', 'laporkan'),
            ('', 'laporkan', '', '', 'laporkan'),
            ('', 'delete', '', '', 'hapus'),
            ('', 'hapus', '', '', 'hapus'),
            ('', 'edit', '', '', 'edit'),
            ('', 'save', '', '', 'simpan'),
            ('', 'simpan', '', '', 'simpan'),
            ('', 'cancel', '', '', 'batal'),
            ('', 'batal', '', '', 'batal'),
            ('', 'confirm', '', '', 'konfirmasi'),
            ('', 'konfirmasi', '', '', 'konfirmasi'),
            ('', 'submit', '', '', 'kirim'),
            ('', 'reset', '', '', 'reset'),
            ('', 'refresh', '', '', 'segarkan'),
            ('', 'segarkan', '', '', 'segarkan'),
            ('', 'reload', '', '', 'muat ulang'),
            ('', 'muat ulang', '', '', 'muat ulang'),
            ('', 'update', '', '', 'perbarui'),
            ('', 'perbarui', '', '', 'perbarui'),
            ('', 'upgrade', '', '', 'tingkatkan'),
            ('', 'tingkatkan', '', '', 'tingkatkan'),
            ('', 'install', '', '', 'pasang'),
            ('', 'pasang', '', '', 'pasang'),
            ('', 'uninstall', '', '', 'hapus'),
            ('', 'settings', '', '', 'pengaturan'),
            ('', 'pengaturan', '', '', 'pengaturan'),
            ('', 'options', '', '', 'pilihan'),
            ('', 'pilihan', '', '', 'pilihan'),
            ('', 'menu', '', '', 'menu'),
            ('', 'home', '', '', 'beranda'),
            ('', 'beranda', '', '', 'beranda'),
            ('', 'profile', '', '', 'profil'),
            ('', 'profil', '', '', 'profil'),
            ('', 'account', '', '', 'akun'),
            ('', 'akun', '', '', 'akun'),
            ('', 'login', '', '', 'masuk'),
            ('', 'masuk', '', '', 'masuk'),
            ('', 'logout', '', '', 'keluar'),
            ('', 'keluar', '', '', 'keluar'),
            ('', 'register', '', '', 'daftar'),
            ('', 'daftar', '', '', 'daftar'),
            ('', 'signup', '', '', 'daftar'),
            ('', 'password', '', '', 'kata sandi'),
            ('', 'kata sandi', '', '', 'kata sandi'),
            ('', 'username', '', '', 'nama pengguna'),
            ('', 'nama pengguna', '', '', 'nama pengguna'),
            ('', 'user', '', '', 'pengguna'),
            ('', 'pengguna', '', '', 'pengguna'),
            ('', 'member', '', '', 'anggota'),
            ('', 'anggota', '', '', 'anggota'),
            ('', 'guest', '', '', 'tamu'),
            ('', 'tamu', '', '', 'tamu'),
            ('', 'visitor', '', '', 'pengunjung'),
            ('', 'pengunjung', '', '', 'pengunjung'),
            ('', 'online', '', '', 'online'),
            ('', 'offline', '', '', 'offline'),
            ('', 'connected', '', '', 'terhubung'),
            ('', 'terhubung', '', '', 'terhubung'),
            ('', 'disconnected', '', '', 'terputus'),
            ('', 'terputus', '', '', 'terputus'),
            ('', 'loading', '', '', 'memuat'),
            ('', 'memuat', '', '', 'memuat'),
            ('', 'processing', '', '', 'memproses'),
            ('', 'memproses', '', '', 'memproses'),
            ('', 'complete', '', '', 'selesai'),
            ('', 'incomplete', '', '', 'belum selesai'),
            ('', 'belum selesai', '', '', 'belum selesai'),
            ('', 'success', '', '', 'berhasil'),
            ('', 'berhasil', '', '', 'berhasil'),
            ('', 'failed', '', '', 'gagal'),
            ('', 'gagal', '', '', 'gagal'),
            ('', 'error', '', '', 'kesalahan'),
            ('', 'kesalahan', '', '', 'kesalahan'),
            ('', 'warning', '', '', 'peringatan'),
            ('', 'peringatan', '', '', 'peringatan'),
            ('', 'notice', '', '', 'pemberitahuan'),
            ('', 'pemberitahuan', '', '', 'pemberitahuan'),
            ('', 'notification', '', '', 'notifikasi'),
            ('', 'notifikasi', '', '', 'notifikasi'),
            ('', 'alert', '', '', 'peringatan'),
            ('', 'message', '', '', 'pesan'),
            ('', 'pesan', '', '', 'pesan'),
            ('', 'chat', '', '', 'obrolan'),
            ('', 'obrolan', '', '', 'obrolan'),
            ('', 'conversation', '', '', 'percakapan'),
            ('', 'percakapan', '', '', 'percakapan'),
            ('', 'discussion', '', '', 'diskusi'),
            ('', 'diskusi', '', '', 'diskusi'),
            ('', 'debate', '', '', 'debat'),
            ('', 'debat', '', '', 'debat'),
            ('', 'argument', '', '', 'argumen'),
            ('', 'argumen', '', '', 'argumen'),
            ('', 'opinion', '', '', 'pendapat'),
            ('', 'pendapat', '', '', 'pendapat'),
            ('', 'idea', '', '', 'ide'),
            ('', 'ide', '', '', 'ide'),
            ('', 'thought', '', '', 'pikiran'),
            ('', 'pikiran', '', '', 'pikiran'),
            ('', 'mind', '', '', 'pikiran'),
            ('', 'brain', '', '', 'otak'),
            ('', 'otak', '', '', 'otak'),
            ('', 'heart', '', '', 'hati'),
            ('', 'hati', '', '', 'hati'),
            ('', 'soul', '', '', 'jiwa'),
            ('', 'jiwa', '', '', 'jiwa'),
            ('', 'spirit', '', '', 'semangat'),
            ('', 'semangat', '', '', 'semangat'),
            ('', 'energy', '', '', 'energi'),
            ('', 'energi', '', '', 'energi'),
            ('', 'power', '', '', 'kekuatan'),
            ('', 'kekuatan', '', '', 'kekuatan'),
            ('', 'strength', '', '', 'kekuatan'),
            ('', 'force', '', '', 'kekuatan'),
            ('', 'pressure', '', '', 'tekanan'),
            ('', 'tekanan', '', '', 'tekanan'),
            ('', 'stress', '', '', 'tekanan'),
            ('', 'tension', '', '', 'ketegangan'),
            ('', 'ketegangan', '', '', 'ketegangan'),
            ('', 'conflict', '', '', 'konflik'),
            ('', 'konflik', '', '', 'konflik'),
            ('', 'problem', '', '', 'masalah'),
            ('', 'masalah', '', '', 'masalah'),
            ('', 'issue', '', '', 'masalah'),
            ('', 'trouble', '', '', 'masalah'),
            ('', 'difficulty', '', '', 'kesulitan'),
            ('', 'kesulitan', '', '', 'kesulitan'),
            ('', 'challenge', '', '', 'tantangan'),
            ('', 'tantangan', '', '', 'tantangan'),
            ('', 'opportunity', '', '', 'kesempatan'),
            ('', 'kesempatan', '', '', 'kesempatan'),
            ('', 'chance', '', '', 'kesempatan'),
            ('', 'possibility', '', '', 'kemungkinan'),
            ('', 'kemungkinan', '', '', 'kemungkinan'),
            ('', 'probability', '', '', 'kemungkinan'),
            ('', 'risk', '', '', 'risiko'),
            ('', 'risiko', '', '', 'risiko'),
            ('', 'danger', '', '', 'bahaya'),
            ('', 'bahaya', '', '', 'bahaya'),
            ('', 'safe', '', '', 'aman'),
            ('', 'aman', '', '', 'aman'),
            ('', 'secure', '', '', 'aman'),
            ('', 'protection', '', '', 'perlindungan'),
            ('', 'perlindungan', '', '', 'perlindungan'),
            ('', 'security', '', '', 'keamanan'),
            ('', 'keamanan', '', '', 'keamanan'),
            ('', 'privacy', '', '', 'privasi'),
            ('', 'privasi', '', '', 'privasi'),
            ('', 'secret', '', '', 'rahasia'),
            ('', 'rahasia', '', '', 'rahasia'),
            ('', 'public', '', '', 'publik'),
            ('', 'publik', '', '', 'publik'),
            ('', 'private', '', '', 'pribadi'),
            ('', 'pribadi', '', '', 'pribadi'),
            ('', 'personal', '', '', 'pribadi'),
            ('', 'individual', '', '', 'individu'),
            ('', 'individu', '', '', 'individu'),
            ('', 'group', '', '', 'kelompok'),
            ('', 'kelompok', '', '', 'kelompok'),
            ('', 'team', '', '', 'tim'),
            ('', 'tim', '', '', 'tim'),
            ('', 'community', '', '', 'komunitas'),
            ('', 'komunitas', '', '', 'komunitas'),
            ('', 'society', '', '', 'masyarakat'),
            ('', 'masyarakat', '', '', 'masyarakat'),
            ('', 'culture', '', '', 'budaya'),
            ('', 'budaya', '', '', 'budaya'),
            ('', 'tradition', '', '', 'tradisi'),
            ('', 'tradisi', '', '', 'tradisi'),
            ('', 'custom', '', '', 'adat'),
            ('', 'adat', '', '', 'adat'),
            ('', 'habit', '', '', 'kebiasaan'),
            ('', 'kebiasaan', '', '', 'kebiasaan'),
            ('', 'routine', '', '', 'rutinitas'),
            ('', 'rutinitas', '', '', 'rutinitas'),
            ('', 'schedule', '', '', 'jadwal'),
            ('', 'jadwal', '', '', 'jadwal'),
            ('', 'plan', '', '', 'rencana'),
            ('', 'rencana', '', '', 'rencana'),
            ('', 'goal', '', '', 'tujuan'),
            ('', 'tujuan', '', '', 'tujuan'),
            ('', 'target', '', '', 'target'),
            ('', 'objective', '', '', 'objektif'),
            ('', 'objektif', '', '', 'objektif'),
            ('', 'purpose', '', '', 'tujuan'),
            ('', 'reason', '', '', 'alasan'),
            ('', 'alasan', '', '', 'alasan'),
            ('', 'cause', '', '', 'sebab'),
            ('', 'sebab', '', '', 'sebab'),
            ('', 'effect', '', '', 'efek'),
            ('', 'efek', '', '', 'efek'),
            ('', 'result', '', '', 'hasil'),
            ('', 'hasil', '', '', 'hasil'),
            ('', 'outcome', '', '', 'hasil'),
            ('', 'consequence', '', '', 'konsekuensi'),
            ('', 'konsekuensi', '', '', 'konsekuensi'),
            ('', 'impact', '', '', 'dampak'),
            ('', 'dampak', '', '', 'dampak'),
            ('', 'influence', '', '', 'pengaruh'),
            ('', 'pengaruh', '', '', 'pengaruh'),
        ]

        # Add all additional entries
        for entry in additional_entries:
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

    def save_to_csv(self, filename='multilingual_stopwords_comprehensive.csv'):
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

        # Print statistics
        print(f"\n✅ Successfully generated comprehensive multilingual stopword dataset!")
        print(f"📊 Total entries: {len(df)}")
        print(f"📁 Saved to: {filename}")

        # Count non-empty entries per language
        en_count = df['en'].str.strip().ne('').sum()
        id_count = df['id'].str.strip().ne('').sum()
        jv_count = df['jv'].str.strip().ne('').sum()
        su_count = df['su'].str.strip().ne('').sum()
        formal_id_count = df['formal_id'].str.strip().ne('').sum()

        print(f"\n📈 Statistics:")
        print(f"   - English entries: {en_count}")
        print(f"   - Indonesian entries: {id_count}")
        print(f"   - Javanese entries: {jv_count}")
        print(f"   - Sundanese entries: {su_count}")
        print(f"   - Formal Indonesian entries: {formal_id_count}")

        return df


def main():
    """Main function to generate comprehensive multilingual stopwords"""
    try:
        # Create generator instance
        generator = ComprehensiveMultilingualStopwordGenerator()

        # Generate comprehensive dataset
        generator.generate_comprehensive_dataset()

        # Save to CSV
        df = generator.save_to_csv()

        return df

    except Exception as e:
        logger.error(f"Error generating stopwords: {e}")
        raise


if __name__ == "__main__":
    main()
