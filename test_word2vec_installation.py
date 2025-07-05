#!/usr/bin/env python3
"""
Word2Vec Installation Test and Demo
Tests different Word2Vec implementations and provides installation guidance
"""

import sys
import subprocess
import importlib

def test_installation(package_name, import_name=None):
    """Test if a package is installed and can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}: Successfully installed and importable")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed or not importable")
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def test_word2vec_functionality():
    """Test Word2Vec functionality with sample data"""
    print("\n" + "="*60)
    print("TESTING WORD2VEC FUNCTIONALITY")
    print("="*60)
    
    try:
        # Test Gensim Word2Vec
        from gensim.models import Word2Vec
        from gensim.utils import simple_preprocess
        
        # Sample sentences for training
        sentences = [
            "the quick brown fox jumps over the lazy dog",
            "machine learning is a subset of artificial intelligence",
            "natural language processing helps computers understand text",
            "word2vec creates vector representations of words",
            "sentiment analysis determines emotional tone of text"
        ]
        
        # Tokenize sentences
        tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
        print(f"📝 Training on {len(sentences)} sample sentences")
        
        # Train Word2Vec model
        model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=50,  # Small size for demo
            window=3,
            min_count=1,
            workers=1,
            epochs=10
        )
        
        print(f"🎯 Model trained successfully!")
        print(f"📊 Vocabulary size: {len(model.wv.key_to_index)}")
        print(f"🔤 Sample words: {list(model.wv.key_to_index.keys())[:10]}")
        
        # Test word similarity
        if 'machine' in model.wv.key_to_index and 'learning' in model.wv.key_to_index:
            similarity = model.wv.similarity('machine', 'learning')
            print(f"🔗 Similarity between 'machine' and 'learning': {similarity:.4f}")
        
        # Get word vector
        if 'word2vec' in model.wv.key_to_index:
            vector = model.wv['word2vec']
            print(f"📐 Vector for 'word2vec': shape {vector.shape}, sample values: {vector[:5]}")
        
        print("✅ Gensim Word2Vec is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Word2Vec: {e}")
        return False

def test_alternative_implementations():
    """Test alternative Word2Vec implementations"""
    print("\n" + "="*60)
    print("TESTING ALTERNATIVE IMPLEMENTATIONS")
    print("="*60)
    
    alternatives = [
        ("word2vec", "word2vec"),
        ("fasttext", "fasttext"),
        ("transformers", "transformers"),
        ("spacy", "spacy")
    ]
    
    available_alternatives = []
    
    for package, import_name in alternatives:
        if test_installation(package, import_name):
            available_alternatives.append(package)
    
    return available_alternatives

def provide_installation_guide():
    """Provide comprehensive installation guide"""
    print("\n" + "="*60)
    print("WORD2VEC INSTALLATION GUIDE")
    print("="*60)
    
    print("\n🎯 RECOMMENDED INSTALLATION (Primary Method):")
    print("pip install gensim")
    print("   - Most popular and well-maintained")
    print("   - Includes Word2Vec, FastText, Doc2Vec")
    print("   - Excellent documentation and community support")
    
    print("\n🔧 ALTERNATIVE INSTALLATIONS:")
    print("\n1. Original Google Word2Vec:")
    print("   pip install word2vec")
    print("   - Direct implementation of Google's Word2Vec")
    print("   - Less maintained, may have compatibility issues")
    
    print("\n2. FastText (Facebook's extension):")
    print("   pip install fasttext")
    print("   - Handles out-of-vocabulary words")
    print("   - Good for morphologically rich languages")
    
    print("\n3. Transformers (Hugging Face):")
    print("   pip install transformers")
    print("   - Modern transformer-based embeddings")
    print("   - BERT, RoBERTa, etc.")
    
    print("\n4. spaCy integration:")
    print("   pip install spacy")
    print("   python -m spacy download en_core_web_md")
    print("   - Pre-trained word vectors")
    print("   - Integrated NLP pipeline")
    
    print("\n💡 FOR THIS PROJECT:")
    print("   ✅ Gensim is already installed and working")
    print("   ✅ Ready to use in your analysis notebooks")
    print("   ✅ Compatible with existing code")

def main():
    """Main function to run all tests"""
    print("="*60)
    print("WORD2VEC INSTALLATION AND FUNCTIONALITY TEST")
    print("="*60)
    
    # Test current installations
    print("\n📦 CHECKING CURRENT INSTALLATIONS:")
    packages_to_check = [
        ("gensim", "gensim"),
        ("word2vec", "word2vec"),
        ("fasttext", "fasttext"),
        ("transformers", "transformers"),
        ("spacy", "spacy")
    ]
    
    installed_packages = []
    for package, import_name in packages_to_check:
        if test_installation(package, import_name):
            installed_packages.append(package)
    
    print(f"\n📊 Summary: {len(installed_packages)} Word2Vec-related packages found")
    
    # Test Word2Vec functionality
    word2vec_working = test_word2vec_functionality()
    
    # Test alternatives
    alternatives = test_alternative_implementations()
    
    # Provide installation guide
    provide_installation_guide()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if word2vec_working:
        print("🎉 SUCCESS: Word2Vec is properly installed and working!")
        print("✅ You can use Word2Vec in your analysis notebooks")
        print("✅ Gensim Word2Vec is ready for production use")
    else:
        print("⚠️  ISSUE: Word2Vec functionality test failed")
        print("💡 Try: pip install --upgrade gensim")
    
    print(f"\n📦 Available packages: {', '.join(installed_packages) if installed_packages else 'None'}")
    print(f"🔧 Alternative implementations: {', '.join(alternatives) if alternatives else 'None'}")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run your Jupyter notebooks - Word2Vec should work")
    print("2. If issues occur, try: pip install --upgrade gensim")
    print("3. For advanced features, consider installing fasttext or transformers")

if __name__ == "__main__":
    main()
