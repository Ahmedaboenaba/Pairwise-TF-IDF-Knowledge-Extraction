"""
pairwise_tfidf.py — Pairwise TF-IDF Keyword Extraction (UAFGK Method)

This script implements the core contribution of the UAFGK paper
(Applied Soft Computing, 2026). Unlike standard TF-IDF which scores
words within a single document, Pairwise TF-IDF finds words that are
important in BOTH a book synopsis AND its paired author biography.

============================================================
  THE FORMULA (from the paper):
  
    Pairwise-TF-IDF(w) = TF(w,d) × IDF(w,D) × TF(w,d') × IDF(w,D')
  
  Where:
    TF(w,d)    = occurrences of word w in document d / total words in d
    IDF(w,D)   = log( total documents in corpus D / documents in D containing w )
    TF(w,d')   = occurrences of w in paired document d' / total words in d'
    IDF(w,D')  = log( total documents in corpus D' / documents in D' containing w )
  
  d  = a book synopsis         D  = corpus of ALL book synopses
  d' = paired author biography  D' = corpus of ALL author biographies
============================================================

The key insight: if a word scores high in the synopsis but NOT in the
bio (or vice versa), the product goes to zero. Only words that are
meaningfully present in BOTH documents get a high pairwise score.

Pipeline:
  1. Load paired data from  data/processed/pairs.csv
  2. Preprocess text (lowercase, remove punctuation & stopwords)
  3. Build two corpora: D (synopses) and D' (bios)
  4. Compute Pairwise TF-IDF for each book–author pair
  5. Save top 20 keywords per pair to results/keyword_outputs/

Usage:
  python src/pairwise_tfidf.py
"""

import os
import re
import math
import pandas as pd
import nltk
from collections import Counter

# Download NLTK stopwords if not already present
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# Load English stopwords once (used throughout the script)
STOP_WORDS = set(stopwords.words('english'))


# ===================================================================
# 1.  LOAD THE PAIRED DATASET
# ===================================================================

def load_pairs(filepath: str) -> pd.DataFrame:
    """
    Loads the paired book–author CSV file.

    Args:
        filepath: path to pairs.csv

    Returns:
        DataFrame with columns: book_title, author_name, book_synopsis, author_bio
    """
    print(f"[1/6] Loading paired data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find '{filepath}'.\n"
            "Run data_loader.py first to create the paired dataset."
        )

    df = pd.read_csv(filepath)

    required = ['book_title', 'author_name', 'book_synopsis', 'author_bio']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in {filepath}")

    df.dropna(subset=['book_synopsis', 'author_bio'], inplace=True)
    print(f"   Loaded {len(df)} book–author pairs.")

    return df


# ===================================================================
# 2.  TEXT PREPROCESSING
# ===================================================================

def preprocess(text: str) -> list:
    """
    Cleans and tokenizes a raw text string.

    Steps:
      1. Convert to lowercase
      2. Remove all punctuation and special characters
      3. Split into individual words (tokenize by whitespace)
      4. Remove English stopwords
      5. Remove very short words (length < 2)

    Args:
        text: raw document string

    Returns:
        List of cleaned tokens (words)
    """
    if not isinstance(text, str):
        return []

    # Step 2a: lowercase
    text = text.lower()

    # Step 2b: remove punctuation — keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 2c: tokenize by whitespace
    tokens = text.split()

    # Step 2d: remove stopwords and very short words
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) >= 2]

    return tokens


# ===================================================================
# 3.  TF AND IDF COMPUTATION (from scratch)
# ===================================================================

def compute_tf(tokens: list) -> dict:
    """
    Computes Term Frequency for each word in a single document.

    Formula:  TF(w, d) = count(w in d) / total_words(d)

    This tells us how often a word appears relative to the document
    length. Common words in the document get higher TF scores.

    Args:
        tokens: list of preprocessed words from one document

    Returns:
        Dict mapping each word to its TF value
    """
    total_words = len(tokens)

    if total_words == 0:
        return {}

    # Count occurrences of each word
    word_counts = Counter(tokens)

    # Divide each count by total number of words
    tf = {word: count / total_words for word, count in word_counts.items()}

    return tf


def compute_idf(corpus: list) -> dict:
    """
    Computes Inverse Document Frequency for all words across a corpus.

    Formula:  IDF(w, D) = log( N / df(w) )

    Where:
      N     = total number of documents in the corpus
      df(w) = number of documents that contain word w

    Words that appear in many documents get LOW IDF (they're common/generic).
    Words that appear in few documents get HIGH IDF (they're distinctive).

    Args:
        corpus: list of documents, where each document is a list of tokens

    Returns:
        Dict mapping each word to its IDF value
    """
    N = len(corpus)  # total number of documents

    if N == 0:
        return {}

    # Step 1: count how many documents contain each word (document frequency)
    doc_freq = Counter()
    for doc_tokens in corpus:
        # Use a set so each word is counted once per document
        unique_words = set(doc_tokens)
        for word in unique_words:
            doc_freq[word] += 1

    # Step 2: compute IDF for each word
    # We add a small check: if df = 0 somehow, skip that word
    idf = {}
    for word, df in doc_freq.items():
        idf[word] = math.log(N / df)

    return idf


# ===================================================================
# 4.  PAIRWISE TF-IDF COMPUTATION
# ===================================================================

def pairwise_tfidf(
    synopsis_tokens: list,
    bio_tokens: list,
    idf_synopses: dict,
    idf_bios: dict,
    top_n: int = 20
) -> list:
    """
    Computes the Pairwise TF-IDF score for each word shared between
    a book synopsis and its paired author biography.

    ================================================================
    FORMULA:  Pairwise-TF-IDF(w) = TF(w,d) × IDF(w,D) × TF(w,d') × IDF(w,D')
    ================================================================

    The multiplication ensures that ONLY words which are important
    in BOTH documents get a high score. If a word is missing from
    either document, its TF in that document is 0, making the entire
    product 0.

    Args:
        synopsis_tokens: preprocessed tokens from the book synopsis (d)
        bio_tokens:      preprocessed tokens from the author bio (d')
        idf_synopses:    IDF scores computed over ALL synopses (D)
        idf_bios:        IDF scores computed over ALL bios (D')
        top_n:           how many top keywords to return

    Returns:
        List of (word, pairwise_score) tuples, sorted descending
    """
    # ----- Step A: Compute TF for the synopsis (d) -----
    tf_synopsis = compute_tf(synopsis_tokens)

    # ----- Step B: Compute TF for the bio (d') -----
    tf_bio = compute_tf(bio_tokens)

    # ----- Step C: Find words that appear in BOTH documents -----
    # Only these words can have a non-zero pairwise score,
    # because if a word is missing from one document, its TF = 0
    # and the product becomes 0.
    shared_words = set(tf_synopsis.keys()) & set(tf_bio.keys())

    # ----- Step D: Compute Pairwise TF-IDF for each shared word -----
    pairwise_scores = {}

    for word in shared_words:
        # TF(w, d)  — how frequent is this word in the synopsis?
        tf_d = tf_synopsis[word]

        # IDF(w, D) — how distinctive is this word across ALL synopses?
        idf_d = idf_synopses.get(word, 0.0)

        # TF(w, d') — how frequent is this word in the paired bio?
        tf_d_prime = tf_bio[word]

        # IDF(w, D') — how distinctive is this word across ALL bios?
        idf_d_prime = idf_bios.get(word, 0.0)

        # ============================================
        # THE CORE FORMULA FROM THE PAPER:
        #   Pairwise-TF-IDF(w) = TF × IDF × TF' × IDF'
        # ============================================
        score = tf_d * idf_d * tf_d_prime * idf_d_prime

        if score > 0:
            pairwise_scores[word] = score

    # ----- Step E: Sort by score and return top_n -----
    sorted_keywords = sorted(
        pairwise_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_keywords[:top_n]


# ===================================================================
# 5.  MAIN PIPELINE
# ===================================================================

def main():
    """
    Runs the full Pairwise TF-IDF keyword extraction pipeline.
    """
    # --- Paths ---
    PAIRS_PATH  = os.path.join('data', 'processed', 'pairs.csv')
    OUTPUT_PATH = os.path.join('results', 'keyword_outputs', 'pairwise_tfidf_keywords.csv')

    # ---------------------------------------------------------------
    # Step 1: Load data
    # ---------------------------------------------------------------
    df = load_pairs(PAIRS_PATH)

    # ---------------------------------------------------------------
    # Step 2: Preprocess all texts
    # ---------------------------------------------------------------
    print("\n[2/6] Preprocessing all texts (lowercase, remove punctuation & stopwords)...")

    # Tokenize every synopsis and every bio
    synopsis_tokens_list = [preprocess(text) for text in df['book_synopsis']]
    bio_tokens_list      = [preprocess(text) for text in df['author_bio']]

    print(f"   Preprocessed {len(synopsis_tokens_list)} synopses and {len(bio_tokens_list)} bios.")

    # ---------------------------------------------------------------
    # Step 3: Build the two corpora and compute IDF for each
    # ---------------------------------------------------------------
    #   D  = corpus of all book synopses (one document per book)
    #   D' = corpus of all author biographies (one document per author)
    # ---------------------------------------------------------------
    print("\n[3/6] Computing IDF across synopsis corpus (D)...")
    idf_synopses = compute_idf(synopsis_tokens_list)
    print(f"   Vocabulary size in D: {len(idf_synopses)} unique words.")

    print("\n[4/6] Computing IDF across biography corpus (D')...")
    idf_bios = compute_idf(bio_tokens_list)
    print(f"   Vocabulary size in D': {len(idf_bios)} unique words.")

    # ---------------------------------------------------------------
    # Step 4: Compute Pairwise TF-IDF for each book–author pair
    # ---------------------------------------------------------------
    print("\n[5/6] Computing Pairwise TF-IDF for each book–author pair...")

    all_keywords = []   # list of semicolon-separated keyword strings
    all_scores = []     # list of semicolon-separated score strings

    for i in range(len(df)):
        # Get the top keywords for this pair
        top_kw = pairwise_tfidf(
            synopsis_tokens=synopsis_tokens_list[i],
            bio_tokens=bio_tokens_list[i],
            idf_synopses=idf_synopses,
            idf_bios=idf_bios,
            top_n=20
        )

        # Format as semicolon-separated strings
        keywords_str = "; ".join([word for word, score in top_kw])
        scores_str   = "; ".join([f"{score:.6f}" for word, score in top_kw])

        all_keywords.append(keywords_str)
        all_scores.append(scores_str)

    print(f"   Processed {len(df)} pairs.")

    # ---------------------------------------------------------------
    # Step 5: Save results
    # ---------------------------------------------------------------
    df_out = pd.DataFrame({
        'book_title':  df['book_title'].values,
        'author_name': df['author_name'].values,
        'keywords':    all_keywords,
        'scores':      all_scores
    })

    print(f"\n[6/6] Saving results to: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"   ✓ Saved {len(df_out)} rows.")

    # ---------------------------------------------------------------
    # Step 6: Print 3 example outputs for visual inspection
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EXAMPLE OUTPUTS — Pairwise TF-IDF (top 10 keywords + scores)")
    print("=" * 70)

    num_examples = min(3, len(df_out))
    for i in range(num_examples):
        row = df_out.iloc[i]

        kw_list    = row['keywords'].split('; ')[:10]
        score_list = row['scores'].split('; ')[:10]

        print(f"\n  Book:   {row['book_title']}")
        print(f"  Author: {row['author_name']}")
        print(f"  {'Rank':<6} {'Keyword':<25} {'Pairwise Score':<15}")
        print(f"  {'----':<6} {'-------':<25} {'--------------':<15}")

        for j, (kw, sc) in enumerate(zip(kw_list, score_list), 1):
            print(f"  {j:<6} {kw:<25} {sc:<15}")

    print("\n" + "=" * 70)
    print(f"  Full results saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
