"""
vanilla_tfidf.py — Standard TF-IDF Keyword Extraction Baseline

This script extracts keywords from each document INDEPENDENTLY using
sklearn's TfidfVectorizer. It serves as a baseline to compare against
the Pairwise TF-IDF method (which finds words important in BOTH the
book synopsis AND its author's biography simultaneously).

Pipeline:
  1. Load paired data from  data/processed/pairs.csv
  2. Fit TF-IDF on all synopses → extract top 20 keywords per synopsis
  3. Fit TF-IDF on all bios    → extract top 20 keywords per bio
  4. Combine & deduplicate both lists → final top 20 unique keywords
  5. Save to  results/keyword_outputs/vanilla_tfidf_keywords.csv

Usage:
  python src/vanilla_tfidf.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# 1.  LOAD THE PAIRED DATASET
# ---------------------------------------------------------------------------

def load_pairs(filepath: str) -> pd.DataFrame:
    """
    Loads the paired book–author CSV file.

    Args:
        filepath: path to pairs.csv

    Returns:
        DataFrame with columns: book_title, author_name, book_synopsis, author_bio
    """
    print(f"[1/5] Loading paired data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find '{filepath}'.\n"
            "Run data_loader.py first to create the paired dataset."
        )

    df = pd.read_csv(filepath)

    # Make sure the expected columns exist
    required = ['book_title', 'author_name', 'book_synopsis', 'author_bio']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in {filepath}")

    # Drop any rows with missing text (can't extract keywords from nothing)
    df.dropna(subset=['book_synopsis', 'author_bio'], inplace=True)
    print(f"   Loaded {len(df)} book–author pairs.")

    return df


# ---------------------------------------------------------------------------
# 2.  EXTRACT TOP-N KEYWORDS FROM A TF-IDF MATRIX
# ---------------------------------------------------------------------------

def extract_top_keywords(tfidf_matrix, feature_names: np.ndarray, top_n: int = 20) -> list:
    """
    For each document (row) in the TF-IDF matrix, returns the top_n
    keywords sorted by their TF-IDF score (highest first).

    Args:
        tfidf_matrix:  sparse or dense matrix from TfidfVectorizer
        feature_names: array of vocabulary words matching the matrix columns
        top_n:         how many keywords to keep per document

    Returns:
        A list of lists — one keyword list per document.
        Each inner list contains (word, score) tuples sorted descending.
    """
    all_keywords = []

    for row_idx in range(tfidf_matrix.shape[0]):
        # Get the TF-IDF scores for this document
        row = tfidf_matrix[row_idx].toarray().flatten()

        # Find the indices of the top_n highest scores
        top_indices = np.argsort(row)[::-1][:top_n]

        # Build the keyword list: (word, score)
        keywords = [(feature_names[i], float(row[i]))
                     for i in top_indices if row[i] > 0]

        all_keywords.append(keywords)

    return all_keywords


# ---------------------------------------------------------------------------
# 3.  COMBINE & DEDUPLICATE KEYWORDS FROM BOTH SOURCES
# ---------------------------------------------------------------------------

def combine_keywords(synopsis_kw: list, bio_kw: list, top_n: int = 20) -> list:
    """
    Merges keyword lists from two sources (synopsis and bio).
    If a word appears in both lists, keeps the higher score.
    Returns the top_n unique keywords sorted by score.

    Args:
        synopsis_kw: list of (word, score) from the synopsis TF-IDF
        bio_kw:      list of (word, score) from the bio TF-IDF
        top_n:       final number of keywords to keep

    Returns:
        List of (word, score) tuples — deduplicated, sorted, trimmed.
    """
    # Use a dict to keep the highest score for each word
    merged = {}

    for word, score in synopsis_kw:
        merged[word] = max(merged.get(word, 0.0), score)

    for word, score in bio_kw:
        merged[word] = max(merged.get(word, 0.0), score)

    # Sort by score descending, then take top_n
    sorted_keywords = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]


# ---------------------------------------------------------------------------
# 4.  MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    """
    Runs the full Vanilla TF-IDF keyword extraction pipeline.
    """
    # --- Paths ---
    PAIRS_PATH  = os.path.join('data', 'processed', 'pairs.csv')
    OUTPUT_PATH = os.path.join('results', 'keyword_outputs', 'vanilla_tfidf_keywords.csv')

    # Step 1: Load data
    df = load_pairs(PAIRS_PATH)

    # Step 2a: Fit TF-IDF on all book synopses
    print("\n[2/5] Fitting TF-IDF on book synopses...")
    synopsis_vectorizer = TfidfVectorizer(
        stop_words='english',   # remove common English stopwords
        max_features=10000,     # limit vocabulary size
        max_df=0.95,            # ignore words in >95% of documents
        min_df=2                # ignore words in fewer than 2 documents
    )
    synopsis_matrix = synopsis_vectorizer.fit_transform(df['book_synopsis'])
    synopsis_features = np.array(synopsis_vectorizer.get_feature_names_out())
    synopsis_keywords = extract_top_keywords(synopsis_matrix, synopsis_features, top_n=20)
    print(f"   Extracted top 20 keywords from {len(synopsis_keywords)} synopses.")

    # Step 2b: Fit TF-IDF on all author biographies
    print("\n[3/5] Fitting TF-IDF on author biographies...")
    bio_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        max_df=0.95,
        min_df=2
    )
    bio_matrix = bio_vectorizer.fit_transform(df['author_bio'])
    bio_features = np.array(bio_vectorizer.get_feature_names_out())
    bio_keywords = extract_top_keywords(bio_matrix, bio_features, top_n=20)
    print(f"   Extracted top 20 keywords from {len(bio_keywords)} biographies.")

    # Step 3: Combine and deduplicate for each book–author pair
    print("\n[4/5] Combining and deduplicating keyword lists...")
    final_keywords = []
    for i in range(len(df)):
        combined = combine_keywords(synopsis_keywords[i], bio_keywords[i], top_n=20)
        # Store just the words as a semicolon-separated string
        keyword_str = "; ".join([word for word, score in combined])
        final_keywords.append(keyword_str)

    df_out = pd.DataFrame({
        'book_title':  df['book_title'].values,
        'author_name': df['author_name'].values,
        'keywords':    final_keywords
    })

    # Step 4: Save results
    print(f"\n[5/5] Saving results to: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"   ✓ Saved {len(df_out)} rows.")

    # Step 5: Print 3 example outputs for visual inspection
    print("\n" + "=" * 60)
    print("  EXAMPLE OUTPUTS (top 10 keywords shown)")
    print("=" * 60)

    num_examples = min(3, len(df_out))
    for i in range(num_examples):
        row = df_out.iloc[i]
        # Show only the first 10 keywords for readability
        kw_list = row['keywords'].split('; ')[:10]
        print(f"\n  Book: {row['book_title']}")
        print(f"  Author: {row['author_name']}")
        print(f"  Top 10 keywords: {', '.join(kw_list)}")

    print("\n" + "=" * 60)
    print(f"  Full results saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
