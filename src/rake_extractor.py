"""
rake_extractor.py — RAKE Keyword Extraction Baseline

This script extracts keyword phrases from each document INDEPENDENTLY
using the RAKE (Rapid Automatic Keyword Extraction) algorithm. It serves
as a second baseline alongside Vanilla TF-IDF, to compare against the
Pairwise TF-IDF method from the UAFGK paper.

RAKE works differently from TF-IDF — it scores multi-word phrases based
on word co-occurrence patterns within a single document, without needing
a corpus. This means each document is processed on its own.

Pipeline:
  1. Load paired data from  data/processed/pairs.csv
  2. Run RAKE on each book synopsis  → top 20 keyword phrases
  3. Run RAKE on each author bio     → top 20 keyword phrases
  4. Combine & deduplicate both lists → final top 20 unique phrases
  5. Save to  results/keyword_outputs/rake_keywords.csv

Usage:
  python src/rake_extractor.py
"""

import os
import pandas as pd
from rake_nltk import Rake


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

    # Drop any rows with missing text
    df.dropna(subset=['book_synopsis', 'author_bio'], inplace=True)
    print(f"   Loaded {len(df)} book–author pairs.")

    return df


# ---------------------------------------------------------------------------
# 2.  EXTRACT TOP-N KEYWORD PHRASES USING RAKE
# ---------------------------------------------------------------------------

def rake_extract(text: str, rake_instance: Rake, top_n: int = 20) -> list:
    """
    Runs RAKE on a single document and returns the top_n keyword phrases
    with their scores.

    RAKE scores phrases by summing the degree/frequency ratio of each
    word in the phrase. Higher scores = more distinctive phrases.

    Args:
        text:           the document text to extract from
        rake_instance:  a pre-configured Rake object
        top_n:          how many keyword phrases to keep

    Returns:
        List of (phrase, score) tuples, sorted by score descending.
    """
    # Handle empty or non-string input
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []

    # Run RAKE on this text
    rake_instance.extract_keywords_from_text(text)

    # get_ranked_phrases_with_scores() returns [(score, phrase), ...]
    # already sorted by score descending
    ranked = rake_instance.get_ranked_phrases_with_scores()

    # Convert to (phrase, score) and take top_n
    return [(phrase, float(score)) for score, phrase in ranked[:top_n]]


# ---------------------------------------------------------------------------
# 3.  COMBINE & DEDUPLICATE KEYWORDS FROM BOTH SOURCES
# ---------------------------------------------------------------------------

def combine_keywords(synopsis_kw: list, bio_kw: list, top_n: int = 20) -> list:
    """
    Merges keyword phrase lists from two sources (synopsis and bio).
    If the same phrase appears in both, keeps the higher score.
    Returns the top_n unique phrases sorted by score.

    Args:
        synopsis_kw: list of (phrase, score) from the synopsis
        bio_kw:      list of (phrase, score) from the bio
        top_n:       final number of phrases to keep

    Returns:
        List of (phrase, score) tuples — deduplicated, sorted, trimmed.
    """
    # Use a dict to keep the highest score for each phrase
    merged = {}

    for phrase, score in synopsis_kw:
        # Normalize to lowercase for deduplication
        key = phrase.lower().strip()
        merged[key] = max(merged.get(key, 0.0), score)

    for phrase, score in bio_kw:
        key = phrase.lower().strip()
        merged[key] = max(merged.get(key, 0.0), score)

    # Sort by score descending, then take top_n
    sorted_keywords = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]


# ---------------------------------------------------------------------------
# 4.  MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    """
    Runs the full RAKE keyword extraction pipeline.
    """
    # --- Paths ---
    PAIRS_PATH  = os.path.join('data', 'processed', 'pairs.csv')
    OUTPUT_PATH = os.path.join('results', 'keyword_outputs', 'rake_keywords.csv')

    # Step 1: Load data
    df = load_pairs(PAIRS_PATH)

    # Create a single RAKE instance to reuse for all documents.
    # - min_length=1: allow single-word keywords
    # - max_length=4: phrases up to 4 words (keeps things readable)
    # - Uses English stopwords by default to split phrases
    rake = Rake(min_length=1, max_length=4)

    # Step 2: Extract keywords from each synopsis and each bio
    print("\n[2/5] Extracting RAKE keywords from book synopses...")
    synopsis_keywords = []
    for text in df['book_synopsis']:
        synopsis_keywords.append(rake_extract(text, rake, top_n=20))
    print(f"   Processed {len(synopsis_keywords)} synopses.")

    print("\n[3/5] Extracting RAKE keywords from author biographies...")
    bio_keywords = []
    for text in df['author_bio']:
        bio_keywords.append(rake_extract(text, rake, top_n=20))
    print(f"   Processed {len(bio_keywords)} biographies.")

    # Step 3: Combine and deduplicate for each book–author pair
    print("\n[4/5] Combining and deduplicating keyword lists...")
    final_keywords = []
    for i in range(len(df)):
        combined = combine_keywords(synopsis_keywords[i], bio_keywords[i], top_n=20)
        # Store just the phrases as a semicolon-separated string
        keyword_str = "; ".join([phrase for phrase, score in combined])
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
    print("  EXAMPLE OUTPUTS (top 10 keyword phrases shown)")
    print("=" * 60)

    num_examples = min(3, len(df_out))
    for i in range(num_examples):
        row = df_out.iloc[i]
        # Show only the first 10 phrases for readability
        kw_list = row['keywords'].split('; ')[:10]
        print(f"\n  Book: {row['book_title']}")
        print(f"  Author: {row['author_name']}")
        print(f"  Top 10 keywords:")
        for j, kw in enumerate(kw_list, 1):
            print(f"    {j:>2}. {kw}")

    print("\n" + "=" * 60)
    print(f"  Full results saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
