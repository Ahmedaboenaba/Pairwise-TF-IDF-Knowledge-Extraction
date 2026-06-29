"""
evaluate.py — Evaluation & Comparison of Keyword Extraction Methods

This script compares three keyword extraction methods side by side:
  1. Vanilla TF-IDF  (standard single-document baseline)
  2. RAKE            (phrase-based baseline)
  3. Pairwise TF-IDF (our UAFGK paper implementation)

It answers the question: does Pairwise TF-IDF actually find keywords
that are relevant to BOTH documents in a pair, better than the baselines?

Metrics:
  A. Lexical Overlap Score  — what % of keywords appear in BOTH the
     synopsis AND the bio? (Higher = better at finding shared terms)
  B. Keyword Diversity      — how many unique keywords does each method
     produce across all books? (Higher = less repetitive/generic)

It also produces a side-by-side qualitative comparison table for 5
randomly selected books so you can visually inspect the differences.

Usage:
  python src/evaluate.py
"""

import os
import re
import random
import pandas as pd
import nltk

# Download NLTK stopwords if needed
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


# ===================================================================
# HELPER: Parse semicolon-separated keyword strings into lists
# ===================================================================

def parse_keywords(keyword_str: str) -> list:
    """
    Splits a semicolon-separated keyword string into a clean list.

    Args:
        keyword_str: e.g. "neural networks; deep learning; vision"

    Returns:
        List of lowercase, stripped keyword strings.
    """
    if not isinstance(keyword_str, str) or len(keyword_str.strip()) == 0:
        return []

    return [kw.strip().lower() for kw in keyword_str.split(';') if kw.strip()]


# ===================================================================
# HELPER: Normalize text for matching
# ===================================================================

def normalize_text(text: str) -> str:
    """
    Lowercases text and removes punctuation, so we can check
    whether a keyword appears inside a document.

    Args:
        text: raw document string

    Returns:
        Cleaned lowercase string with only letters and spaces.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# ===================================================================
# 1.  LOAD ALL DATA
# ===================================================================

def load_all_data(pairs_path, vanilla_path, rake_path, pairwise_path):
    """
    Loads the original paired data and all three keyword result files.

    Returns:
        Tuple of (pairs_df, vanilla_df, rake_df, pairwise_df)
    """
    print("[1/4] Loading all data files...")

    for path, name in [(pairs_path, "pairs.csv"),
                       (vanilla_path, "vanilla_tfidf_keywords.csv"),
                       (rake_path, "rake_keywords.csv"),
                       (pairwise_path, "pairwise_tfidf_keywords.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find '{path}'.\n"
                f"Make sure you have run all extraction scripts first."
            )

    pairs_df    = pd.read_csv(pairs_path)
    vanilla_df  = pd.read_csv(vanilla_path)
    rake_df     = pd.read_csv(rake_path)
    pairwise_df = pd.read_csv(pairwise_path)

    print(f"   Pairs:          {len(pairs_df)} rows")
    print(f"   Vanilla TF-IDF: {len(vanilla_df)} rows")
    print(f"   RAKE:           {len(rake_df)} rows")
    print(f"   Pairwise TF-IDF:{len(pairwise_df)} rows")

    return pairs_df, vanilla_df, rake_df, pairwise_df


# ===================================================================
# 2.  METRIC A: LEXICAL OVERLAP SCORE
# ===================================================================

def compute_lexical_overlap(keywords_df: pd.DataFrame, pairs_df: pd.DataFrame, method_name: str) -> float:
    """
    For each book, measures what percentage of extracted keywords
    actually appear in BOTH the synopsis AND the bio.

    This is the key metric — Pairwise TF-IDF should score highest here
    because it is specifically designed to find shared terms.

    How it works:
      For each keyword, check if it appears as a substring in the
      normalized synopsis AND the normalized bio. Count the fraction
      that appear in both. Average across all books.

    Args:
        keywords_df: DataFrame with 'book_title' and 'keywords' columns
        pairs_df:    DataFrame with 'book_title', 'book_synopsis', 'author_bio'
        method_name: name string for display

    Returns:
        Average overlap score (0.0 to 1.0)
    """
    # Merge keywords with the original texts on book_title
    merged = keywords_df.merge(
        pairs_df[['book_title', 'book_synopsis', 'author_bio']],
        on='book_title',
        how='inner'
    )

    overlap_scores = []

    for _, row in merged.iterrows():
        keywords = parse_keywords(row['keywords'])

        if len(keywords) == 0:
            continue

        # Normalize both documents for substring matching
        synopsis_norm = normalize_text(row['book_synopsis'])
        bio_norm      = normalize_text(row['author_bio'])

        # Count how many keywords appear in BOTH documents
        in_both = 0
        for kw in keywords:
            # For multi-word phrases (RAKE), check if the whole phrase appears
            # For single words, this still works as substring matching
            kw_clean = re.sub(r'[^a-z\s]', '', kw).strip()
            if kw_clean and kw_clean in synopsis_norm and kw_clean in bio_norm:
                in_both += 1

        # Overlap = fraction of keywords that appear in both documents
        overlap = in_both / len(keywords)
        overlap_scores.append(overlap)

    # Average across all books
    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    return avg_overlap


# ===================================================================
# 3.  METRIC B: KEYWORD DIVERSITY SCORE
# ===================================================================

def compute_keyword_diversity(keywords_df: pd.DataFrame, method_name: str) -> dict:
    """
    Measures how diverse/varied the keywords are across all books.

    Two sub-metrics:
      a) Avg unique keywords per book — are keywords varied within each book?
      b) Total unique keywords across ALL books — is the method producing
         a rich vocabulary, or repeating the same generic words everywhere?

    A method that always outputs "the", "book", "author" for every book
    would score low on diversity.

    Args:
        keywords_df: DataFrame with 'keywords' column
        method_name: name string for display

    Returns:
        Dict with 'avg_unique_per_book' and 'total_unique_across_all'
    """
    all_unique = set()         # unique keywords across entire dataset
    per_book_counts = []       # unique count per individual book

    for kw_str in keywords_df['keywords']:
        keywords = parse_keywords(kw_str)
        unique_in_book = set(keywords)
        per_book_counts.append(len(unique_in_book))
        all_unique.update(unique_in_book)

    avg_per_book = sum(per_book_counts) / len(per_book_counts) if per_book_counts else 0.0

    return {
        'avg_unique_per_book':     round(avg_per_book, 2),
        'total_unique_across_all': len(all_unique)
    }


# ===================================================================
# 4.  QUALITATIVE: SIDE-BY-SIDE COMPARISON TABLE
# ===================================================================

def build_comparison_table(
    pairs_df: pd.DataFrame,
    vanilla_df: pd.DataFrame,
    rake_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    num_examples: int = 5,
    top_n: int = 10
) -> pd.DataFrame:
    """
    For a random sample of books, shows the top keywords from all
    three methods in one table so you can visually compare them.

    Args:
        pairs_df:     original paired data
        vanilla_df:   Vanilla TF-IDF keywords
        rake_df:      RAKE keywords
        pairwise_df:  Pairwise TF-IDF keywords
        num_examples: how many books to sample
        top_n:        how many keywords to show per method

    Returns:
        DataFrame with columns: book_title, author_name,
        vanilla_keywords, rake_keywords, pairwise_keywords
    """
    # Find books that exist in ALL three result files
    common_titles = (
        set(vanilla_df['book_title']) &
        set(rake_df['book_title']) &
        set(pairwise_df['book_title'])
    )

    if len(common_titles) == 0:
        print("   [Warning] No books found in common across all 3 methods.")
        return pd.DataFrame()

    # Sample up to num_examples books randomly
    sample_titles = random.sample(
        sorted(common_titles),
        min(num_examples, len(common_titles))
    )

    rows = []
    for title in sample_titles:
        # Look up the author name from the pairs data
        pair_row = pairs_df[pairs_df['book_title'] == title]
        author = pair_row['author_name'].values[0] if len(pair_row) > 0 else "Unknown"

        # Get keywords from each method (take only top_n)
        v_row = vanilla_df[vanilla_df['book_title'] == title]
        r_row = rake_df[rake_df['book_title'] == title]
        p_row = pairwise_df[pairwise_df['book_title'] == title]

        v_kw = parse_keywords(v_row['keywords'].values[0])[:top_n] if len(v_row) > 0 else []
        r_kw = parse_keywords(r_row['keywords'].values[0])[:top_n] if len(r_row) > 0 else []
        p_kw = parse_keywords(p_row['keywords'].values[0])[:top_n] if len(p_row) > 0 else []

        rows.append({
            'book_title':         title,
            'author_name':        author,
            'vanilla_keywords':   "; ".join(v_kw),
            'rake_keywords':      "; ".join(r_kw),
            'pairwise_keywords':  "; ".join(p_kw),
        })

    return pd.DataFrame(rows)


# ===================================================================
# 5.  MAIN PIPELINE
# ===================================================================

def main():
    """
    Runs the full evaluation and comparison pipeline.
    """
    # --- Paths ---
    PAIRS_PATH    = os.path.join('data', 'processed', 'pairs.csv')
    VANILLA_PATH  = os.path.join('results', 'keyword_outputs', 'vanilla_tfidf_keywords.csv')
    RAKE_PATH     = os.path.join('results', 'keyword_outputs', 'rake_keywords.csv')
    PAIRWISE_PATH = os.path.join('results', 'keyword_outputs', 'pairwise_tfidf_keywords.csv')
    COMPARISON_OUTPUT = os.path.join('results', 'comparison_table.csv')

    # ---------------------------------------------------------------
    # Step 1: Load all data
    # ---------------------------------------------------------------
    pairs_df, vanilla_df, rake_df, pairwise_df = load_all_data(
        PAIRS_PATH, VANILLA_PATH, RAKE_PATH, PAIRWISE_PATH
    )

    # ---------------------------------------------------------------
    # Step 2: Compute Lexical Overlap Score for each method
    # ---------------------------------------------------------------
    print("\n[2/4] Computing Lexical Overlap Scores...")
    print("   (What % of keywords appear in BOTH the synopsis AND the bio?)\n")

    overlap_vanilla  = compute_lexical_overlap(vanilla_df, pairs_df, "Vanilla TF-IDF")
    overlap_rake     = compute_lexical_overlap(rake_df, pairs_df, "RAKE")
    overlap_pairwise = compute_lexical_overlap(pairwise_df, pairs_df, "Pairwise TF-IDF")

    print(f"   Vanilla TF-IDF  : {overlap_vanilla:.4f}  ({overlap_vanilla*100:.1f}%)")
    print(f"   RAKE            : {overlap_rake:.4f}  ({overlap_rake*100:.1f}%)")
    print(f"   Pairwise TF-IDF : {overlap_pairwise:.4f}  ({overlap_pairwise*100:.1f}%)")

    # ---------------------------------------------------------------
    # Step 3: Compute Keyword Diversity for each method
    # ---------------------------------------------------------------
    print("\n[3/4] Computing Keyword Diversity Scores...")
    print("   (Are keywords varied, or does the method repeat generic words?)\n")

    div_vanilla  = compute_keyword_diversity(vanilla_df, "Vanilla TF-IDF")
    div_rake     = compute_keyword_diversity(rake_df, "RAKE")
    div_pairwise = compute_keyword_diversity(pairwise_df, "Pairwise TF-IDF")

    print(f"   Vanilla TF-IDF  : {div_vanilla['avg_unique_per_book']:.1f} avg/book, "
          f"{div_vanilla['total_unique_across_all']} total unique")
    print(f"   RAKE            : {div_rake['avg_unique_per_book']:.1f} avg/book, "
          f"{div_rake['total_unique_across_all']} total unique")
    print(f"   Pairwise TF-IDF : {div_pairwise['avg_unique_per_book']:.1f} avg/book, "
          f"{div_pairwise['total_unique_across_all']} total unique")

    # ---------------------------------------------------------------
    # Step 4: Build side-by-side comparison table
    # ---------------------------------------------------------------
    print("\n[4/4] Building side-by-side comparison for 5 random books...")

    comparison_df = build_comparison_table(
        pairs_df, vanilla_df, rake_df, pairwise_df,
        num_examples=5, top_n=10
    )

    # Save the comparison table
    os.makedirs(os.path.dirname(COMPARISON_OUTPUT), exist_ok=True)
    comparison_df.to_csv(COMPARISON_OUTPUT, index=False)
    print(f"   ✓ Comparison table saved to: {COMPARISON_OUTPUT}")

    # Print the side-by-side comparison to console
    print("\n" + "=" * 90)
    print("  SIDE-BY-SIDE KEYWORD COMPARISON (5 random books, top 10 keywords)")
    print("=" * 90)

    for _, row in comparison_df.iterrows():
        v_list = row['vanilla_keywords'].split('; ') if row['vanilla_keywords'] else []
        r_list = row['rake_keywords'].split('; ') if row['rake_keywords'] else []
        p_list = row['pairwise_keywords'].split('; ') if row['pairwise_keywords'] else []

        # Pad lists to same length for aligned printing
        max_len = max(len(v_list), len(r_list), len(p_list), 1)
        v_list += [''] * (max_len - len(v_list))
        r_list += [''] * (max_len - len(r_list))
        p_list += [''] * (max_len - len(p_list))

        print(f"\n  Book: {row['book_title']}")
        print(f"  Author: {row['author_name']}")
        print(f"  {'#':<4} {'Vanilla TF-IDF':<25} {'RAKE':<25} {'Pairwise TF-IDF':<25}")
        print(f"  {'--':<4} {'-'*23:<25} {'-'*23:<25} {'-'*23:<25}")

        for j in range(min(max_len, 10)):
            print(f"  {j+1:<4} {v_list[j]:<25} {r_list[j]:<25} {p_list[j]:<25}")

    # ---------------------------------------------------------------
    # FINAL SUMMARY TABLE
    # ---------------------------------------------------------------
    print("\n\n" + "=" * 75)
    print("  FINAL SUMMARY — METHOD COMPARISON")
    print("=" * 75)
    print(f"  {'Method':<20} {'Lexical Overlap':<20} {'Avg Unique/Book':<18} {'Total Unique':<15}")
    print(f"  {'-'*18:<20} {'-'*18:<20} {'-'*16:<18} {'-'*13:<15}")

    print(f"  {'Vanilla TF-IDF':<20} "
          f"{overlap_vanilla*100:>14.1f}%     "
          f"{div_vanilla['avg_unique_per_book']:>12.1f}      "
          f"{div_vanilla['total_unique_across_all']:>10}")

    print(f"  {'RAKE':<20} "
          f"{overlap_rake*100:>14.1f}%     "
          f"{div_rake['avg_unique_per_book']:>12.1f}      "
          f"{div_rake['total_unique_across_all']:>10}")

    print(f"  {'Pairwise TF-IDF':<20} "
          f"{overlap_pairwise*100:>14.1f}%     "
          f"{div_pairwise['avg_unique_per_book']:>12.1f}      "
          f"{div_pairwise['total_unique_across_all']:>10}")

    print("=" * 75)

    # Determine the winner for each metric
    overlap_scores = {
        'Vanilla TF-IDF': overlap_vanilla,
        'RAKE': overlap_rake,
        'Pairwise TF-IDF': overlap_pairwise
    }
    diversity_scores = {
        'Vanilla TF-IDF': div_vanilla['total_unique_across_all'],
        'RAKE': div_rake['total_unique_across_all'],
        'Pairwise TF-IDF': div_pairwise['total_unique_across_all']
    }

    best_overlap   = max(overlap_scores, key=overlap_scores.get)
    best_diversity = max(diversity_scores, key=diversity_scores.get)

    print(f"\n  🏆 Best Lexical Overlap:    {best_overlap} ({overlap_scores[best_overlap]*100:.1f}%)")
    print(f"  🏆 Best Keyword Diversity:  {best_diversity} ({diversity_scores[best_diversity]} unique keywords)")
    print("=" * 75)


if __name__ == '__main__':
    main()
