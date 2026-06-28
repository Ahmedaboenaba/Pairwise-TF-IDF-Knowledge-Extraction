"""
data_loader.py — Paired Document Collector for Pairwise TF-IDF

This script builds the dataset needed for the UAFGK-style Pairwise TF-IDF
keyword extraction method. It pairs each book synopsis (from Goodreads)
with its author's biography (from Wikipedia), so we can later find keywords
that are important in BOTH documents simultaneously.

Pipeline:
  1. Load Goodreads books CSV  →  (title, authors, description)
  2. For each unique author, fetch their Wikipedia biography
  3. Merge into paired rows:  book_title | author_name | book_synopsis | author_bio
  4. Save the clean pairs to  data/processed/pairs.csv

Usage:
  python src/data_loader.py
"""

import os
import pandas as pd
import wikipedia
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1.  LOAD THE GOODREADS BOOKS DATASET
# ---------------------------------------------------------------------------

def load_goodreads_books(filepath: str) -> pd.DataFrame:
    """
    Reads the Goodreads books CSV and keeps only the columns we need.
    Drops rows that have missing titles, authors, or descriptions.

    Args:
        filepath: path to the raw CSV (e.g. 'data/raw/books.csv')

    Returns:
        A cleaned DataFrame with columns: title, authors, description
    """
    print(f"[1/4] Loading Goodreads dataset from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find the dataset at '{filepath}'.\n"
            "Please download the Goodreads books CSV from Kaggle and place it there."
        )

    df = pd.read_csv(filepath, on_bad_lines='skip')

    # Keep only the three columns we need
    required_cols = ['title', 'author', 'description']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV. "
                             f"Available columns: {list(df.columns)}")

    df = df[required_cols].copy()

    # Drop rows where any of the three fields is missing
    before = len(df)
    df.dropna(subset=required_cols, inplace=True)
    after = len(df)
    print(f"   Loaded {before} rows, kept {after} after dropping missing values.")

    return df


# ---------------------------------------------------------------------------
# 2.  FETCH AUTHOR BIOGRAPHIES FROM WIKIPEDIA
# ---------------------------------------------------------------------------

def fetch_author_bios(author_names: list, min_words: int = 100) -> dict:
    """
    For each unique author name, tries to fetch their Wikipedia biography.
    Skips authors whose page is not found, is ambiguous, or whose bio
    is shorter than `min_words` words.

    Args:
        author_names: list of unique author name strings
        min_words:    minimum word count for a bio to be kept (default 100)

    Returns:
        A dict  { author_name: biography_text }
        Also prints a summary of how many were skipped and why.
    """
    print(f"\n[2/4] Fetching Wikipedia biographies for {len(author_names)} unique authors...")

    bios = {}                # successful fetches
    skipped_not_found = []   # author page does not exist
    skipped_ambiguous = []   # Wikipedia returned multiple possible pages
    skipped_too_short = []   # bio exists but is under min_words

    for name in tqdm(author_names, desc="   Fetching bios"):
        try:
            # Search Wikipedia for the author's page
            page = wikipedia.page(name, auto_suggest=True)
            bio_text = page.content

            # Check minimum length
            word_count = len(bio_text.split())
            if word_count < min_words:
                skipped_too_short.append(name)
                continue

            bios[name] = bio_text

        except wikipedia.exceptions.PageError:
            # No Wikipedia page matches this author name
            skipped_not_found.append(name)

        except wikipedia.exceptions.DisambiguationError:
            # Multiple pages match — we can't pick automatically
            skipped_ambiguous.append(name)

        except Exception as e:
            # Catch any other network / API errors gracefully
            skipped_not_found.append(name)

    # Print a short report
    print(f"\n   Wikipedia fetch results:")
    print(f"     ✓ Biographies collected : {len(bios)}")
    print(f"     ✗ Not found on Wikipedia: {len(skipped_not_found)}")
    print(f"     ✗ Ambiguous (multiple)  : {len(skipped_ambiguous)}")
    print(f"     ✗ Too short (<{min_words} words): {len(skipped_too_short)}")

    return bios


# ---------------------------------------------------------------------------
# 3.  BUILD THE PAIRED DATAFRAME
# ---------------------------------------------------------------------------

def build_paired_dataframe(books_df: pd.DataFrame, author_bios: dict) -> pd.DataFrame:
    """
    Joins book rows with their author's Wikipedia bio.
    Only keeps rows where we successfully fetched a bio.

    The result has four columns:
        book_title    — title of the book
        author_name   — name of the author
        book_synopsis — the Goodreads description
        author_bio    — the Wikipedia biography

    Args:
        books_df:     DataFrame with columns (title, author, description)
        author_bios:  dict { author_name: bio_text }

    Returns:
        A clean paired DataFrame ready for Pairwise TF-IDF.
    """
    print(f"\n[3/4] Building paired book–author DataFrame...")

    rows = []
    for _, row in books_df.iterrows():
        author = row['author']

        # Some Goodreads entries list multiple authors separated by '/'
        # We take the first (primary) author only
        primary_author = author.split('/')[0].strip()

        if primary_author in author_bios:
            rows.append({
                'book_title':    row['title'],
                'author_name':   primary_author,
                'book_synopsis': row['description'],
                'author_bio':    author_bios[primary_author],
            })

    paired_df = pd.DataFrame(rows)
    print(f"   Created {len(paired_df)} book–author pairs.")

    return paired_df


# ---------------------------------------------------------------------------
# 4.  SAVE TO CSV
# ---------------------------------------------------------------------------

def save_pairs(paired_df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the paired DataFrame to a CSV file.
    Creates the output directory if it doesn't exist.

    Args:
        paired_df:   the paired DataFrame to save
        output_path: destination file path (e.g. 'data/processed/pairs.csv')
    """
    print(f"\n[4/4] Saving paired data to: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    paired_df.to_csv(output_path, index=False)

    print(f"   ✓ Saved {len(paired_df)} rows successfully.")


# ---------------------------------------------------------------------------
# 5.  MAIN — run the full pipeline
# ---------------------------------------------------------------------------

def main():
    """
    Runs the complete data collection pipeline:
      Goodreads CSV  →  Wikipedia bios  →  paired CSV
    """
    # --- Paths (relative to project root) ---
    RAW_BOOKS_PATH = os.path.join('data', 'raw', 'books.csv')
    OUTPUT_PATH    = os.path.join('data', 'processed', 'pairs.csv')

    # Step 1: Load books
    books_df = load_goodreads_books(RAW_BOOKS_PATH)

    # Step 2: Get unique authors and fetch their bios
    # Take only the primary author (before the '/' separator)
    books_df['primary_author'] = books_df['author'].apply(
        lambda x: x.split('/')[0].strip()
    )
    unique_authors = books_df['primary_author'].unique().tolist()
    author_bios = fetch_author_bios(unique_authors, min_words=100)

    # Step 3: Build paired DataFrame
    paired_df = build_paired_dataframe(books_df, author_bios)

    # Step 4: Save to CSV
    save_pairs(paired_df, OUTPUT_PATH)

    # --- Final summary ---
    print("\n" + "=" * 55)
    print("  DATA COLLECTION COMPLETE")
    print("=" * 55)
    print(f"  Total books in raw dataset  : {len(books_df)}")
    print(f"  Unique authors found        : {len(unique_authors)}")
    print(f"  Authors with valid bios     : {len(author_bios)}")
    print(f"  Final book–author pairs     : {len(paired_df)}")
    print(f"  Output saved to             : {OUTPUT_PATH}")
    print("=" * 55)


if __name__ == '__main__':
    main()
