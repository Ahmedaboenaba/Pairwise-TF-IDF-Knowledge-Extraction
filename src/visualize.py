"""
visualize.py — Visualization of Keyword Extraction Results

Generates publication-ready plots comparing the three keyword
extraction methods: Vanilla TF-IDF, RAKE, and Pairwise TF-IDF.

Outputs (saved to results/ folder):
  1. Word clouds      — one per method (3 images)
  2. Bar chart        — top 15 most frequent keywords per method
  3. Metrics chart    — grouped bar chart comparing evaluation scores

Usage:
  python src/visualize.py

Note: After running evaluate.py, fill in the metric values in the
      EVALUATION METRICS section below before running this script.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter


# ===================================================================
# *** EVALUATION METRICS — FILL THESE IN AFTER RUNNING evaluate.py ***
# ===================================================================
# These are placeholder values. Replace them with the actual numbers
# printed by evaluate.py after you run all three extraction scripts.

METRICS = {
    'Vanilla TF-IDF': {
        'lexical_overlap': 0.0,    # e.g. 0.12 means 12%
        'keyword_diversity': 0.0,  # avg unique keywords per book
    },
    'RAKE': {
        'lexical_overlap': 0.0,
        'keyword_diversity': 0.0,
    },
    'Pairwise TF-IDF': {
        'lexical_overlap': 0.0,
        'keyword_diversity': 0.0,
    },
}
# ===================================================================


# Color palette — consistent colors for each method across all charts
METHOD_COLORS = {
    'Vanilla TF-IDF':  '#3498db',   # blue
    'RAKE':            '#e67e22',   # orange
    'Pairwise TF-IDF': '#2ecc71',   # green
}

# Wordcloud color schemes — one per method for visual distinction
WC_COLORMAPS = {
    'Vanilla TF-IDF':  'Blues',
    'RAKE':            'Oranges',
    'Pairwise TF-IDF': 'Greens',
}


# ===================================================================
# HELPER: Load and parse keyword CSVs
# ===================================================================

def load_keywords(filepath: str) -> list:
    """
    Loads a keyword CSV and returns a flat list of all keywords
    across all books (for frequency analysis).

    Args:
        filepath: path to keyword CSV (has 'keywords' column,
                  semicolon-separated)

    Returns:
        List of individual keyword strings (lowercased).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find '{filepath}'.")

    df = pd.read_csv(filepath)
    all_keywords = []

    for kw_str in df['keywords'].dropna():
        keywords = [kw.strip().lower() for kw in kw_str.split(';') if kw.strip()]
        all_keywords.extend(keywords)

    return all_keywords


def get_top_n_keywords(keywords: list, n: int = 15) -> list:
    """
    Counts keyword frequencies and returns the top N as
    (keyword, count) tuples.

    Args:
        keywords: flat list of keyword strings
        n:        how many top keywords to return

    Returns:
        List of (keyword, count) tuples, sorted descending.
    """
    counts = Counter(keywords)
    return counts.most_common(n)


# ===================================================================
# PLOT 1: WORD CLOUDS
# ===================================================================

def generate_word_clouds(keyword_data: dict, output_dir: str):
    """
    Creates one word cloud per method and saves each as a PNG.
    Word size reflects how frequently that keyword appears across
    all books for that method.

    Args:
        keyword_data: dict { method_name: [list of keywords] }
        output_dir:   directory to save images
    """
    print("\n[1/3] Generating word clouds...")

    filenames = {
        'Vanilla TF-IDF':  'wordcloud_vanilla.png',
        'RAKE':            'wordcloud_rake.png',
        'Pairwise TF-IDF': 'wordcloud_pairwise.png',
    }

    for method, keywords in keyword_data.items():
        if len(keywords) == 0:
            print(f"   [Skip] No keywords for {method}")
            continue

        # Build frequency dict from the keyword list
        freq = Counter(keywords)

        # Create the word cloud
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=WC_COLORMAPS.get(method, 'viridis'),
            max_words=100,
            max_font_size=80,
            min_font_size=10,
            prefer_horizontal=0.7,
        )
        wc.generate_from_frequencies(freq)

        # Save the word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f"Word Cloud — {method}", fontsize=16, fontweight='bold', pad=15)
        ax.axis('off')

        out_path = os.path.join(output_dir, filenames[method])
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"   ✓ Saved: {out_path}")


# ===================================================================
# PLOT 2: TOP 15 KEYWORDS BAR CHART (side by side)
# ===================================================================

def plot_top_keywords_comparison(keyword_data: dict, output_dir: str, top_n: int = 15):
    """
    Creates a figure with 3 horizontal bar subplots — one per method —
    showing the top N most frequent keywords across all books.

    Args:
        keyword_data: dict { method_name: [list of keywords] }
        output_dir:   directory to save the image
        top_n:        how many keywords to show per method
    """
    print("\n[2/3] Generating top keywords comparison bar chart...")

    methods = list(keyword_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"Top {top_n} Most Frequent Keywords by Method",
                 fontsize=18, fontweight='bold', y=1.02)

    for idx, method in enumerate(methods):
        ax = axes[idx]
        top_kw = get_top_n_keywords(keyword_data[method], n=top_n)

        if len(top_kw) == 0:
            ax.set_title(f"{method}\n(no data)", fontsize=13)
            continue

        # Reverse so highest-frequency keyword is at the top
        words  = [kw for kw, count in top_kw][::-1]
        counts = [count for kw, count in top_kw][::-1]
        color  = METHOD_COLORS.get(method, '#888888')

        ax.barh(words, counts, color=color, edgecolor='white', height=0.6)
        ax.set_xlabel("Frequency (across all books)", fontsize=11)
        ax.set_title(method, fontsize=14, fontweight='bold', pad=10)
        ax.tick_params(axis='y', labelsize=10)

        # Add count labels on the bars
        for i, (w, c) in enumerate(zip(words, counts)):
            ax.text(c + 0.3, i, str(c), va='center', fontsize=9, color='#333')

    plt.tight_layout()

    out_path = os.path.join(output_dir, 'top_keywords_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"   ✓ Saved: {out_path}")


# ===================================================================
# PLOT 3: EVALUATION METRICS GROUPED BAR CHART
# ===================================================================

def plot_metrics_comparison(output_dir: str):
    """
    Creates a grouped bar chart comparing the 3 methods on:
      - Lexical Overlap Score
      - Keyword Diversity Score

    Uses the values defined in the METRICS dict at the top of this file.

    Args:
        output_dir: directory to save the image
    """
    print("\n[3/3] Generating evaluation metrics comparison chart...")

    methods = list(METRICS.keys())
    overlap_vals   = [METRICS[m]['lexical_overlap'] * 100 for m in methods]   # convert to %
    diversity_vals = [METRICS[m]['keyword_diversity'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Evaluation Metrics Comparison",
                 fontsize=18, fontweight='bold', y=1.02)

    # --- Left chart: Lexical Overlap ---
    colors = [METHOD_COLORS[m] for m in methods]
    bars1 = ax1.bar(methods, overlap_vals, color=colors, edgecolor='white', width=0.5)

    ax1.set_ylabel("Lexical Overlap (%)", fontsize=12)
    ax1.set_title("Keywords Found in BOTH Documents", fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylim(0, max(overlap_vals) * 1.3 if max(overlap_vals) > 0 else 10)
    ax1.tick_params(axis='x', labelsize=10)

    # Add value labels on bars
    for bar, val in zip(bars1, overlap_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Right chart: Keyword Diversity ---
    bars2 = ax2.bar(methods, diversity_vals, color=colors, edgecolor='white', width=0.5)

    ax2.set_ylabel("Avg Unique Keywords / Book", fontsize=12)
    ax2.set_title("Keyword Diversity", fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0, max(diversity_vals) * 1.3 if max(diversity_vals) > 0 else 10)
    ax2.tick_params(axis='x', labelsize=10)

    for bar, val in zip(bars2, diversity_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    out_path = os.path.join(output_dir, 'metrics_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"   ✓ Saved: {out_path}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    """
    Loads all keyword results and generates all three visualization sets.
    """
    # --- Paths ---
    VANILLA_PATH  = os.path.join('results', 'keyword_outputs', 'vanilla_tfidf_keywords.csv')
    RAKE_PATH     = os.path.join('results', 'keyword_outputs', 'rake_keywords.csv')
    PAIRWISE_PATH = os.path.join('results', 'keyword_outputs', 'pairwise_tfidf_keywords.csv')
    OUTPUT_DIR    = 'results'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all keyword lists
    print("Loading keyword results...")
    keyword_data = {
        'Vanilla TF-IDF':  load_keywords(VANILLA_PATH),
        'RAKE':            load_keywords(RAKE_PATH),
        'Pairwise TF-IDF': load_keywords(PAIRWISE_PATH),
    }

    for method, kws in keyword_data.items():
        print(f"   {method}: {len(kws)} total keyword occurrences")

    # Generate all plots
    generate_word_clouds(keyword_data, OUTPUT_DIR)
    plot_top_keywords_comparison(keyword_data, OUTPUT_DIR)
    plot_metrics_comparison(OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 55)
    print("  ALL VISUALIZATIONS GENERATED")
    print("=" * 55)
    print(f"  📊 results/wordcloud_vanilla.png")
    print(f"  📊 results/wordcloud_rake.png")
    print(f"  📊 results/wordcloud_pairwise.png")
    print(f"  📊 results/top_keywords_comparison.png")
    print(f"  📊 results/metrics_comparison.png")
    print("=" * 55)

    # Remind user to fill in metrics if they are all zero
    all_zero = all(
        METRICS[m]['lexical_overlap'] == 0 and METRICS[m]['keyword_diversity'] == 0
        for m in METRICS
    )
    if all_zero:
        print("\n  ⚠️  The metrics chart uses placeholder values (all 0.0).")
        print("     Run evaluate.py first, then update the METRICS dict")
        print("     at the top of visualize.py with your actual scores.")
        print("     Then re-run: python src/visualize.py")


if __name__ == '__main__':
    main()
