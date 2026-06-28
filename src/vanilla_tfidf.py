import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple

class VanillaTFIDFExtractor:
    """
    Standard TF-IDF Keyword Extractor baseline.
    Computes TF-IDF scores for a collection of documents independently
    and extracts top keywords for individual documents.
    """
    def __init__(self, max_features: int = 10000, stop_words: str = 'english'):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
        self.tfidf_matrix = None
        self.feature_names = None

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fits the TF-IDF vectorizer on the document corpus and transforms it.
        
        Args:
            documents (List[str]): List of document texts.
            
        Returns:
            np.ndarray: TF-IDF matrix.
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(documents).toarray()
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        return self.tfidf_matrix

    def extract_top_keywords(self, doc_idx: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extracts the top-N keywords and their scores for a document at doc_idx.
        
        Args:
            doc_idx (int): Index of the document in the fitted corpus.
            top_n (int): Number of top keywords to return.
            
        Returns:
            List[Tuple[str, float]]: List of (keyword, TF-IDF score) sorted by score descending.
        """
        if self.tfidf_matrix is None or self.feature_names is None:
            raise ValueError("The TF-IDF extractor has not been fitted yet. Call fit_transform first.")
            
        if doc_idx < 0 or doc_idx >= len(self.tfidf_matrix):
            raise IndexError("doc_idx is out of bounds for the fitted corpus.")

        row = self.tfidf_matrix[doc_idx]
        sorted_indices = np.argsort(row)[::-1]
        
        top_keywords = []
        for idx in sorted_indices[:top_n]:
            if row[idx] > 0:
                top_keywords.append((self.feature_names[idx], float(row[idx])))
                
        return top_keywords
