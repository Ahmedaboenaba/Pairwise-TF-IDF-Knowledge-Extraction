import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Union

class PairwiseTFIDFExtractor:
    """
    Pairwise TF-IDF Knowledge Extractor.
    Extracts keywords representing the mutual/shared information between a pair
    of documents (e.g., book synopsis + author biography) by combining their
    individual TF-IDF representations.
    """
    def __init__(self, max_features: int = 10000, stop_words: str = 'english'):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
        self.feature_names = None
        self.fitted = False

    def fit(self, all_documents: List[str]):
        """
        Fits the TF-IDF vectorizer on a corpus representing all available texts.
        This establishes the vocabulary and IDF weights.
        
        Args:
            all_documents (List[str]): Full collection of documents (e.g. combined books + authors).
        """
        self.vectorizer.fit(all_documents)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.fitted = True

    def _get_tfidf_dict(self, text: str) -> Dict[str, float]:
        """
        Computes TF-IDF scores for a single document text and returns a word -> score dict.
        """
        if not self.fitted:
            raise ValueError("Extractor must be fitted with a corpus before transforming texts.")
        
        # Transform the single text
        vector = self.vectorizer.transform([text]).toarray()[0]
        
        # Build mapping of word -> score
        return {self.feature_names[i]: float(vector[i]) for i in np.nonzero(vector)[0]}

    def extract_pairwise_keywords(
        self, 
        doc_a: str, 
        doc_b: str, 
        metric: str = 'product', 
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extracts mutual keywords between two paired documents.
        
        Args:
            doc_a (str): First document text (e.g. book synopsis).
            doc_b (str): Second document text (e.g. author biography).
            metric (str): Combination metric. Options:
                - 'product': tfidf_a(w) * tfidf_b(w)
                - 'sum': tfidf_a(w) + tfidf_b(w)
                - 'min': min(tfidf_a(w), tfidf_b(w))  (intersection focus)
                - 'harmonic': 2 * (tfidf_a * tfidf_b) / (tfidf_a + tfidf_b)
            top_n (int): Number of top keywords to return.
            
        Returns:
            List[Tuple[str, float]]: List of (keyword, pairwise_score) sorted descending.
        """
        scores_a = self._get_tfidf_dict(doc_a)
        scores_b = self._get_tfidf_dict(doc_b)
        
        # Find shared words (intersection) or union depending on metric needs
        # For 'product', 'min', and 'harmonic', we only care about terms present in both
        if metric in ['product', 'min', 'harmonic']:
            shared_words = set(scores_a.keys()).intersection(set(scores_b.keys()))
        else:
            # For 'sum', we might look at union
            shared_words = set(scores_a.keys()).union(set(scores_b.keys()))
            
        pairwise_scores = {}
        
        for word in shared_words:
            sa = scores_a.get(word, 0.0)
            sb = scores_b.get(word, 0.0)
            
            if metric == 'product':
                score = sa * sb
            elif metric == 'sum':
                score = sa + sb
            elif metric == 'min':
                score = min(sa, sb)
            elif metric == 'harmonic':
                if (sa + sb) > 0:
                    score = 2 * (sa * sb) / (sa + sb)
                else:
                    score = 0.0
            else:
                raise ValueError(f"Unknown pairwise metric: {metric}")
                
            pairwise_scores[word] = score
            
        # Sort by score descending
        sorted_keywords = sorted(pairwise_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]
