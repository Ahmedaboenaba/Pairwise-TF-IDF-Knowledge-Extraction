import nltk
from typing import List, Tuple

# Attempt to import rake-nltk, provide a fallback warning or custom implementation skeleton if not installed
try:
    from rake_nltk import Rake
except ImportError:
    # We will provide a simple mockup/warning if rake-nltk isn't installed yet
    Rake = None

class RakeExtractor:
    """
    RAKE (Rapid Automatic Keyword Extraction) baseline extractor.
    Extracts key phrases from individual documents based on word co-occurrences.
    """
    def __init__(self, language: str = 'english', min_length: int = 1, max_length: int = 4):
        self.language = language
        self.min_length = min_length
        self.max_length = max_length
        
        # Ensure NLTK punkt and stopwords are downloaded (required by rake-nltk)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
            
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")
            
        if Rake is not None:
            self.rake = Rake(
                language=self.language,
                min_length=self.min_length,
                max_length=self.max_length
            )
        else:
            self.rake = None
            print("[Warning] rake-nltk is not installed. RakeExtractor will return mock results until dependency is installed.")

    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extracts top key phrases and their RAKE scores from a given text.
        
        Args:
            text (str): The raw document text.
            top_n (int): The number of top key phrases to retrieve.
            
        Returns:
            List[Tuple[str, float]]: List of (phrase, score) sorted by score descending.
        """
        if not text or not isinstance(text, str):
            return []
            
        if self.rake is not None:
            # rake-nltk usage
            self.rake.extract_keywords_from_text(text)
            # get ranked phrases with scores
            ranked_phrases_with_scores = self.rake.get_ranked_phrases_with_scores()
            return [(phrase, float(score)) for score, phrase in ranked_phrases_with_scores[:top_n]]
        else:
            # Fallback simple word frequency/mock if rake-nltk is missing
            words = [w.lower() for w in text.split() if len(w) > 3]
            freq = {}
            for w in words:
                freq[w] = freq.get(w, 0.0) + 1.0
            sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            return sorted_words[:top_n]
