import os

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CommonModule:
    @staticmethod
    def similarity_score(document1, document2):
        vectorizer = CountVectorizer()
        return cosine_similarity(vectorizer.fit_transform([document1, document2]))[0][1]

    @staticmethod
    def sentiment(line):
        return SentimentIntensityAnalyzer().polarity_scores(line)
