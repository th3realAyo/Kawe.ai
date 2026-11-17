import numpy as np
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import re

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

class TextRankSummarizer:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def clean_sentences(self, text):
        """
        Preprocess the input text into cleaned sentences
        """
        sentences = sent_tokenize(text)
        cleaned_sentences = [
            re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences
        ]
        return sentences, cleaned_sentences

    def sentence_similarity(self, sent1, sent2):
        """
        Compute cosine similarity between two sentences
        """
        sent1_words = [word for word in word_tokenize(sent1) if word not in self.stopwords]
        sent2_words = [word for word in word_tokenize(sent2) if word not in self.stopwords]
        
        all_words = list(set(sent1_words + sent2_words))
        vec1 = [sent1_words.count(word) for word in all_words]
        vec2 = [sent2_words.count(word) for word in all_words]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.sqrt(np.sum(np.square(vec1)))
        norm2 = np.sqrt(np.sum(np.square(vec2)))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def build_similarity_matrix(self, sentences):
        """
        Build a similarity matrix for the sentences
        """
        matrix = np.zeros((len(sentences), len(sentences)))
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences):
                if i != j:
                    matrix[i][j] = self.sentence_similarity(sent1, sent2)
        return matrix

    def summarize(self, text, num_sentences=3):
        """
        Summarize the text using TextRank
        """
        original_sentences, cleaned_sentences = self.clean_sentences(text)

        # Nothing to summarize, or too short to bother ranking.
        if not original_sentences:
            return ""
        if len(original_sentences) <= num_sentences:
            return ' '.join(original_sentences)

        similarity_matrix = self.build_similarity_matrix(cleaned_sentences)

        # Rank sentences using PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        # Rank sentence indices by score (highest first)
        ranked_indices = sorted(
            range(len(original_sentences)),
            key=lambda i: scores[i],
            reverse=True
        )

        # Select top N indices, then restore original document order so
        # the summary reads coherently instead of jumbled by score.
        top_indices = sorted(ranked_indices[:num_sentences])
        summary = ' '.join(original_sentences[i] for i in top_indices)
        return summary
