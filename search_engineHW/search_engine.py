# -------------------------------------------------------------
# AUTHOR: Bryce Jimenez Stone
# FILENAME: search_engine.py
# SPECIFICATION: This program implements a search engine for Question 7.
#               It reads documents from collection.csv, processes a query "I love dogs"
#               using tokenization, stopword removal (pronouns, conjunctions, articles),
#               Porter stemming, and creates binary vectors with unigrams and bigrams.
#               Documents are ranked by dot product with the query vector.
# FOR: CS 5180 - Assignment #1
# TIME SPENT: 3 hours
# -----------------------------------------------------------*/

# ---------------------------------------------------------
# Importing some Python libraries
# ---------------------------------------------------------
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

documents = []

# ---------------------------------------------------------
# Reading the data in a csv file
# ---------------------------------------------------------
with open('collection.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            documents.append(row[0])

# ---------------------------------------------------------
# Print original documents
# ---------------------------------------------------------
# --> add your Python code here
print("Documents:", documents)


# ---------------------------------------------------------
# Define a custom tokenizer function with Porter stemmer and stopwords
# ---------------------------------------------------------
def stemmed_words(text):
    """
    Custom tokenizer that performs:
    - Tokenization and lowercasing
    - Stopword removal (pronouns, conjunctions, articles)
    - Porter stemming
    """
    # Define stopwords (pronouns, conjunctions, articles)
    stop_words = {'i', 'me', 'my', 'mine', 'myself',
                  'you', 'your', 'yours', 'yourself',
                  'he', 'him', 'his', 'himself',
                  'she', 'her', 'hers', 'herself',
                  'it', 'its', 'itself',
                  'we', 'us', 'our', 'ours', 'ourselves',
                  'they', 'them', 'their', 'theirs', 'themselves',
                  'a', 'an', 'the',
                  'and', 'but', 'or', 'for', 'nor', 'so', 'yet'}

    # Initialize stemmer
    stemmer = PorterStemmer()

    # Tokenization and lowercasing
    text = text.lower()
    # Simple tokenization by splitting on whitespace
    tokens = text.split()

    # Remove punctuation from tokens
    cleaned_tokens = []
    for token in tokens:
        # Remove punctuation from start and end of token
        token = token.strip('.,!?;:"()[]{}')
        if token:  # if token is not empty after stripping
            cleaned_tokens.append(token)

    # Stopword removal and stemming
    result = []
    for token in cleaned_tokens:
        if token not in stop_words:
            result.append(stemmer.stem(token))

    return result


# ---------------------------------------------------------
# Instantiate CountVectorizer informing 'word' as the analyzer, Porter stemmer as the tokenizer,
# stop_words as the identified stop words, unigrams and bigrams as the ngram_range,
# and binary representation as the weighting scheme
# ---------------------------------------------------------
# --> add your Python code here
vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=stemmed_words,
    ngram_range=(1, 2),  # unigrams and bigrams
    binary=True,  # binary term weights
    stop_words=None  # we handle stopwords in our tokenizer
)

# ---------------------------------------------------------
# Fit the vectorizer to the documents and encode them
# ---------------------------------------------------------
# --> add your Python code here
vectorizer.fit(documents)
document_matrix = vectorizer.transform(documents)

# ---------------------------------------------------------
# Inspect vocabulary
# ---------------------------------------------------------
print("Vocabulary:", vectorizer.get_feature_names_out().tolist())

# ---------------------------------------------------------
# Fit the vectorizer to the query and encode it
# ---------------------------------------------------------
# --> add your Python code here
query = "I love dogs"
query_vector = vectorizer.transform([query])

# ---------------------------------------------------------
# Convert matrices to plain Python lists
# ---------------------------------------------------------
# --> add your Python code here
doc_vectors = document_matrix.toarray().tolist()
query_vector = query_vector.toarray().flatten().tolist()

# ---------------------------------------------------------
# Compute dot product
# ---------------------------------------------------------
scores = []
# --> add your Python code here
for i, doc_vec in enumerate(doc_vectors):
    score = 0
    for j in range(len(query_vector)):
        score += query_vector[j] * doc_vec[j]
    scores.append(score)

print("Scores:", scores)

# ---------------------------------------------------------
# Sort documents by score (descending)
# ---------------------------------------------------------
ranking = []
# --> add your Python code here
# Create list of (index, score) tuples
doc_scores = [(i, scores[i]) for i in range(len(scores))]
# Sort by score in descending order
ranking = sorted(doc_scores, key=lambda x: x[1], reverse=True)

print("Ranking (doc_index, score):", ranking)