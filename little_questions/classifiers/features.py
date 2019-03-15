from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def tfidf_vect(count_feats):
    """Tf-Idf takes into account the frequency of a word in a document,
    weighted by how frequently it appears in the entire corpus.
    Common words like “the” or “that” will have high term frequencies,
    but when you weigh them by the inverse of the document frequency,
    that would be 1 (because they appear in every document), and since TfIdf
    uses log values, that weight will actually be 0 since log 1 = 0.
    By comparison, if one document contains the word “soccer”,
    and it’s the only document on that topic out of a set of 100 documents,
     then the inverse frequency will be 100, so its Tf-Idf value will be
     boosted, signifying that the document is uniquely related to the topic
     of “soccer”. The TfidfVectorizer in sklearn will return a matrix with
     the tf-idf of each word in each document, with higher values for words
     which are specific to that document, and low (0) values for words that
     appear throughout the corpus."""
    tfidf_transformer = TfidfTransformer()
    return tfidf_transformer.fit_transform(count_feats)


def count_vect(data):
    """segment each text file into words and count # of times each word
    occurs in each document and finally assign each word an integer id.
    Each unique word in our dictionary will correspond to a feature """
    count_vect = CountVectorizer()
    return count_vect.fit_transform(data)
