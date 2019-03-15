import re
import nltk


def normalize(X, stemmer=None):
    documents = []
    if stemmer is None:
        stemmer = nltk.stem.WordNetLemmatizer()
    if isinstance(stemmer, str):
        if "porter" in stemmer.lower():
            stemmer = nltk.PorterStemmer()
        elif "wordnet" in stemmer.lower():
            stemmer = nltk.stem.WordNetLemmatizer()
        elif "snowball" in stemmer.lower():
            stemmer = nltk.stem.SnowballStemmer("english",
                                                ignore_stopwords=True)

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
    return documents
