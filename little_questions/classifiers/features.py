import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
import nltk


class WordFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []

        def is_numeric(input_str):
            try:
                float(input_str)
                return True
            except ValueError:
                return False

        for sent in X:
            toks = nltk.word_tokenize(sent)
            wfeat = {}
            for idx, w in enumerate(toks):
                wfeat.update({
                    f'word_{idx}': w,
                    f'is_first_{idx}': idx == 0,
                    f'is_last_{idx}': idx == len(toks) - 1,
                    f'is_capitalized_{idx}': w[0].upper() == w[0],
                    f'is_all_caps_{idx}': w.upper() == w,
                    f'is_all_lower_{idx}': w.lower() == w,
                    f'prefix-1_{idx}': w[0],
                    f'prefix-2_{idx}': w[:2],
                    f'prefix-3_{idx}': w[:3],
                    f'suffix-1_{idx}': w[-1],
                    f'suffix-2_{idx}': w[-2:],
                    f'suffix-3_{idx}': w[-3:],
                    f'prev_word_{idx}': '' if idx == 0 else toks[idx - 1],
                    f'next_word_{idx}': '' if idx == len(toks) - 1 else toks[idx + 1],
                    f'has_hyphen_{idx}': '-' in w,
                    f'is_numeric_{idx}': is_numeric(w),
                    f'capitals_inside_{idx}': w[1:].lower() != w[1:]
                })
            feats += [wfeat]
        return feats


class WordFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transformer = WordFeaturesTransformer()
        self._vectorizer = DictVectorizer(sparse=False)

    def get_feature_names(self):
        return self._vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        X = self._transformer.transform(X)
        self._vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = self._transformer.transform(X, **transform_params)
        return self._vectorizer.transform(X)


def normalize(X, stemmer=None, lemmatize=True):
    documents = []

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
        if lemmatize:
            stemmer = stemmer or nltk.PorterStemmer()
            document = document.split()
            try:
                document = [stemmer.lemmatize(word) for word in document]
            except:
               pass

            document = ' '.join(document)

        documents.append(document)
    return documents
