from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from little_questions.classifiers.features import normalize
from little_questions.classifiers.lang.ca.postag import pos_tag_ca, \
    load_ca_tagger
from nltk.stem.snowball import SnowballStemmer


class LemmatizerTransformerCA(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        # TODO better stemmer
        return normalize(X, stemmer=SnowballStemmer('spanish'),
                         **transform_params)


class POSTaggerTransformerCA(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []
        tagger = load_ca_tagger()
        for sent in X:
            tagged_words = pos_tag_ca(sent, tagger)
            tag = {}
            for idx, w in enumerate(tagged_words):
                tag[w[0]] = w[1]
                tag[idx] = w[1]
                tag["contains_" + w[1]] = True
            feats += [tag]

        return feats


class POSTaggerVectorizerCA(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_transformer = POSTaggerTransformerCA()
        self._dict_vectorizer = DictVectorizer()

    def get_feature_names(self):
        return self._dict_vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        X = self._pos_transformer.transform(X)
        self._dict_vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = self._pos_transformer.transform(X, **transform_params)
        return self._dict_vectorizer.transform(X)



