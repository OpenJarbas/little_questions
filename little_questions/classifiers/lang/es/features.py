from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from little_questions.classifiers.features import normalize
from little_questions.classifiers.lang.es.postag import pos_tag_es, \
    load_es_tagger
from nltk.stem.snowball import SnowballStemmer


class LemmatizerTransformerES(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize(X, stemmer=SnowballStemmer('spanish'),
                         **transform_params)


class POSTaggerTransformerES(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tagger = load_es_tagger()

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []
        for sent in X:
            tagged_words = pos_tag_es(sent, self.tagger)
            tag = {}
            for idx, w in enumerate(tagged_words):
                tag[w[0]] = w[1]
                tag[idx] = w[1]
                tag["contains_" + w[1]] = True
            feats += [tag]

        return feats


class POSTaggerVectorizerES(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_transformer = POSTaggerTransformerES()
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



