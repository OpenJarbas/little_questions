from sklearn.base import BaseEstimator, TransformerMixin
from little_questions.classifiers.features import normalize
from nltk.stem.snowball import FrenchStemmer


class LemmatizerTransformerFR(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize(X, stemmer=FrenchStemmer(),
                         **transform_params)

