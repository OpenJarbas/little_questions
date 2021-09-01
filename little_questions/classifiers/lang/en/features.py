from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from little_questions.classifiers.features import normalize
from little_questions.classifiers.lang.en import YES_NO_STARTERS, \
    COMMAND_STARTERS
import nltk


class LemmatizerTransformerEN(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize(X, stemmer=nltk.stem.WordNetLemmatizer(),
                         **transform_params)


class POSTaggerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []
        for sent in X:
            words = nltk.word_tokenize(sent)
            tagged_words = nltk.pos_tag(words)
            tag = {}
            for idx, w in enumerate(tagged_words):
                tag[w[0]] = w[1]
                tag[idx] = w[1]
                tag["contains_" + w[1]] = True
            feats += [tag]

        return feats


class POSTaggerVectorizerEN(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_transformer = POSTaggerTransformer()
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


class QuestionFeaturesTransformerEN(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []

        for sent in X:
            first_word = sent.split(" ")[0]
            s_feature = {
                'is_yes_no': first_word in YES_NO_STARTERS,
                'is_wh': sent.startswith('wh'),
                "is_command": first_word in COMMAND_STARTERS
            }
            feats += [s_feature]
        return feats


class QuestionFeaturesVectorizerEN(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transformer = QuestionFeaturesTransformerEN()
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
