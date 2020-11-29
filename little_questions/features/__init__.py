from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
import nltk
from little_questions.features.preprocess import normalize


class LemmatizerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize(X, **transform_params)


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


class POSTaggerVectorizer(BaseEstimator, TransformerMixin):
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


class NERTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []
        for sent in X:
            words = nltk.word_tokenize(sent)
            tagged_words = nltk.pos_tag(words)

            named_entity = nltk.ne_chunk(tagged_words)
            post_tag = {}
            for idx, x in enumerate(named_entity):
                if isinstance(x, nltk.tree.Tree):
                    post_tag.update(
                        {idx: x.__dict__["_label"],
                         " ".join([e[1] for e in x]): "pos_tag",
                         " ".join([e[0] for e in x]): x.__dict__["_label"]}
                    )
            feats += [post_tag]
        return feats


class NERVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ner_transformer = NERTransformer()
        self._dict_vectorizer = DictVectorizer()

    def get_feature_names(self):
        return self._dict_vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        X = self._ner_transformer.transform(X, **kwargs)
        self._dict_vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = self._ner_transformer.transform(X, **transform_params)
        return self._dict_vectorizer.transform(X)


if __name__ == "__main__":
    from pprint import pprint

    v = NERVectorizer()
    x = ["Elon Musk works for SpaceX", "Elon Musk was born in Pretoria"]
    # d = DictVectorizer()
    # v = v.fit_transform(x)

    print(v.fit_transform(x))
