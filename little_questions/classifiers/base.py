import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC as _LinearSVC
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.naive_bayes import MultinomialNB as _MultinomialNB
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
from sklearn.linear_model import \
    PassiveAggressiveClassifier as _PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier as _SGDClassifier
from sklearn.linear_model import Perceptron as _Perceptron

from little_questions.classifiers.lang import get_pipeline
from little_questions.models import get_model_path


# this is meant to be subclassed per language, need to create datasets and
# TODO train a proper classifier
class SentenceScorer:
    @staticmethod
    def predict(text):
        score = SentenceScorer.score(text)
        best = max(score, key=lambda key: score[key])
        return best

    @staticmethod
    def score(text):
        return {
            "question": SentenceScorer.question_score(text),
            "statement": SentenceScorer.statement_score(text),
            "exclamation": SentenceScorer.exclamation_score(text),
            "command": SentenceScorer.command_score(text),
            "request": SentenceScorer.request_score(text)
        }

    @staticmethod
    def question_score(text):
        if text.endswith("?"):
            return 0.8
        return 0.4

    @staticmethod
    def statement_score(text):
        if text.endswith("."):
            return 0.5
        return 0

    @staticmethod
    def exclamation_score(text):
        if text.endswith("!"):
            return 0.6
        return 0

    @staticmethod
    def command_score(text):
        if text.endswith("."):
            return 0.6
        if text.endswith("!"):
            return 0.5
        return 0

    @staticmethod
    def request_score(text):
        if text.endswith("."):
            return 0.5
        if text.endswith("?"):
            return 0.5
        return 0


class Classifier:
    def __init__(self, pipeline_id):
        self.pipeline_id = pipeline_id.lower().split("-")[0]
        self.clf = None

    def train(self, train_data, target_data):
        raise NotImplementedError

    @property
    def pipeline(self):
        return []

    def predict(self, text):
        return self.clf.predict(text)

    def save(self, path):
        joblib.dump(self.clf, path)

    def load_from_file(self, path=None):
        path = path or get_model_path(self.pipeline_id)
        self.clf = joblib.load(path)
        return self


class LinearSVCTextClassifier(Classifier):
    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _LinearSVC())
        ]


class LogRegTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _LogisticRegression(multi_class="multinomial",
                                        solver="lbfgs"))
        ]


class RandomForestTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _RandomForestClassifier())
        ]


class NaiveBayesTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _MultinomialNB())
        ]


class PassiveAggressiveTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _PassiveAggressiveClassifier(loss="hinge"))
        ]


class SGDTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3))
        ]


class PerceptronTextClassifier(Classifier):

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_pipeline(self.pipeline_id)),
            ('clf', _Perceptron())
        ]

