from little_questions.classifiers.lang.en import SentenceScorerEN
from little_questions.classifiers.base import Classifier, \
    SentenceScorer, LinearSVCTextClassifier

_LAZY_LOADING = {}


def get_scorer(lang=None):
    if lang:
        lang = lang.lower()
        if lang.startswith("en"):
            return SentenceScorerEN()
    return SentenceScorer()


def get_classifier(model_id):
    global _LAZY_LOADING
    model_id = model_id.lower()
    if model_id in _LAZY_LOADING:
        return _LAZY_LOADING[model_id]
    classifier = Classifier(model_id)
    classifier.load_from_file()
    _LAZY_LOADING[model_id] = classifier
    return classifier
