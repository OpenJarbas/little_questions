from os.path import join, isfile

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer

from text_classifikation.classifiers import BaseClassifier
from text_classifikation.classifiers.pipelines import pipeline__text

from little_questions.settings import DATA_PATH, DEFAULT_CLASSIFIER, \
    DEFAULT_MAIN_CLASSIFIER, DEFAULT_SENTENCE_CLASSIFIER, MODELS_PATH
from little_questions.classifiers.pipelines import pipelines as base_pipelines, \
    pipeline_unions as base_pipeline_unions, default_pipeline
from little_questions.classifiers.features import DictTransformer


class QuestionClassifier(BaseClassifier):

    def __init__(self, name=DEFAULT_CLASSIFIER, auto_load=True):
        super().__init__(name)
        if auto_load and isfile(join(MODELS_PATH, name)):
            self.load(join(MODELS_PATH, name))
        elif auto_load and isfile(join(MODELS_PATH, name + '_model.pkl')):
            self.load(join(MODELS_PATH, name + '_model.pkl'))

    def load(self, path=None):
        path = path or join(MODELS_PATH, self.name + '_model.pkl')
        super().load(path)
        return self

    def save(self, path=None):
        path = path or join(MODELS_PATH, self.name + '_model.pkl')
        super().save(path)

    def find_best_pipeline(self, train_data, target_data, test_data,
                           test_label, pipelines=None, unions=None,
                           outfolder=None, save_all=False,
                           skip_existing=True, verbose=True):
        pipelines = pipelines or base_pipelines
        unions = unions or base_pipeline_unions
        super().find_best_pipeline(train_data, target_data, test_data,
                                   test_label, pipelines, unions, outfolder,
                                   save_all, skip_existing, verbose)

    @property
    def parameters(self):
        return {'features__text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'features__text__tfidf__use_idf': (True, False)}

    @property
    def pipeline(self):
        return [
            ('features', default_pipeline),
            ('clf', self.classifier_class)
        ]

    @staticmethod
    def load_data(filename=join(DATA_PATH, "questions.txt")):
        train_data = []
        target_data = []
        with open(filename, 'r') as f:
            for line in f:
                label = line.split(" ")[0]
                question = " ".join(line.split(" ")[1:])
                train_data.append(question.strip())
                target_data.append(label.strip())
        return train_data, target_data

    def load_test_data(self, filename=join(DATA_PATH, "questions_test.txt")):
        return self.load_data(filename)

    def evaluate_model(self, path=join(DATA_PATH, "questions_test.txt")):
        return super().evaluate_model(path)


class MainQuestionClassifier(QuestionClassifier):
    def __init__(self, name=DEFAULT_MAIN_CLASSIFIER):
        super().__init__(name)

    def load_data(self, filename=join(DATA_PATH, "questions.txt")):
        train_data = []
        target_data = []
        with open(filename, 'r') as f:
            for line in f:
                label = line.split(" ")[0]
                question = " ".join(line.split(" ")[1:])
                train_data.append(question.strip())
                target_data.append(label.strip().split(":")[0])  # main
        return train_data, target_data


class SentenceClassifier(BaseClassifier):
    def __init__(self, name=DEFAULT_SENTENCE_CLASSIFIER, auto_load=True):
        super().__init__(name)
        if auto_load and isfile(join(MODELS_PATH, name)):
            self.load(join(MODELS_PATH, name))
        elif auto_load and isfile(join(MODELS_PATH, name + '_model.pkl')):
            self.load(join(MODELS_PATH, name + '_model.pkl'))
    
    def load(self, path=None):
        path = path or join(MODELS_PATH, self.name + '_model.pkl')
        super().load(path)

    def save(self, path=None):
        path = path or join(MODELS_PATH, self.name + '_model.pkl')
        super().save(path)

    @staticmethod
    def load_data(filename=join(DATA_PATH, "sentences.txt")):
        train_data = []
        target_data = []
        with open(filename, 'r') as f:
            for line in f:
                label = line.split(" ")[0]
                question = " ".join(line.split(" ")[1:])
                train_data.append(question.strip())
                target_data.append(label.strip())
        return train_data, target_data

    def load_test_data(self, filename=join(DATA_PATH, "sentences_test.txt")):
        return self.load_data(filename)

    def evaluate_model(self, path=join(DATA_PATH, "sentences_test.txt")):
        return super().evaluate_model(path)


def best_pipeline(clf):
    train, train_label = clf.load_data()
    test, test_label = clf.load_test_data()
    best_score, best_pipelin, acs = clf.find_best_pipeline(train, train_label,
                                                           test, test_label)
    return best_score, best_pipelin


if __name__ == "__main__":
    from little_questions.classifiers import QuestionClassifier
    from little_questions.classifiers import MainQuestionClassifier

    classifier = QuestionClassifier()
    question = "who made you"
    preds = classifier.predict([question])
    assert preds[0] == "HUM:ind"

    classifier = MainQuestionClassifier()
    question = "who made you"
    preds = classifier.predict([question])
    assert preds[0] == "HUM"
