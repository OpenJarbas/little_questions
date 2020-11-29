from little_questions.settings import DATA_PATH, DEFAULT_CLASSIFIER, \
    DEFAULT_MAIN_CLASSIFIER, DEFAULT_SENTENCE_CLASSIFIER, MODELS_PATH
from little_questions.classifiers.features import DictTransformer
from little_questions.classifiers.pipelines import default_pipelines, \
    default_pipeline_unions, pipeline__text, default_pipeline
from os.path import join, isfile
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, f1_score, recall_score, precision_score
import json
from little_questions.utils.log import LOG


class BaseClassifier:
    def __init__(self, name):
        self.name = name.replace('_model.pkl', "")
        self.text_clf = None

    def find_best_pipeline(self, train_data, target_data, test_data,
                           test_label, pipelines=None, unions=None,
                           outfolder=None, save_all=False,
                           skip_existing=True, save_best=True, weights=None):
        weights = weights or {"f1": 0.5,
                              "acc": 0.2,
                              "recall": 0.15,
                              "precision": 0.15}
        pipelines = pipelines or default_pipelines
        unions = unions or default_pipeline_unions
        LOG.debug("Finding best of:" +
                  str(sorted(list(pipelines.keys()) +
                             list(unions.keys()))))
        best_score = 0
        best_clf = None
        best_p = None
        outfolder = outfolder or MODELS_PATH
        model_scores = {}
        if isfile(join(outfolder, self.name + ".json")):
            with open(join(outfolder, self.name + ".json"), "r") as f:
                model_scores = json.load(f)
            LOG.debug("resuming from: " + str(model_scores))
            best_p = max(model_scores,
                         key=lambda key: model_scores[key]["score"])
            best_score = model_scores[best_p]["score"]
            LOG.info("Current best: " + best_p + " - " + str(best_score))
        for p in pipelines:
            if p in model_scores:
                LOG.debug("skipping: " + p)
                continue
            try:
                path = join(outfolder, self.name + "_" + p + '_model.pkl')
                if isfile(path) and skip_existing:
                    LOG.debug("already fitted :" +
                              self.name + "_" + p + " - SKIPPING")

                    if self.name + "_" + p not in model_scores:
                        with open(join(outfolder, self.name + "_acc.json"),
                                  "w") as f:
                            json.dump(model_scores, f, indent=4)
                    continue

                LOG.debug("fitting:" + p)
                line = [
                    ('features', pipelines[p]),
                    ('clf', self.classifier_class)
                ]
                clf = Pipeline(line)
                clf.fit(train_data, target_data)
                preds = clf.predict(test_data)
                accuracy = accuracy_score(test_label, preds)
                recall = recall_score(test_label, preds,
                                      labels=np.unique(preds),
                                      average='weighted')
                f1 = f1_score(test_label, preds,
                              labels=np.unique(preds),
                              average='weighted')
                precision = precision_score(test_label, preds,
                                            labels=np.unique(preds),
                                            average='weighted')
                score = weights["f1"] * f1 + \
                        weights["acc"] * accuracy + \
                        weights["recall"] * recall + \
                        weights["precision"] * precision

                model_scores[p] = {"score": score, "f1": f1,
                                   "accuracy": accuracy, "recall": recall,
                                   "precision": precision}
                LOG.debug(p + " - F1_score: " + str(f1))
                LOG.debug(p + " - Accuracy: " + str(accuracy))
                LOG.debug(p + " - Recall: " + str(recall))
                LOG.debug(p + " - Precision: " + str(precision))
                LOG.info(p + " - Score: " + str(score))
                with open(join(outfolder, self.name + ".json"), "w") as f:
                    json.dump(model_scores, f, indent=4)
                if score == best_score:
                    LOG.info("Classifiers are tied: " + best_p + " - " + p)
                    LOG.debug("Keeping " + best_p)
                elif score > best_score:
                    if best_p:
                        LOG.debug(
                            "Old best: " + best_p + " - " + str(best_score))
                    best_score = score
                    best_clf = clf
                    best_p = p
                    LOG.info(
                        "Current best: " + best_p + " - " + str(best_score))
                if save_all:
                    joblib.dump(clf, path)
            except Exception as e:
                LOG.exception(e)
                LOG.error("bad pipeline: " + p)

        for p in unions:
            if p in model_scores:
                continue
            try:
                path = join(outfolder, self.name + "_" + p + '_model.pkl')
                if isfile(path) and skip_existing:
                    continue
                line = [
                    ('features', unions[p]),
                    ('clf', self.classifier_class)
                ]
                clf = Pipeline(line)
                clf.fit(train_data, target_data)
                preds = clf.predict(test_data)
                accuracy = accuracy_score(test_label, preds)
                recall = recall_score(test_label, preds,
                                      labels=np.unique(preds),
                                      average='weighted')
                f1 = f1_score(test_label, preds,
                              labels=np.unique(preds),
                              average='weighted')
                precision = precision_score(test_label, preds,
                                            labels=np.unique(preds),
                                            average='weighted')
                score = weights["f1"] * f1 + \
                        weights["acc"] * accuracy + \
                        weights["recall"] * recall + \
                        weights["precision"] * precision

                model_scores[p] = {"score": score, "f1": f1,
                                   "accuracy": accuracy, "recall": recall,
                                   "precision": precision}
                LOG.debug(p + " - F1_score: " + str(f1))
                LOG.debug(p + " - Accuracy: " + str(accuracy))
                LOG.debug(p + " - Recall: " + str(recall))
                LOG.debug(p + " - Precision: " + str(precision))
                LOG.info(p + " - Score: " + str(score))

                if score == best_score:
                    LOG.info("Classifiers are tied: " + best_p + " - " + p)
                    LOG.debug("Keeping " + best_p)
                elif score > best_score:
                    if best_p:
                        LOG.info(
                            "Old best: " + best_p + " - " + str(best_score))
                    best_score = score
                    best_clf = clf
                    best_p = p
                    LOG.info(
                        "Current best: " + best_p + " - " + str(best_score))
                with open(join(outfolder, self.name + ".json"), "w") as f:
                    json.dump(model_scores, f, indent=4, sort_keys=True)
                if save_all:
                    joblib.dump(clf, path)
            except Exception as e:
                LOG.exception(e)
                LOG.debug("bad union: " + p)
        best_pipeline = max(model_scores,
                            key=lambda key: model_scores[key]["score"])
        LOG.info("Current best: " + best_pipeline + " - " +
                 str(model_scores[best_pipeline]["score"]))
        if not save_all and save_best and best_clf:
            path = join(outfolder, self.name + "_" +
                        best_pipeline + '_model.pkl')
            LOG.info("Saving best model:" + path)
            joblib.dump(best_clf, path)
        return model_scores[best_pipeline][
                   "score"], best_pipeline, model_scores

    def train(self, train_data, target_data):
        self.text_clf = Pipeline(self.pipeline)
        self.text_clf.fit(train_data, target_data)
        return self.text_clf

    @property
    def pipeline(self):
        return [
            ('features', pipeline__text),
            ('clf', self.classifier_class)
        ]

    @property
    def classifier_class(self):
        raise NotImplementedError

    @property
    def parameters(self):
        return {'features__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'features__tfidf__use_idf': (True, False)}

    def grid_search(self, train_data, target_data):
        self.text_clf = Pipeline(self.pipeline)
        gs_clf = GridSearchCV(self.text_clf, self.parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(train_data, target_data)
        LOG.debug("best_score", gs_clf.best_score_)
        LOG.debug("best_params", gs_clf.best_params_)
        return gs_clf

    def predict(self, text):
        return self.text_clf.predict(text)

    def save(self, path=None):
        path = path or join(MODELS_PATH, self.name + '_model.pkl')
        joblib.dump(self.text_clf, path)

    def load(self, path=None):
        path = path or join(MODELS_PATH, self.name + '_model.pkl')
        self.text_clf = joblib.load(path)
        return self

    def load_data(self, filename):
        raise NotImplementedError
        return train_data, target_data

    def load_test_data(self, filename):
        return self.load_data(filename)

    def evaluate_model(self, path):
        X_test, y_test = self.load_test_data(path)
        preds = self.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        matrix = confusion_matrix(y_test, preds)
        return accuracy, report, matrix


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


def find_best_pipeline(clf, weights=None, **kwargs):
    weights = weights or {"f1": 0.5,
                          "acc": 0.15,
                          "recall": 0.2,
                          "precision": 0.15}
    train, train_label = clf.load_data()
    test, test_label = clf.load_test_data()
    best_score, best_pipelin, acs = clf.find_best_pipeline(train, train_label,
                                                           test, test_label,
                                                           weights=weights,
                                                           **kwargs)
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
