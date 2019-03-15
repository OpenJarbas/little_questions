from little_questions.parsers import BasicQuestionParser
from little_questions.classifiers.preprocess import normalize
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from os.path import join
from little_questions.settings import MODELS_PATH, DATA_PATH, nlp, AFFIRMATIONS
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, \
    classification_report, confusion_matrix


class TextTransformer:
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize(X, **transform_params)


class DictTransformer:
    """ transofmr a list of sentences into a list of dicts """
    parser = BasicQuestionParser()

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []
        for sent in X:
            dict_feats = self.parser.parse(sent)
            text = nlp(sent)
            s_feature = {
                'tag': "",
                'is_wh': False,
                'is_affirmation': False
            }
            for token in text:
                if token.text.lower().startswith('wh'):
                    s_feature['is_wh'] = True
                if token.text.lower() in AFFIRMATIONS:
                    s_feature['is_affirmation'] = True
                s_feature['tag'] = token.tag_
                break
            s_feature.update(dict_feats)
            feats.append(s_feature)

        return feats


class QuestionClassifier(object):
    def __init__(self, name="default"):
        self.name = name
        self.text_clf = None

    def train(self, train_data, target_data):
        self.text_clf = Pipeline(self.pipeline)
        self.text_clf.fit(train_data, target_data)
        return self.text_clf

    @property
    def pipeline(self):
        raise NotImplementedError

    @property
    def parameters(self):
        return {'features__text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'features__text__tfidf__use_idf': (True, False)}

    def grid_search(self, train_data, target_data):
        self.text_clf = Pipeline(self.pipeline)
        gs_clf = GridSearchCV(self.text_clf, self.parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(train_data, target_data)
        print("best_score", gs_clf.best_score_)
        print("best_params", gs_clf.best_params_)
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

    def load_data(self, filename=join(DATA_PATH, "questions.txt")):
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
        X_test, y_test = self.load_test_data(path)
        preds = self.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, preds))

        try:
            print("Precision:", precision_score(y_test, preds))
        except:
            pass
        print(classification_report(y_test, preds))
        return confusion_matrix(y_test, preds)

    def grid_search(self, train_data, target_data):
        parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                      'tfidf__use_idf': (True, False)}
        gs_clf = GridSearchCV(self.text_clf, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(train_data, target_data)
        print("best_score", gs_clf.best_score_)
        print("best_params", gs_clf.best_params_)
        return gs_clf
