from little_questions.classifiers import QuestionClassifier
from little_questions.settings import DATA_PATH, MODELS_PATH
from os.path import join
from sklearn.metrics import accuracy_score, precision_score, \
    classification_report, confusion_matrix
from sklearn.externals import joblib


class MultiModelClassifier(QuestionClassifier):

    def __init__(self, name="default"):
        super().__init__(name)
        self.models = {}
        self.main_model = None

    def load_data_per_label(self, filename=join(DATA_PATH, "questions.txt")):
        data = {}
        with open(filename, 'r') as f:
            for line in f:
                label = line.split(" ")[0]
                question = " ".join(line.split(" ")[1:]).strip()
                main = label.strip().split(":")[0]
                secondary = label.strip().split(":")[1]
                if main not in data:
                    data[main] = {}
                if secondary not in data[main]:
                    data[main][secondary] = []
                data[main][secondary] += [question]
        return data

    def train_per_label(self, data=None):
        data = data or self.load_data_per_label()
        for main in data:
            train = []
            target = []
            for sec in data[main]:
                for q in data[main][sec]:
                    train += [q]
                    target += [sec]
            self.models[main] = self.train(train, target)
            path = join(MODELS_PATH, self.name + '_' + main + '_model.pkl')
            self.save(path)
        self.text_clf = self.main_model
        return self.models

    def predict(self, text):
        return self.predict_main(text)[0], self.predict_secondary(text)[0]

    def predict_main(self, text):
        return self.main_model.predict(text)

    def predict_secondary(self, text, main_label=None):
        main_label = main_label or self.predict_main(text)[0]
        if main_label in self.models:
            clf = self.models[main_label]
            return clf.predict(text)
        return []

    def evaluate_model(self, path=join(DATA_PATH, "questions_test.txt")):
        data = self.load_data_per_label(path)
        X_test = []
        y_test = []
        for main in data:
            d = data[main]
            x = []
            y = []
            for sec in d:
                x += d[sec]
                y += [sec] * len(d[sec])
            X_test += x
            y_test += y

            preds = self.predict_secondary(x, main)
            print(main, "Accuracy:", accuracy_score(y, preds))

        preds = self.predict_secondary(X_test)
        print("Global Accuracy:", accuracy_score(y_test, preds))

        try:
            print("Precision:", precision_score(y_test, preds))
        except:
            pass
        print(classification_report(y_test, preds))
        return confusion_matrix(y_test, preds)

    def load(self, path=MODELS_PATH):
        main_path = join(path, self.name + '_main_model.pkl')
        self.main_model = joblib.load(main_path)
        for cat in ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]:
            m_path = join(path, self.name + "_" + cat + "_model.pkl")
            self.models[cat] = joblib.load(m_path)
        return self



if __name__ == "__main__":
    mdl = "sgd"
    clf = MultiModelClassifier(mdl).load()
    clf.evaluate_model()
    """
    given the main label, predict the secondary
    
    sgd
    
    ENTY Accuracy: 0.5531914893617021
    ABBR Accuracy: 1.0
    HUM Accuracy: 0.9230769230769231
    NUM Accuracy: 0.7787610619469026
    DESC Accuracy: 0.9637681159420289
    LOC Accuracy: 0.9259259259259259

    logreg
    
    ENTY Accuracy: 0.40425531914893614
    HUM Accuracy: 0.8615384615384616
    DESC Accuracy: 0.9637681159420289
    ABBR Accuracy: 1.0
    NUM Accuracy: 0.672566371681416
    LOC Accuracy: 0.8888888888888888
    
    naive
    
    ENTY Accuracy: 0.1276595744680851
    LOC Accuracy: 0.7037037037037037
    NUM Accuracy: 0.6106194690265486
    HUM Accuracy: 0.8461538461538461
    DESC Accuracy: 0.9637681159420289
    ABBR Accuracy: 0.8888888888888888
    """