from os.path import join, dirname

from little_questions.classifiers.base import LinearSVCTextClassifier, \
    LogRegTextClassifier, RandomForestTextClassifier, \
    NaiveBayesTextClassifier, PassiveAggressiveTextClassifier, \
    SGDTextClassifier, PerceptronTextClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xdg import BaseDirectory as XDG

DATA_PATH = join(dirname(__file__), "clean_data")
REPORTS_PATH = join(dirname(__file__), "reports")


def _train_6(model_name, dataset, clf):
    train_data_path = join(DATA_PATH, dataset)

    def load_data(filename):
        x = []
        y = []
        with open(filename, 'r') as f:
            for line in f:
                label = line.split(" ")[0]
                question = " ".join(line.split(" ")[1:])
                x.append(question.strip())
                y.append(label.split(":")[0].strip())

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.15,
                                                            stratify=y)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = load_data(train_data_path)

    clf.train(x_train, y_train)

    clf.save(join(XDG.save_data_path("little_questions"),
                  model_name + ".pkl"))

    preds = clf.predict(x_test)

    report = classification_report(y_test, preds)

    print(report)

    with open(join(REPORTS_PATH, model_name + ".txt"), "w") as f:
        f.write(report)


def _train_52(model_name, dataset, clf):
    train_data_path = join(DATA_PATH, dataset)

    def load_data(filename):
        x = []
        y = []
        with open(filename, 'r') as f:
            for line in f:
                label = line.split(" ")[0]
                question = " ".join(line.split(" ")[1:])
                x.append(question.strip())
                y.append(label.strip())

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.15,
                                                            stratify=y)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = load_data(train_data_path)

    clf.train(x_train, y_train)

    clf.save(join(XDG.save_data_path("little_questions"),
                  model_name + ".pkl"))

    preds = clf.predict(x_test)

    report = classification_report(y_test, preds)

    print(report)

    with open(join(REPORTS_PATH, model_name + ".txt"), "w") as f:
        f.write(report)


def train_svm_6():
    model_name = "questions6_svm_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = LinearSVCTextClassifier("en")
    _train_6(model_name, dataset, clf)


def train_svm_52():
    model_name = "questions52_svm_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = LinearSVCTextClassifier("en")
    _train_52(model_name, dataset, clf)


def train_nb_sent2():
    model_name = "questions2_nb_EN_0.7.0a1"
    dataset = "simple_tags_EN_0.7.0a1.txt"
    clf = NaiveBayesTextClassifier("naive")
    _train_6(model_name, dataset, clf)


# other experiments
def train_logreg_6():
    model_name = "questions6_lr_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = LogRegTextClassifier("en")
    _train_6(model_name, dataset, clf)


def train_forest_6():
    model_name = "questions6_forest_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = RandomForestTextClassifier("en")
    _train_6(model_name, dataset, clf)


def train_logreg_52():
    model_name = "questions52_lr_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = LogRegTextClassifier("en")
    _train_52(model_name, dataset, clf)


def train_forest_52():
    model_name = "questions52_forest_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = RandomForestTextClassifier("en")
    _train_52(model_name, dataset, clf)


def train_nb_6():
    model_name = "questions6_nb_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = NaiveBayesTextClassifier("naive")
    _train_6(model_name, dataset, clf)


def train_nb_52():
    model_name = "questions52_nb_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = NaiveBayesTextClassifier("naive")
    _train_52(model_name, dataset, clf)


def train_pa_6():
    model_name = "questions6_pa_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = PassiveAggressiveTextClassifier("en")
    _train_6(model_name, dataset, clf)


def train_pa_52():
    model_name = "questions52_pa_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = PassiveAggressiveTextClassifier("en")
    _train_52(model_name, dataset, clf)


def train_sgd_6():
    model_name = "questions6_sgd_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = SGDTextClassifier("en")
    _train_6(model_name, dataset, clf)


def train_sgd_52():
    model_name = "questions52_sgd_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = SGDTextClassifier("en")
    _train_52(model_name, dataset, clf)


def train_perceptron_6():
    model_name = "questions6_perceptron_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = PerceptronTextClassifier("en")
    _train_6(model_name, dataset, clf)


def train_perceptron_52():
    model_name = "questions52_perceptron_EN_0.7.0a1"
    dataset = "raw_questions_0.7.0a1.txt"
    clf = PerceptronTextClassifier("en")
    _train_52(model_name, dataset, clf)


train_perceptron_6()
train_perceptron_52()
