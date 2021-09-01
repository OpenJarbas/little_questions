from os.path import join, dirname

from little_questions.classifiers.base import *
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
    model_name = "questions6_svm_DE_googtx_0.7.0a1"
    dataset = "raw_questions_DE_googtx0.7.0a1.txt"
    clf = LinearSVCTextClassifier("de")
    _train_6(model_name, dataset, clf)


def train_svm_52():
    model_name = "questions52_svm_DE_googtx_0.7.0a1"
    dataset = "raw_questions_DE_googtx0.7.0a1.txt"
    clf = LinearSVCTextClassifier("de")
    _train_52(model_name, dataset, clf)


train_svm_6()
train_svm_52()
