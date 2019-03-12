from sklearn import linear_model
import numpy as np
from sklearn.externals import joblib

from little_questions.features import featurize, label_encoder
from little_questions.settings import DATA_PATH, MODELS_PATH
from os.path import join


def load_data(filename):
    res = []
    with open(filename, 'r') as f:
        for line in f:
            label = line.split(" ")[0]
            question = " ".join(line.split(" ")[1:])
            res.append((question.strip(), label.strip()))
    return res


def train_logreg(training_data_path=join(DATA_PATH, "questions.txt")):
    train_data = load_data(training_data_path)
    question_vectors = np.asarray([featurize(line[0]) for line in train_data])
    train_labels = label_encoder.transform([line[1] for line in train_data])

    clf = linear_model.LogisticRegression(multi_class='multinomial',
                                          solver='lbfgs')
    clf.fit(question_vectors, train_labels)

    joblib.dump(clf, join(MODELS_PATH, 'logreg_model.pkl'))

    print("Saved model to disk")
    return clf


def predict_question_category(question, clf=None):

    clf = clf or joblib.load(join(MODELS_PATH, 'logreg_model.pkl'))

    return label_encoder.inverse_transform(
        clf.predict([featurize(question.lower())]))


if __name__ == '__main__':
    train = True
    if train:
        clf = train_logreg()
    else:
        clf = None

    questions = ["what do dogs and cats have in common",
                 "tell me about evil",
                 "how to kill animals ( a cow ) and make meat",
                 "what is a living being",
                 "why are humans living beings",
                 "give examples of animals",
                 "what is the speed of light",
                 "when were you born",
                 "where do you store your data",
                 "will you die",
                 "have you finished booting",
                 "should i program artificial stupidity",
                 "who made you",
                 "how long until sunset",
                 "how long ago was sunrise",
                 "how much is bitcoin worth",
                 "which city has more people",
                 "whose dog is this",
                 "did you know that dogs are animals",
                 "do you agree that dogs are animals",
                 "what time is it?",
                 "not a question"]
    for q in questions:
        print(predict_question_category(q, clf))
