from little_questions.features import _parse
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import numpy as np
from os.path import join
import json
from little_questions.settings import DATA_PATH, MODELS_PATH


class TrainModel(object):
    def __init__(self, filepath=join(DATA_PATH, "questions.txt")):
        self.sents = []
        self.cat = []
        self.cat2 = []
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            words = l.split(" ")
            self.cat += [words[0].split(":")[0]]
            self.cat2 += [words[0].split(":")[1]]
            self.sents += [" ".join(words[1:])]

    def label_encoder(self):
        lencoder = LabelEncoder().fit(self.cat)
        print('Dumping Label Encoder to models directory')
        joblib.dump(lencoder, join(MODELS_PATH, 'label_encoder.pkl'))
        return lencoder

    def feature_extraction(self):
        # Get Special Features
        print('Loading vectorizing models')
        dict_vect = joblib.load(join(MODELS_PATH, 'dict_vect.pkl'))
        tfidf_vect = joblib.load(join(MODELS_PATH, 'tfidf_vect.pkl'))

        print('Transforming documents to feature vectors')
        with open(join(DATA_PATH, "dict_features.json")) as fi:
            special_features = json.load(fi)
        spec_features = dict_vect.transform(special_features)
        tfidf_features = tfidf_vect.transform(self.sents).toarray()
        # Merge 2 feature vectors
        return np.concatenate((spec_features, tfidf_features), axis=1)

    def train(self):
        # Extract Features
        print('Extracting Features')
        X = self.feature_extraction()
        # Pre Processing
        print('Encoding Labels')
        lencoder = self.label_encoder()

        print('Training Model')
        model = GradientBoostingClassifier(
            random_state=42)  # Life Universe and Everything
        y = lencoder.transform(self.cat)

        model.fit(X, y)
        print('Training Complete, dumping model to models directory')
        joblib.dump(model, join(MODELS_PATH, 'GradientBoostingClassifier.pkl'))


class Test(object):
    def __init__(self, modelpath=join(MODELS_PATH,
                                      "GradientBoostingClassifier.pkl")):
        print('Loading vectorizing models')
        self.dict_vect = joblib.load(join(MODELS_PATH, 'dict_vect.pkl'))
        self.tfidf_vect = joblib.load(join(MODELS_PATH, 'tfidf_vect.pkl'))
        print('Loading model from directory')
        self.model = joblib.load(modelpath)

    def featurize(self, sent):
        print('Transforming sentence to feature vectors')
        special_features = [_parse(sent)]

        spec_features = self.dict_vect.transform(special_features)
        tfidf_features = self.tfidf_vect.transform([sent]).toarray()

        return np.concatenate((spec_features, tfidf_features), axis=1)

    def label_decoder(self, labels):
        lencoder = joblib.load(join(MODELS_PATH, 'label_encoder.pkl'))
        return lencoder.inverse_transform(labels)

    def predict(self, question):
        print('Creating feature vector')
        X_test = self.featurize(question)

        print('Predicting Question Type')
        predicted = self.label_decoder(self.model.predict(X_test))

        return predicted


if __name__ == "__main__":
    train = True
    if train:
        model = TrainModel()
        model.train()

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
                 "not a question"]
    predictor = Test()
    for q in questions:
        print(predictor.predict(q))

    # TODO test data set
