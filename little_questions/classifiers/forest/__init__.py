from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from little_questions.classifiers import QuestionClassifier


class ForestQuestionClassifier(QuestionClassifier):
    def __init__(self):
        super().__init__("forest")

    def train(self, train_data, target_data, n_estimators=1000,
              random_state=0):
        self.text_clf = Pipeline([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf-forest', RandomForestClassifier(
                                      n_estimators=n_estimators,
                                      random_state=random_state)),
                                  ])
        self.text_clf = self.text_clf.fit(train_data, target_data)
        return self.text_clf

if __name__ == '__main__':
    train = True
    clf = ForestQuestionClassifier()
    if train:
        t, tt = clf.load_data()
        clf.train(t, tt)
        clf.save()
    else:
        clf.load()

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
        print(q, clf.predict([q]))
