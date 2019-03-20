from sklearn.naive_bayes import MultinomialNB
from little_questions.classifiers import QuestionClassifier, \
    SimpleQuestionClassifier


class NaiveQuestionClassifier(QuestionClassifier):
    def __init__(self):
        super().__init__("naive")

    @property
    def classifier_class(self):
        return MultinomialNB()

    @property
    def parameters(self):
        return {'features__text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'features__text__tfidf__use_idf': (True, False),
                'clf__fit_prior': (True, False)}


class SimpleNaiveQuestionClassifier(NaiveQuestionClassifier,
                                    SimpleQuestionClassifier):
    def __init__(self, name="naive_main"):
        super().__init__(name)

if __name__ == '__main__':
    train = True
    clf = NaiveQuestionClassifier()
    if train:
        t, tt = clf.load_data()
        clf.train(t, tt)
        clf.save()
    else:
        clf.load()
    # model performance
    # curacy: 0.518
    # Accuracy: Accuracy: 0.514
    # Accuracy: 0.402 - no dict features
    clf.evaluate_model()

    # visual inspection

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
    #for q in questions:
    #    print(q, clf.predict([q]))