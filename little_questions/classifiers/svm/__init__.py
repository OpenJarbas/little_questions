from sklearn.svm import SVC, LinearSVC
from little_questions.classifiers import QuestionClassifier,  \
    SimpleQuestionClassifier


class SVCQuestionClassifier(QuestionClassifier):
    def __init__(self, name="svc"):
        super().__init__(name)

    @property
    def classifier_class(self):
        return SVC(kernel='poly')

    @property
    def parameters(self):
        return {'features__text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'features__text__tfidf__use_idf': (True, False),
                'clf__probability': (True, False),
                'clf__shrinking': (True, False),
                'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid',
                                'precomputed'),
                'clf__decision_function_shape': ('ovo', 'ovr')}


class SimpleSVCQuestionClassifier(SVCQuestionClassifier,
                                  SimpleQuestionClassifier):
    def __init__(self, name="svc_main"):
        super().__init__(name)


class LinearSVCQuestionClassifier(QuestionClassifier):
    def __init__(self, name="linear_svc"):
        super().__init__(name)

    @property
    def classifier_class(self):
        return LinearSVC()

    @property
    def parameters(self):
        return {'features__text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'features__text__tfidf__use_idf': (True, False),
                'clf__probability': (True, False),
                'clf__shrinking': (True, False),
                'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid',
                                'precomputed'),
                'clf__decision_function_shape': ('ovo', 'ovr')}


class SimpleLinearSVCQuestionClassifier(LinearSVCQuestionClassifier,
                                        SimpleQuestionClassifier):
    def __init__(self, name="linear_svc_main"):
        super().__init__(name)


if __name__ == '__main__':
    train = True
    clf = SVCQuestionClassifier()
    if train:
        t, tt = clf.load_data()
        clf.train(t, tt)
        clf.save()
    else:
        clf.load()
    # model performance
    # Accuracy: 0.752 - sigmoid kernel
    # Accuracy: 0.11 - rbf kernel
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
    # for q in questions:
    #    print(q, clf.predict([q]))
