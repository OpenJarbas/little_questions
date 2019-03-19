from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from little_questions.classifiers import QuestionClassifier, DictTransformer, \
    TextTransformer, SimpleQuestionClassifier
from sklearn.feature_extraction import DictVectorizer


class SGDQuestionClassifier(QuestionClassifier):
    def __init__(self):
        super().__init__("sgd")

    @property
    def pipeline(self):
        return [
            ('features', FeatureUnion([
                ('text', Pipeline([('norm', TextTransformer()),
                                   ('vect', CountVectorizer()),
                                   ('tfidf', TfidfTransformer())])),
                ('intent', Pipeline([('dict', DictTransformer()),
                                     ('dict_vec', DictVectorizer())]))])),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
              alpha=1e-3, n_iter=5, random_state=42))
        ]

    @property
    def parameters(self):
        return {'features__text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'features__text__tfidf__use_idf': (True, False),
                'clf__early_stopping': (True, False),
                'clf__shuffle': (True, False),
                'clf__learning_rate': ('constant', 'optimal', 'adaptive',
                                       'invscaling'),
                'clf__penalty': ('l1', 'l2', 'elasticnet'),
                'clf__loss': ('hinge', 'log', 'modified_huber',
                              'squared_hinge', 'perceptron'),
                'clf__fit_intercept': (True, False)}


class SimpleSGDQuestionClassifier(SGDQuestionClassifier, SimpleQuestionClassifier):
    def __init__(self, name="sgd_main"):
        super().__init__(name)

if __name__ == '__main__':
    train = True
    clf = SGDQuestionClassifier()
    if train:
        t, tt = clf.load_data()
        clf.train(t, tt)
        clf.save()
    else:
        clf.load()
        # clf.load("/home/user/PycharmProjects/question_parser/little_questions/models/logreg_no_dict_model.pkl")
    # model performance
    # Accuracy: 0.752 - normalize + dict + count + tfidf
    # Accuracy: 0.744 - count + tfidf
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
