from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from little_questions.classifiers import QuestionClassifier, DictTransformer, \
    TextTransformer
from sklearn.feature_extraction import DictVectorizer


class LogRegQuestionClassifier(QuestionClassifier):
    def __init__(self):
        super().__init__("logreg")

    @property
    def pipeline(self):
        return [
            ('features', FeatureUnion([
                ('text', Pipeline([('norm', TextTransformer()),
                                   ('vect', CountVectorizer()),
                                   ('tfidf', TfidfTransformer())])),
                ('intent', Pipeline([('dict', DictTransformer()),
                                     ('dict_vec', DictVectorizer())]))])),
            ('clf', LogisticRegression(multi_class="multinomial",
                                       solver="lbfgs"))
        ]

    @property
    def parameters(self):
        return {'features__text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'features__text__tfidf__use_idf': (True, False),
                'clf__warm_start': (True, False),
                'clf__solver': (
                    'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}


class SimpleLogRegQuestionClassifier(LogRegQuestionClassifier):
    def __init__(self, name="logreg_main"):
        super().__init__(name)


if __name__ == '__main__':
    train = False
    search = False
    clf = LogRegQuestionClassifier()
    if search:
        t, tt = clf.load_data()
        clf.grid_search(t, tt)
    if train:
        t, tt = clf.load_data()

        clf.train(t, tt)
        clf.save()
    else:
        clf.load()
    # model performance
    # Accuracy: 0.7 - normalize + dict + count + tfidf
    # Accuracy: 0.684 - dict + count + tfidf
    # Accuracy: 0.678 - count + tfidf
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
