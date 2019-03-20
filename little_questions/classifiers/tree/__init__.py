from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from little_questions.classifiers import QuestionClassifier, \
    SimpleQuestionClassifier


class TreeQuestionClassifier(QuestionClassifier):
    def __init__(self, name="tree"):
        super().__init__(name)

    @property
    def classifier_class(self):
        return DecisionTreeClassifier()


class SimpleTreeQuestionClassifier(TreeQuestionClassifier,
                                   SimpleQuestionClassifier):
    def __init__(self, name="tree_main"):
        super().__init__(name)


class ExtraTreeQuestionClassifier(QuestionClassifier):
    def __init__(self, name="extra_tree"):
        super().__init__(name)

    @property
    def classifier_class(self):
        return ExtraTreeClassifier()


class SimpleExtraTreeQuestionClassifier(ExtraTreeQuestionClassifier,
                                        SimpleQuestionClassifier):
    def __init__(self, name="extra_tree_main"):
        super().__init__(name)


if __name__ == '__main__':
    train = True
    clf = SimpleExtraTreeQuestionClassifier()
    if train:
        t, tt = clf.load_data()
        clf.train(t, tt)
        clf.save()
    else:
        clf.load()
    clf.evaluate_model()  # Accuracy: 0.7
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
