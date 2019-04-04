from little_questions.settings import DEFAULT_CLASSIFIER, MODELS_PATH
from little_questions.parsers import BasicQuestionParser
from little_questions.classifiers import QuestionClassifier
import random
from os.path import join


class Question(object):
    parser = BasicQuestionParser()
    #classifier = QuestionClassifier()

    def __init__(self, text):
        self.text = text
        self._answers = []
        self._parsed = None
        #self._classification = self.classifier.predict([text])[0]

    @property
    def subquestions(self):
        return self.parser.chunk_question(self.text)

    @property
    def intent_data(self):
        return self.calc_intent(True, True)

    def calc_intent(self, use_cached=False, cache=True):
        if use_cached and self._parsed is not None:
            return self._parsed
        s_feature = self.parser.parse(self.text)
        if cache:
            self._parsed = s_feature
        return s_feature

    @property
    def main_type(self):
        return #self._classification.split(":")[0]

    @property
    def secondary_type(self):
        return #self._classification.split(":")[1]

    def add_answer(self, answer):
        if answer not in self._answers:
            self._answers.append(answer)

    @property
    def answer(self):
        if not len(self._answers):
            return None
        return random.choice(self._answers)


if __name__ == "__main__":
    from pprint import pprint

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
                 "not a question",
                 "who made you"]

    for q in questions:
        question = Question(q)
        print("Q:", q)
        print("Intent:", question.intent_data["QuestionIntent"])
        pprint(question.intent_data["slots"])
        print("SUB_QUESTIONS:")
        pprint(question.subquestions)
        #print(question.main_type, question.secondary_type)
