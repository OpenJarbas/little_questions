from little_questions.settings import AFFIRMATIONS, nlp
from little_questions.parsers import BasicQuestionParser
from little_questions.features import featurize
from little_questions.utils.word_vectors import WordTwoVec

import random


class Question(object):
    parser = BasicQuestionParser()
    vector_model = None#WordTwoVec()

    def __init__(self, text, main_type="unknown", secondary_type="unknown"):
        self.text = text
        self._main_type = main_type
        self._secondary_type = secondary_type
        self._answers = []
        self._parsed = None

    @property
    def parse_data(self):
        return self.parse(True, True)

    def parse(self, use_cached=False, cache=True):
        if use_cached and self._parsed is not None:
            return self._parsed
        s_feature = self.parser.parse(self.text)
        if cache:
            self._parsed = s_feature
        return s_feature

    @property
    def word_vector(self):
        return self.vector_model.embed(self.text)

    @property
    def features(self):
        return featurize(self.text)

    def add_answer(self, answer):
        if answer not in self._answers:
            self._answers.append(answer)

    @property
    def main_type(self):
        return self._main_type

    @property
    def secondary_type(self):
        return self._secondary_type

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
                 "not a question"]

    for q in questions:
        question = Question(q)
        pprint(question.parse_data)
        print(question.word_vector)
