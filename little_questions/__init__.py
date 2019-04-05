from little_questions.settings import DEFAULT_CLASSIFIER, MODELS_PATH, \
    AFFIRMATIONS
from little_questions.parsers import BasicQuestionParser, SlotParser, \
    QuestionChunker, SentenceScorer
from little_questions.parsers.neural import NeuralQuestionParser
from little_questions.classifiers import QuestionClassifier
from nltk import word_tokenize, pos_tag


class Sentence(object):
    sentence_classifier = SentenceScorer()

    def __init__(self, text):
        self._text = text
        self._sent_classification = self.sentence_classifier.predict(text)
        if self._sent_classification == "command":
            self.__class__ = Command
        elif self._sent_classification == "question":
            self.__class__ = Question
        elif self._sent_classification == "exclamation":
            self.__class__ = Exclamation
        elif self._sent_classification == "statement":
            self.__class__ = Statement
        elif self._sent_classification == "request":
            self.__class__ = Request

    @property
    def text(self):
        return self._text

    @property
    def score(self):
        return self.sentence_classifier.score(self.text)

    @property
    def pos_tag(self):
        return pos_tag(word_tokenize(self.text))

    @property
    def is_exclamation(self):
        return isinstance(self, Exclamation)

    @property
    def is_request(self):
        return isinstance(self, Request)

    @property
    def is_statement(self):
        return isinstance(self, Statement)

    @property
    def is_command(self):
        return isinstance(self, Command)

    @property
    def is_question(self):
        return isinstance(self, Question)

    @property
    def sentence_type(self):
        return self._sent_classification

    @property
    def topics(self):
        return SlotParser.parse_topics(self.text)

    @property
    def entities(self):
        return SlotParser.parse_slots(self.text)

    @property
    def sub_steps(self):
        return QuestionChunker.decompose_question(self.text)

    def __str__(self):
        return self.text


class Question(Sentence):
    parser = NeuralQuestionParser()
    question_classifier = QuestionClassifier()

    def __init__(self, text):
        super().__init__(text)
        self._answers = []
        self._parsed = None
        self._classification = self.question_classifier.predict([text])[0]

    @property
    def topics(self):
        topcs = super().topics
        if self.main_type == "HUM":
            topcs["person"] = True
        elif self.main_type == "ENTY":
            topcs["entity"] = True
        elif self.main_type == "NUM":
            if self.secondary_type in ["date", "period"]:
                topcs["date"] = True
            else:
                topcs["quantity"] = True
        elif self.main_type == "LOC":
            topcs["location"] = True
        elif self.secondary_type in ["speed", "dist", "temp", "volsize"]:
            topcs["property"] = True
        else:
            topcs["thing"] = True
        return topcs

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
        return self._classification.split(":")[0]

    @property
    def secondary_type(self):
        return self._classification.split(":")[1]

    @property
    def pretty_label(self):
        pretty_main = self.main_type
        if self.main_type == "ENTY":
            pretty_main = "Entity"
        elif self.main_type == "DESC":
            pretty_main = "Description"
        elif self.main_type == "NUM":
            pretty_main = "Numeric"
        elif self.main_type == "HUM":
            pretty_main = "Human"
        elif self.main_type == "LOC":
            pretty_main = "Location"
        elif self.main_type == "ABBR":
            pretty_main = "Abbreviation"

        pretty_sec = self.secondary_type
        if self.secondary_type == "def":
            pretty_sec = "definition"
        elif self.secondary_type == "desc":
            pretty_sec = "description"
        elif self.secondary_type == "ind":
            pretty_sec = "individual"
        elif self.secondary_type == "dist":
            pretty_sec = "distance"
        elif self.secondary_type == "volsize":
            pretty_sec = "volume"
        elif self.secondary_type == "temp":
            pretty_sec = "temperature"
        elif self.secondary_type == "gr":
            pretty_sec = "group or organization of persons"
        elif self.secondary_type == "abb":
            pretty_sec = "abbreviation"
        elif self.secondary_type == "exp":
            pretty_sec = "expression abbreviated"
        elif self.secondary_type == "body":
            pretty_sec = "organs of body"
        elif self.secondary_type == "cremat":
            pretty_sec = "inventions, books and other creative pieces"
        elif self.secondary_type == "dismed":
            pretty_sec = "diseases and medicine"
        elif self.secondary_type == "lang":
            pretty_sec = "language"
        elif self.secondary_type == "termeq":
            pretty_sec = "equivalent terms"
        elif self.secondary_type == "veh":
            pretty_sec = "vehicles"

        return pretty_sec + " (" + pretty_main + ")"


class Command(Sentence):
    pass


class Request(Command):
    pass


class Exclamation(Sentence):
    pass


class Statement(Sentence):
    pass


if __name__ == "__main__":
    from pprint import pprint

    questions = [
        "what do dogs and cats have in common",
        "tell me about evil",
        "what is a living being",
        "how to kill animals ( a cow ) and make meat",
        "why are humans living beings",
        "give examples of animals",
        "what is the speed of light",
        "when is your birthday",
        "when were you born",
        "where do you store your data",
        "will you die",
        "should i program artificial stupidity",
        "who made you",
        "how long until world war 3",
        "how long ago was sunrise",
        "which city has more people",
        "did you know that dogs are animals",
        "do you agree that dogs are animals",
        "who made you",
        "whose dog is this",
        "how much is bitcoin worth",

        "have you finished booting",

        "how tall is the eiffel tower",
        "how big is an elephant",
        "how large is the car",
        "how fast is a zebra",

        "not a question"]

    for q in questions:
        break
        question = Question(q)
        print("Q:", q)
        # print("Intent:", question.intent_data["QuestionIntent"])
        # pprint(question.entities)
        # print("SUB_QUESTIONS:")
        # pprint(question.sub_steps)
        print(question.sentence_type, question.pretty_label)
        print("____")

    from little_questions.data import SAMPLE_QUESTIONS
    import random

    questions = SAMPLE_QUESTIONS
    random.shuffle(questions)

    for q in questions:
        question = Question(q)
        print("Q:", q)
        print("Intent:", question.intent_data["QuestionIntent"])
        pprint(question.entities)
        print("SUB_QUESTIONS:")
        pprint(question.sub_steps)
        print(question.pretty_label)
        print("____")
