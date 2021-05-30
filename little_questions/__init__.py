from little_questions.settings import DEFAULT_CLASSIFIER, MODELS_PATH, \
    AFFIRMATIONS
from little_questions.classifiers import QuestionClassifier, SentenceScorer
from nltk import word_tokenize, pos_tag



class Sentence:
    sentence_classifier = SentenceScorer()
    question_classifier = QuestionClassifier()

    def __init__(self, text):
        self._text = text
        self._sent_classification = self.sentence_classifier.predict(text)
        self._classification = self.question_classifier.predict([text])[0]
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

    def __str__(self):
        return self.text


class Question(Sentence):
    pass


class Command(Sentence):
    pass


class Request(Command):
    pass


class Exclamation(Sentence):
    pass


class Statement(Sentence):
    pass

