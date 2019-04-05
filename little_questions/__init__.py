from little_questions.settings import DEFAULT_CLASSIFIER, MODELS_PATH,\
    AFFIRMATIONS
from little_questions.parsers import BasicQuestionParser, SlotParser, \
    QuestionChunker
from little_questions.parsers.neural import NeuralQuestionParser
from little_questions.classifiers import QuestionClassifier, SentenceClassifier


class Command(object):
    sentence_classifier = SentenceClassifier()

    def __init__(self, text):
        self._text = text
        self._sent_classification = self.sentence_classifier.predict([text])[0]

    @property
    def text(self):
        return self._text

    @property
    def is_statement(self):
        return not self.is_command and not self.is_question

    @property
    def is_command(self):
        # TODO detect imperative
        return self._sent_classification == "command"

    @property
    def is_question(self):
        valid_starts = ["how", "why", "when", "who", "where", "which", "what",
                        "whose"] + AFFIRMATIONS
        if self.text.split(" ")[0].lower().strip() in valid_starts:
            return True
        starts = ["in what ", "on what ", "at what ", "in which"]
        for s in starts:
            if self.text.lower().startswith(s):
                return True
        ends = [" in what", " on what", " for what", " as what", "?"]
        for s in ends:
            if self.text.lower().endswith(s):
                return True
        return self._sent_classification == "question"

    @property
    def sentence_type(self):
        if self.is_question:
            return "question"
        if self.is_command:
            return "command"
        if self.is_statement:
            return "statement"
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


class Question(Command):
    parser = NeuralQuestionParser()
    question_classifier = QuestionClassifier()

    def __init__(self, text):
        super().__init__(text)
        self._answers = []
        self._parsed = None
        self._classification = self.question_classifier.predict([text])[0]

    @property
    def topics(self):
        slots = super().topics
        if self.main_type == "HUM":
            slots["person"] = True
        elif self.main_type == "ENTY":
            slots["entity"] = True
        elif self.main_type == "NUM":
            if self.secondary_type in ["date", "period"]:
                slots["date"] = True
            else:
                slots["quantity"] = True
        elif self.main_type == "LOC":
            slots["location"] = True
        elif self.secondary_type in ["speed", "dist", "temp", "volsize"]:
            slots["property"] = True
        else:
            slots["thing"] = True
        return slots

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
        if question.is_question:
            continue
        print("Q:", q)
        # print("Intent:", question.intent_data["QuestionIntent"])
        # pprint(question.entities)
        # print("SUB_QUESTIONS:")
        # pprint(question.sub_steps)
        print(question.sentence_type)
        pprint(question.topics)
        #print( question.pretty_label)
        print("____")
