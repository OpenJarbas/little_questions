from little_questions.classifiers import get_classifier, get_scorer


class Sentence(str):

    def __new__(cls, content, model="en", scorer=None):

        # lazy load classifiers
        question_classifier = get_classifier(model_id=model)
        if scorer is None:
            if model.startswith("http"):
                # TODO naming convention to extract lang
                if "en" in model:
                    scorer = "en"
            else:
                scorer = model
        sentence_classifier = get_scorer(lang=scorer)

        # classify
        classification = question_classifier.predict([content])[0]
        sent_classification = sentence_classifier.predict(content)

        # change base class
        if sent_classification == "command":
            obj = super(Sentence, Command).__new__(Command, content)
        elif sent_classification == "question":
            obj = super(Sentence, Question).__new__(Question, content)
        elif sent_classification == "exclamation":
            obj = super(Sentence, Exclamation).__new__(Exclamation, content)
        elif sent_classification == "statement":
            obj = super(Sentence, Statement).__new__(Statement, content)
        elif sent_classification == "request":
            obj = super(Sentence, Request).__new__(Request, content)
        else:
            obj = super(Sentence, cls).__new__(cls, content)

        # add new properties
        obj.classification = classification
        obj.model = model
        obj.sentence_type = sent_classification
        obj.score = sentence_classifier.score(content)
        return obj

    def __getattribute__(self, name):
        if name in dir(str):  # only handle str methods here
            def method(self, *args, **kwargs):
                value = getattr(super(), name)(*args, **kwargs)
                # not every string method returns a str:
                if isinstance(value, str):
                    return type(self)(value)
                elif isinstance(value, list):
                    return [type(self)(i) for i in value]
                elif isinstance(value, tuple):
                    return tuple(type(self)(i) for i in value)
                else:  # dict, bool, or int
                    return value
            return method.__get__(self)  # bound method
        else:  # delegate to parent
            return super().__getattribute__(name)

    @property
    def main_label(self):
        return self.classification.split(":")[0]

    @property
    def secondary_label(self):
        return self.classification.split(":")[-1]

    @property
    def pretty_label(self):
        pretty_main = self.main_label
        if self.main_label == "ENTY":
            pretty_main = "Entity"
        elif self.main_label == "DESC":
            pretty_main = "Description"
        elif self.main_label == "NUM":
            pretty_main = "Numeric"
        elif self.main_label == "HUM":
            pretty_main = "Human"
        elif self.main_label == "LOC":
            pretty_main = "Location"
        elif self.main_label == "ABBR":
            pretty_main = "Abbreviation"

        pretty_sec = self.secondary_label
        if self.secondary_label == "def":
            pretty_sec = "definition"
        elif self.secondary_label == "desc":
            pretty_sec = "description"
        elif self.secondary_label == "ind":
            pretty_sec = "individual"
        elif self.secondary_label == "dist":
            pretty_sec = "distance"
        elif self.secondary_label == "volsize":
            pretty_sec = "volume"
        elif self.secondary_label == "temp":
            pretty_sec = "temperature"
        elif self.secondary_label == "gr":
            pretty_sec = "group or organization of persons"
        elif self.secondary_label == "abb":
            pretty_sec = "abbreviation"
        elif self.secondary_label == "exp":
            pretty_sec = "expression abbreviated"
        elif self.secondary_label == "body":
            pretty_sec = "organs of body"
        elif self.secondary_label == "cremat":
            pretty_sec = "inventions, books and other creative pieces"
        elif self.secondary_label == "dismed":
            pretty_sec = "diseases and medicine"
        elif self.secondary_label == "lang":
            pretty_sec = "language"
        elif self.secondary_label == "termeq":
            pretty_sec = "equivalent terms"
        elif self.secondary_label == "veh":
            pretty_sec = "vehicles"

        return pretty_sec + " (" + pretty_main + ")"

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

