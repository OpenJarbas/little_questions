from little_questions.settings import DEFAULT_CLASSIFIER, \
    DEFAULT_MAIN_CLASSIFIER, MODELS_PATH
from little_questions.classifiers.pipelines import default_pipeline
from os.path import join, isfile
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, f1_score, recall_score, precision_score
from sklearn.svm import LinearSVC
from little_questions.settings import AFFIRMATIONS, ALL_POS_TAGS
from nltk import word_tokenize, pos_tag


class LinearSVCTextClassifier:
    def __init__(self, name="linear_svc"):
        self.name = name.replace('_model.pkl', "")
        self.text_clf = None

    def train(self, train_data, target_data):
        self.text_clf = Pipeline(self.pipeline)
        self.text_clf.fit(train_data, target_data)
        return self.text_clf

    @property
    def pipeline(self):
        return [
            ('features', default_pipeline),
            ('clf', LinearSVC())
        ]

    def predict(self, text):
        return self.text_clf.predict(text)

    def save(self, path):
        joblib.dump(self.text_clf, path)

    def load_from_file(self, path):
        self.text_clf = joblib.load(path)
        return self

    def load_data(self, filename):
        raise NotImplementedError
        return train_data, target_data

    def load_test_data(self, filename):
        return self.load_data(filename)

    def evaluate_model(self, path):
        X_test, y_test = self.load_test_data(path)
        preds = self.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        matrix = confusion_matrix(y_test, preds)
        return accuracy, report, matrix


class QuestionClassifier(LinearSVCTextClassifier):
    def __init__(self, name=DEFAULT_CLASSIFIER):
        super().__init__(name)
        if isfile(join(MODELS_PATH, name)):
            self.load_from_file(join(MODELS_PATH, name))

    def load_data(self, filename):
        train_data = []
        target_data = []
        with open(filename, 'r') as f:
            for line in f:
                label = line.split(" ")[0]
                question = " ".join(line.split(" ")[1:])
                train_data.append(question.strip())
                target_data.append(label.strip())
        return train_data, target_data


class MainQuestionClassifier(QuestionClassifier):
    def __init__(self, name=DEFAULT_MAIN_CLASSIFIER):
        super().__init__(name)

    def load_data(self, filename):
        train_data = []
        target_data = []
        with open(filename, 'r') as f:
            for line in f:
                label = line.split(" ")[0]
                question = " ".join(line.split(" ")[1:])
                train_data.append(question.strip())
                target_data.append(label.strip().split(":")[0])  # main
        return train_data, target_data


# no good dataset for training, so this will work for now....
class SentenceScorer:
    @staticmethod
    def predict(text):
        score = SentenceScorer.score(text)
        best = max(score, key=lambda key: score[key])
        return best

    @staticmethod
    def score(text):
        return {
            "question": SentenceScorer.question_score(text),
            "statement": SentenceScorer.statement_score(text),
            "exclamation": SentenceScorer.exclamation_score(text),
            "command": SentenceScorer.command_score(text),
            "request": SentenceScorer.request_score(text)
        }

    @staticmethod
    def _score(text, last_tokens=None, first_tokens=None,
               start_pos_tags=None, end_pos_tags=None, unlikely_words=None,
               unlikely_start_pos_tag=None, unlikely_end_pos_tag=None,
               unlikely_pos_tag=None):
        score = 8
        last_tokens = last_tokens or []
        first_tokens = first_tokens or []
        start_pos_tags = start_pos_tags or []
        end_pos_tags = end_pos_tags or []
        unlikely_words = unlikely_words or []
        unlikely_start_pos_tag = unlikely_start_pos_tag or [t for t in
                                                            ALL_POS_TAGS if
                                                            t not in
                                                            start_pos_tags]
        unlikely_end_pos_tag = unlikely_end_pos_tag or []
        unlikely_pos_tag = unlikely_pos_tag or unlikely_end_pos_tag + unlikely_start_pos_tag
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        if not tokens[-1] in last_tokens:
            score -= 1
        if not tokens[0] in first_tokens:
            score -= 1
        if not tagged[0][1] in start_pos_tags:
            score -= 1
        if not tagged[-1][1] in end_pos_tags:
            score -= 1

        if tagged[0][1] in unlikely_start_pos_tag:
            score -= 1
        if tagged[-1][1] in unlikely_end_pos_tag:
            score -= 1
        for pos in tagged:
            if pos[1] in unlikely_pos_tag:
                score -= 0.1
        for tok in tokens:
            if tok in unlikely_words:
                score -= 0.2
        if score <= 0:
            return 0
        return max(score / 7, 0)

    @staticmethod
    def question_score(text):
        """
        Questions can have two patterns. Some can have ‘yes’ or ‘no’ as an answer.
        For example,
            Do you like Paris?,
            Can you speak Russian?
            Will you marry me?

        Alternatively, they have a pattern that asks an ‘open’ question
        which can have any number of answers, e.g.
            What did you have for breakfast?
            Which newspaper do you read?
            Who is your favourite actor?
        """
        # ends with a question mark
        last_tokens = ["?"]

        # starts with a question word
        first_tokens = ["what", "why", "how", "when", "who", "whose",
                        "which"] + AFFIRMATIONS

        # WDT wh-determiner which
        # WP wh-pronoun who, what
        # WP$ possessive wh-pronoun whose
        # WRB wh-abverb where, when
        start_pos_tags = ["WDT", "WP", "WP$", "WRB", "VB", "VBP", "VBZ",
                          "VBN", "VBG"]

        # pos_tags more likely to be in the end of question
        # questions likely end with a question mark or
        # Noun/Pronoun/Adjective, usually not verbs
        end_pos_tags = [".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ"]

        # all pos tags are unlikely except W*
        unlikely_words = ["!"]
        unlikely_start_pos_tag = [t for t in ALL_POS_TAGS if t not in
                                  start_pos_tags]
        unlikely_end_pos_tag = []
        unlikely_pos_tag = []
        score = SentenceScorer._score(text, last_tokens, first_tokens,
                                      start_pos_tags,
                                      end_pos_tags, unlikely_words,
                                      unlikely_start_pos_tag,
                                      unlikely_end_pos_tag,
                                      unlikely_pos_tag)

        starts = ["in what ", "on what ", "at what ", "in which"]
        for s in starts:
            if text.lower().startswith(s):
                score += 0.1
                break
        # end of sentence
        ends = [" in what", " on what", " for what", " as what", "?"]
        for s in ends:
            if text.lower().endswith(s):
                score += 0.1
                break
        return min(score, 1)

    @staticmethod
    def statement_score(text):
        """
        A statement is defined as having a structure in which there is typically a Subject,
        followed by a verb and then a further unit such as a Direct Object.
        For example,
            Jimmy loves his dog,
            The government will make an announcement at noon,
            She reads two newspapers every day

        """
        last_tokens = ["."]

        # statements often start with "our X", "the X", "we/i X"
        first_tokens = ["we", "i", "the", "our"]

        # Our dog eats any old thing.
        # The dog has already been fed.
        # We have already won several races.
        start_pos_tags = ["PRP", "PRP$", "DT"]

        # Noun/Pronoun/Adjective, usually not verbs
        end_pos_tags = [".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ"]

        unlikely_words = ["what", "why", "how", "when", "who", "whose",
                          "which", "?", "!"]
        unlikely_start_pos_tag = ["WDT", "WP", "WP$", "WRB", "VB",
                                  "VBP", "VBZ", "VBN", "VBG"]
        unlikely_end_pos_tag = []
        unlikely_pos_tag = ["WDT", "WP", "WP$", "WRB"]  # "what"
        score = SentenceScorer._score(text, last_tokens, first_tokens,
                                      start_pos_tags,
                                      end_pos_tags, unlikely_words,
                                      unlikely_start_pos_tag,
                                      unlikely_end_pos_tag,
                                      unlikely_pos_tag)

        # TODO An important feature of declarative sentences is that they
        # have a subject that comes before the verb.

        if text.lower().startswith("be "):
            score -= 0.1
        # random bias, helps disambiguate
        return max(min(score + 0.0001, 1), 0)

    @staticmethod
    def exclamation_score(text):
        """
        Exclamations grammatically have a structure that involves the words what a or how,

            What a nice person you are!
            What a beautiful painting!,
            How clever you are!,
            How wonderful that is!

        (Notice that the Subject goes before the verb in How clever you are!
        If this were a question we would have How clever are you?)

        """
        # often ends with an exclamation mark
        last_tokens = ["!"]

        # Exclamations grammatically have a structure that involves the words what or how,
        first_tokens = ["how", "what"]

        start_pos_tags = ["WP", "WRB"]

        # Noun/Pronoun/Adjective, usually also verbs
        end_pos_tags = [".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ", "VB",
                        "VBP", "VBZ", "VBN", "VBG"]

        # words unlikely to appear in statements,
        # if empty score is not penalized
        unlikely_words = ["why", "when", "who", "whose", "which", "?"]
        # starts with "what" and "how" only
        unlikely_start_pos_tag = ["WDT", "WP$", "NN", "NNP", "NNS", "NNPS",
                                  "DT", "PRP", "JJ", "VB", "VBP", "VBZ",
                                  "VBN", "VBG", "PDT", "RB", "RBR", "RBS"]
        unlikely_end_pos_tag = []
        unlikely_pos_tag = ["WDT", "WP$"]
        score = SentenceScorer._score(text, last_tokens, first_tokens,
                                      start_pos_tags,
                                      end_pos_tags, unlikely_words,
                                      unlikely_start_pos_tag,
                                      unlikely_end_pos_tag,
                                      unlikely_pos_tag)

        # TODO the Subject goes before the verb
        # penalize if doesn't start as expected
        if not text.split(" ")[0].lower() in first_tokens:
            score -= 0.1
        elif not text.lower().startswith("what a ") or \
                not text.lower().startswith("what an "):
            # if it only starts with "what" without "a" it's likely a question
            score -= 0.1
        # penalize if contains a question word
        for w in unlikely_words:
            if w in text.lower():
                score -= 0.05

        # compensate for ambiguous question words
        common_mistakes = ["how many", "how much", "how tall", "how fast",
                           "how big", "how often", "what is", "what are"]
        for w in common_mistakes:
            if text.lower().startswith(w):
                score -= 0.1
        return max(score, 0)

    @staticmethod
    def command_score(text):
        """
        Commands also have a special structure in that they typically lack a Subject.
        Examples are:
            Eat your dinner
            Be quiet
            Open the door, etc.

        Not all imperative sentences are orders or commands.
        They can be social expressions.
            Have a nice day.
            Get well soon.
            Help yourselves to coffee.
        """
        # might end with anything, but usually not question marks
        last_tokens = ["!", "."]

        # starts with infinitive verb
        # only cases exclusive to commands here
        # "be quiet"
        first_tokens = ["be"]

        # starts with a verb, infinitive usually
        # NOTE, adding noun because nltk mistags often
        start_pos_tags = ["VB", "NN", "NNP"]

        # usually not verbs
        end_pos_tags = [".", "NN", "NNP", "NNS", "NNPS", "PRP"]

        # penalize if question words
        unlikely_words = ["what", "why", "how", "when", "who", "whose",
                          "which", "?"]
        unlikely_start_pos_tag = ["WDT", "WP", "WP$", "WRB", "JJ"]
        unlikely_end_pos_tag = ["VB", "VBP", "VBZ", "VBN", "VBG"]
        unlikely_pos_tag = ["WDT", "WP", "WP$", "WRB"]
        score = SentenceScorer._score(text, last_tokens, first_tokens,
                                      start_pos_tags,
                                      end_pos_tags, unlikely_words,
                                      unlikely_start_pos_tag,
                                      unlikely_end_pos_tag,
                                      unlikely_pos_tag)

        # "do" can be part of a question,
        #    "do you believe in god?" or a command
        #    "do your homework"
        # common mistakes in test data, feel free to add more, but verbs can
        #  be anything
        # "Name the prime minister", "Define evil"
        starts = ["do the ", "do your ", "name", "define"]
        for s in starts:
            if text.lower().startswith(s):
                score += 0.1
                break
        return min(score, 1)

    @staticmethod
    def request_score(text):
        """
        We can make a request, which is a type of command,
        sound more polite by using the interrogative.
            Would you feed the dog, please.
            Would you mind shutting the door.
            Could I have that now, thank you.
        """
        # requests are usually neutral, no ? or !
        last_tokens = [".", "?"]

        # starts with would or could
        # "could you pass me the salt"
        first_tokens = ["would", "could", "can"]

        start_pos_tags = ["MD"]

        # Noun/Pronoun/Adjective, usually not verbs
        end_pos_tags = [".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ"]

        unlikely_words = ["!", "?"]
        unlikely_start_pos_tag = [t for t in ALL_POS_TAGS if t not in
                                  start_pos_tags]
        unlikely_end_pos_tag = []
        unlikely_pos_tag = []
        score = SentenceScorer._score(text, last_tokens, first_tokens,
                                      start_pos_tags,
                                      end_pos_tags, unlikely_words,
                                      unlikely_start_pos_tag,
                                      unlikely_end_pos_tag,
                                      unlikely_pos_tag)
        # TODO An important feature of interrogative sentences is that they
        # normally have a subject that comes after an auxiliary verb.

        # if it contains please add a bias
        if "please" in text:
            score += 0.3
        # if it starts with an unexpected word add a penalty
        if not text.split(" ")[0].lower() in first_tokens:
            score -= 0.2
        return max(min(score, 1), 0)
