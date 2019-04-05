from simple_NER.annotators.date import DateTimeNER
from simple_NER.annotators.nltk_ner import NltkNER
from simple_NER.annotators.keyword_ner import KeywordNER
from little_questions.utils import normalize
from little_questions.settings import AFFIRMATIONS, ALL_POS_TAGS
from nltk import word_tokenize, pos_tag


class SentenceScorer(object):
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


class SlotParser(object):
    @staticmethod
    def ner(text):
        ents = []
        for ent in DateTimeNER().extract_entities(text, as_json=True):
            ents.append(ent)
        for ent in NltkNER().extract_entities(text, as_json=True):
            ents.append(ent)
        for ent in KeywordNER().extract_entities(text, as_json=True):
            ents.append(ent)
        return ents

    @staticmethod
    def parse_slots(text):
        slots = {
            "entity": [],
            "thing": [],
            "location": [],
            "person": [],
            "date": [],
            "quantity": [],
            "property": [],
        }
        for e in SlotParser.ner(text):
            if e["entity_type"] == "GPE":
                slots["location"] += [e["value"]]
            elif e["entity_type"] == "ORGANIZATION":
                slots["entity"] += [e["value"]]
            elif e["entity_type"] == "PERSON":
                slots["person"] += [e["value"]]
                slots["entity"] += [e["value"]]
            elif e["entity_type"] == "keyword":
                slots["thing"] += [e["value"]]
            elif e["entity_type"] == "relative_date" or e["entity_type"] == \
                    "date":
                slots["date"] += [e["value"]]
            else:
                name = e["entity_type"].lower()
                if name not in slots:
                    slots[name] = []
                    slots[name] += [e["value"]]
        return slots

    @staticmethod
    def parse_topics(text):
        slots = SlotParser.parse_slots(text)
        about = {}
        for s in slots:
            about[s] = len(slots[s]) > 0
        return about


class QuestionChunker(object):

    @staticmethod
    def extract_question_type(sent):
        # extract question type
        tagged = pos_tag(word_tokenize(sent))
        bucket = []
        for word, pos in tagged:

            if pos.startswith("W"):
                subquestion = {
                    "subintent": "confirm",
                    "subquestion": "question type is " + word,
                    "object": word,
                    "subject": "question",
                    "action": "information retrieval"
                }
                bucket.append(subquestion)
        return bucket

    @staticmethod
    def extract_quantities(sent):
        tagged = pos_tag(word_tokenize(sent))
        bucket = []
        # extract quantities
        qnt = ""
        current_chunk = []
        for word, pos in tagged:
            if pos in ["DT", "POS", ".", "JJ"]:
                continue
            if pos == "CD":
                qnt = word
                continue
            if pos.startswith("N"):
                current_chunk += [word]
            else:
                if qnt and current_chunk:
                    word = " ".join(current_chunk)
                    subquestion = {
                        "subintent": "confirm",
                        "subquestion": word + " has value " + qnt,
                        "object": qnt,
                        "subject": word,
                        "action": "has value"
                    }
                    bucket.append(subquestion)
                    qnt = ""
                if pos.startswith("V") or pos == "IN":
                    current_chunk = []
        if qnt and current_chunk:
            word = " ".join(current_chunk)
            subquestion = {
                "subintent": "confirm",
                "subquestion": word + " has value " + qnt,
                "object": qnt,
                "subject": word,
                "action": "has value"
            }
            bucket.append(subquestion)
        return bucket

    @staticmethod
    def extract_relations(sent):
        tagged = pos_tag(word_tokenize(sent))
        bucket = []
        # extract relations
        pre_chunk = []
        current_chunk = []
        q = "what"
        v = "equivalent to"
        for word, pos in tagged:
            if pos.startswith("W"):
                q = word
                continue
            if pos.startswith("V"):
                v = word
                continue
            if pos in ["DT", "POS", "."]:
                continue

            elif pos == "IN":

                if pre_chunk:
                    n1 = " ".join(pre_chunk)
                    n2 = " ".join(current_chunk)
                    subquestion = {
                        "subintent": "relate to thing",
                        "subquestion": " ".join(["relate", n1, "to", n2]),
                        "subject": n1,
                        "object": n2
                    }
                    if q == "how":
                        subquestion["subintent"] = "step_by_step"
                        subquestion["action"] = v

                    bucket += [subquestion]
                pre_chunk = current_chunk
                current_chunk = []
            elif pos.startswith("N"):
                current_chunk += [word]

        if pre_chunk and current_chunk:
            n1 = " ".join(pre_chunk)
            n2 = " ".join(current_chunk)
            subquestion = {
                "subintent": "relate to thing",
                "subquestion": " ".join(["relate", n1, "to", n2]),
                "subject": n1,
                "object": n2
            }
            bucket += [subquestion]
        return bucket

    @staticmethod
    def extract_adjectives(sent):
        tagged = pos_tag(word_tokenize(sent))
        bucket = []
        # extract properties
        current_chunk = []
        adjective = ""
        for word, pos in tagged:
            if pos in ["DT", "POS", "."]:
                continue
            elif pos.startswith("J"):
                adjective = word
                continue
            elif pos.startswith("N"):
                current_chunk += [word]
            elif adjective:
                n = " ".join(current_chunk)
                subquestion = {
                    "subintent": "confirm",
                    "subquestion": " ".join(
                        ["confirm that", n, "has property", adjective]),
                    "action": "has_property",
                    "subject": n,
                    "object": adjective
                }
                bucket += [subquestion]
                current_chunk = []
                adjective = ""
        if adjective and current_chunk:
            n = " ".join(current_chunk)
            subquestion = {
                "subintent": "confirm",
                "subquestion": " ".join(
                    ["confirm that", n, "has property", adjective]),
                "action": "has_property",
                "subject": n,
                "object": adjective
            }
            bucket += [subquestion]

        return bucket

    @staticmethod
    def extract_actions(sent):
        tagged = pos_tag(word_tokenize(sent))
        tagged = QuestionChunker._normalize_entities(tagged)

        bucket = []
        # extract actions
        pre_chunk = []
        current_chunk = []
        action = ""
        for word, pos in tagged:
            if pos in ["DT", "POS", "."]:
                continue
            if pos.startswith("W"):
                pos = "NN"
                word = "?"
            if pos.startswith("V") or pos.startswith("IN"):

                if pre_chunk and current_chunk:
                    n1 = " ".join(pre_chunk)
                    n2 = " ".join(current_chunk)
                    if n2.endswith(" and"):
                        n2 = n2[:-4]
                    if n1.endswith(" and"):
                        n1 = n1[:-4]
                    subquestion = {
                        "subintent": "select option",
                        "subquestion": " ".join(
                            ["select", n1, " for ", action, n2]),
                        "action": action,
                        "subject": n1,
                        "object": n2
                    }
                    bucket += [subquestion]

                pre_chunk = current_chunk
                current_chunk = []
                if pos.startswith("V"):
                    action = word
            elif pos.startswith("N") or pos == "CC" or pos.startswith("JJ"):
                current_chunk += [word]

        if "?" in pre_chunk:
            pre_chunk = ["?"]
        if pre_chunk and not current_chunk and tagged[-1][1].startswith("V"):
            current_chunk = [tagged[-1][0]]
            action = "retrieve property"

        if pre_chunk and current_chunk:
            n1 = " ".join(pre_chunk)
            n2 = " ".join(current_chunk)
            if n2.endswith(" and"):
                n2 = n2[:-4]
            if n1.endswith(" and"):
                n1 = n1[:-4]
            subquestion = {
                "subintent": "select option",
                "subquestion": " ".join(
                    ["select", n1, "for", action, n2]),
                "action": action,
                "subject": n1,
                "object": n2
            }
            if action == "retrieve property":
                subquestion["subquestion"] = " ".join([action, n2, "for", n1])
                subquestion["subintent"] = "describe_attribute"
            bucket += [subquestion]
        # parse and modify according to question type
        explains = ["how to", "why"]
        dates = ["how long ago", "how long until", "when"]
        quantities = ["how much"]
        commons = ["have in common"]

        # rename actions
        action_maps = {
            "are": "instance of",
            "is": "instance of",
            "has": "has for property"
        }
        for idx, sub in enumerate(bucket):
            if sub["action"] in action_maps:
                bucket[idx]["action"] = action_maps[sub["action"]]

        # subject to intent
        subjs = {"examples": "example",
                 "example": "example"
                 }
        for idx, sub in enumerate(bucket):
            for s in subjs:
                # Q: give examples of animals
                if sub["subject"] == s:
                    bucket[idx]["subintent"] = subjs[s]
                    bucket[idx]["subject"] = "?"
                    bucket[idx]["action"] = "select " + s

        # object to intent
        objs = {}
        for idx, sub in enumerate(bucket):
            for s in objs:
                if sub["object"] == s:
                    bucket[idx]["subintent"] = objs[s]

        # account for question types
        for idx, sub in enumerate(bucket):
            for s in commons:
                # Q: what do dogs and cats have in common
                if s in sent:
                    bucket[idx]["subintent"] = "relate_attributes"
                    bucket[idx]["action"] = "attribute union"
                    bucket[idx][
                        "subquestion"] = "list common attributes for " + sub[
                        "object"]

            if sent.startswith("who "):
                bucket[idx]["subintent"] = "assign_entity"
            elif sent.startswith("where "):
                bucket[idx]["subintent"] = "place"
            elif sent.startswith("when "):

                if sub["subintent"] == "select option":
                    if sub["action"] == "was":
                        bucket[idx]["action"] = "when"
                    if sub["subject"] == "?":
                        bucket[idx]["subquestion"] = \
                            bucket[idx]["subquestion"] \
                                .replace("select ? ", "select date ")
                        bucket[idx]["action"] = "when"
                bucket[idx]["subintent"] = "time"

        for idx, sub in enumerate(bucket):
            if sub["subintent"] == "select option":
                if sub["action"] == "has for property":
                    bucket[idx]["subquestion"] = \
                        "select " + sub["subject"] + " with property " + sub[
                            "object"]
                if sub["action"] == "instance of":
                    bucket[idx]["subquestion"] = \
                        "select " + sub["subject"] + " that are " + sub[
                            "object"]
        return bucket

    @staticmethod
    def extract_attributes(sent):
        tagged = pos_tag(word_tokenize(sent))
        bucket = []
        # extract attributes
        pre_chunk = []
        current_chunk = []
        for word, pos in tagged:
            if pos.startswith("V"):
                v = word
                continue
            if pos in ["DT", "."] or pos.startswith("W"):
                continue

            elif pos == "POS" and word == "'s":

                if pre_chunk:
                    n1 = " ".join(pre_chunk)
                    n2 = " ".join(current_chunk)
                    subquestion = {
                        "subintent": "describe_attribute",
                        "subquestion": " ".join(["retrieve", n2, "from", n1]),
                        "subject": n1,
                        "object": n2
                    }
                    bucket += [subquestion]
                pre_chunk = current_chunk
                current_chunk = []
            elif pos.startswith("N"):
                current_chunk += [word]

        if pre_chunk and current_chunk:
            n1 = " ".join(pre_chunk)
            n2 = " ".join(current_chunk)
            subquestion = {
                "subintent": "describe_attribute",
                "subquestion": " ".join(["retrieve", n2, "from", n1]),
                "subject": n1,
                "object": n2
            }
            bucket += [subquestion]

        return bucket

    @staticmethod
    def _normalize_entities(tagged):
        # normalize entities
        # what -> thing
        """
        for idx, w in enumerate(tagged):
            if w[1].startswith("W"):
                tagged[idx] = (tagged[idx][0], "NN")
                if w[0].lower() == "when":
                    tagged[idx] = ("date", tagged[idx][1])
                elif w[0].lower().startswith("who"):
                    tagged[idx] = ("person", tagged[idx][1])
                elif w[0].lower().startswith("where"):
                    tagged[idx] = ("location", tagged[idx][1])
                # else:
                #    tagged[idx] = ("thing", tagged[idx][1])
        """
        # I -> user
        for idx, w in enumerate(tagged):
            if w[1] == "PRP" and w[0].lower() == "i":
                tagged[idx] = ("user", "NN")
            elif w[1] == "NN" and w[0].lower() == "i":
                tagged[idx] = ("user", "NN")
            elif w[1] == "PRP" and w[0].lower() == "you":
                tagged[idx] = ("self", "NN")
            elif w[1] == "PRP" and w[0].lower() == "me":
                tagged[idx] = ("user", "NN")
            # elif w[1] == "PRP$" and w[0].lower() == "your":
            #    tagged[idx] = ("self", "NN")
        # you -> self
        return tagged

    @staticmethod
    def simplify_sentence(sent):
        # simplification
        sent = sent.lower()
        replaces = {"how long ago ": "when ",
                    "how long until ": "when is ",
                    "when were ": "when was ",

                    "how much ": "what value ",
                    "how many ": "what number ",
                    "tell me about ": "what is ",
                    "give examples ": "examples ",

                    "where do ": "where ",
                    "which city ": "where ",
                    "which place ": "where ",
                    "which country ": "where ",
                    "which location ": "where ",
                    "what city ": "where ",
                    "what place ": "where ",
                    "what country ": "where ",
                    "what location ": "where ",

                    "whose ": "who is the owner of ",

                    "how tall is ": "what is the height of ",
                    "how heavy is ": "what is the weight of ",
                    "how large is ": "what is the width of ",
                    "how big is ": "what is the size of ",
                    "how fast is ": "what is the speed of ",
                    "how hot is ": "what is the temperature of ",
                    "how cold is ": "what is the temperature of ",

                    }
        for r in replaces:
            sent = sent.replace(r, replaces[r])

        bad_ends = [" is this", " is that"]
        for b in bad_ends:
            if sent.endswith(b):
                sent = sent[:-len(b)]
        return sent

    @staticmethod
    def decompose_question(sent):
        sent = QuestionChunker.simplify_sentence(sent)

        # debug
        from pprint import pprint
        tagged = pos_tag(word_tokenize(sent))
        tagged = QuestionChunker._normalize_entities(tagged)
        # pprint(tagged)
        # / end debug

        bucket = []
        bucket += QuestionChunker.extract_quantities(sent)

        bucket += QuestionChunker.extract_adjectives(sent)

        bucket += QuestionChunker.extract_relations(sent)

        bucket += QuestionChunker.extract_attributes(sent)

        bucket += QuestionChunker.extract_actions(sent)

        for idx, q in enumerate(bucket):
            break
            # lemmatize
            for k in ["subquestion", "action"]:
                if k in bucket[idx]:
                    bucket[idx][k] = normalize(q[k])

            # corner cases cleannup
            if bucket[idx]["subquestion"].startswith("select how with value "):
                bucket[idx]["subquestion"] = bucket[idx][
                    "subquestion"].replace("select how with value ",
                                           "select explanation for how ")
            if bucket[idx]["subquestion"].startswith(
                    "select what with value "):
                bucket[idx]["subquestion"] = bucket[idx][
                    "subquestion"].replace("select what with value ",
                                           "select option with value ")
            if bucket[idx]["subquestion"].startswith(
                    "select why with value "):
                bucket[idx]["subquestion"] = bucket[idx][
                    "subquestion"].replace("select why with value ",
                                           "select explanation for why ")

            if "action" in bucket[idx]:
                # has X
                if bucket[idx]["action"] == "has":
                    bucket[idx]["subintent"] = "property_check"
                    bucket[idx]["subquestion"] = bucket[idx][
                        "subquestion"].replace("with value",
                                               "has property")

            if "object" in bucket[idx]:
                # how many/how much/how long
                if bucket[idx]["object"] == "many":
                    bucket[idx]["object"] = "quantity"
                    bucket[idx]["subquestion"] = bucket[idx][
                        "subquestion"].replace("has property many",
                                               "has property quantity")
                if bucket[idx]["object"] == "much":
                    bucket[idx]["object"] = "quantity"
                    bucket[idx]["subquestion"] = bucket[idx][
                        "subquestion"].replace("has property much",
                                               "has property quantity")
                if bucket[idx]["object"] == "long":
                    bucket[idx]["object"] = "quantity"
                    bucket[idx]["subquestion"] = bucket[idx][
                        "subquestion"].replace("has property long",
                                               "has property quantity")
                # workaround which X == Name
                if bucket[idx]["object"].lower() == "which":
                    bucket[idx] = {
                        "subintent": "confirm",
                        "subquestion": "question type is select " +
                                       bucket[idx]["subject"],
                        "object": bucket[idx]["subject"],
                        "subject": "question",
                        "action": "select option"
                    }

                # "X what" -> "X"
                if bucket[idx]["object"].lower().endswith(" what"):
                    bucket[idx]["object"] = bucket[idx]["object"][:-5]
                # "what X" -> "X"
                if bucket[idx]["object"].lower().startswith("what "):
                    bucket[idx]["object"] = bucket[idx]["object"][5:]

            if "subject" in bucket[idx]:
                # person intent
                if bucket[idx]["subject"].startswith("person "):
                    bucket[idx]["subintent"] = "assign_entity"
                    bucket[idx]["subject"] = bucket[idx]["subject"][7:]
                elif bucket[idx]["subject"] == "person":
                    bucket[idx]["subintent"] = "assign_entity"
                # date intent
                if bucket[idx]["subject"] == "date":
                    bucket[idx]["subintent"] = "relate_time_and_thing"
                # "how" -> "thing"
                if bucket[idx]["subject"].lower() == "how":
                    bucket[idx]["subject"] = "thing"

                # what difference
                if bucket[idx]["subject"].lower() == "what difference":
                    bucket[idx] = {
                        "subintent": "unique_attributes",
                        "subquestion": "differences between " +
                                       bucket[idx]["object"],
                        "object": bucket[idx]["subject"],
                        "subject": "differences",
                        "action": "select option"
                    }
                # what -> thing
                if bucket[idx]["subject"].lower() == "what":
                    bucket[idx]["subject"] = "thing"
                # "X what" -> "X"
                if bucket[idx]["subject"].lower().endswith(" what"):
                    bucket[idx]["subject"] = bucket[idx]["subject"][:-5]
                # "what X" -> "X"
                if bucket[idx]["subject"].lower().startswith("what "):
                    bucket[idx]["subject"] = bucket[idx]["subject"][5:]

        return bucket


class BasicQuestionParser(object):
    """
    Poor-man's english question parser. Not even close to conclusive, but
    appears to construct some decent w|a queries and responses.

    """

    def __init__(self, lang="en-us"):
        self.lang = lang

    def parse(self, utterance):
        data = {
            "Question": utterance,
            "QuestionIntent": "unknown",
            "is_affirmation": utterance.split(" ")[0].lower() in AFFIRMATIONS,
            "is_wh": utterance.lower().startswith("wh") or
                     utterance.lower().startswith("how ")}

        # yes no question
        if data["is_affirmation"]:
            data["QuestionIntent"] = "confirm"
        # question
        if data["is_wh"]:
            data["QuestionIntent"] = "retrieve_information"
            utterance = utterance.lower()
            # when questions
            if utterance.startswith("when"):
                data["QuestionIntent"] = "time"
            # who questions
            elif utterance.startswith("who"):
                data["QuestionIntent"] = "assign_entity"
            # quantity questions
            elif utterance.startswith("how many"):
                data["QuestionIntent"] = "quantity"
            # explanation questions
            elif utterance.startswith("how") or utterance.startswith("why"):
                data["QuestionIntent"] = "explain"
            # select option questions
            elif utterance.startswith("which"):
                data["QuestionIntent"] = "select_option"
            # location questions
            elif utterance.startswith("where"):
                data["QuestionIntent"] = "place"

        return data


if __name__ == "__main__":
    from pprint import pprint

    from little_questions.data import SAMPLE_QUESTIONS
    import random

    questions = SAMPLE_QUESTIONS
    random.shuffle(questions)

    scorer = SentenceScorer()
    states = ["Our dog eats any old thing.",
              "We have already won several races.",
              "The dog hasn’t been fed yet.",
              "I am poor",
              "pizza is awesome"]
    excls = ["What a nice person you are!",
             "What a beautiful painting!",
             "How clever you are!",
             "How wonderful that is!"]
    commands = ["Eat your dinner",
                "Be quiet",
                "Open the door",
                "Name your price",
                "Define evil"]
    requests = ["Would you feed the dog, please.",
                "Would you mind shutting the door.",
                "Could I have that now, thank you."]
    questions = questions[:5]

    for q in states:
        print("\nStatement:", q)
        print(scorer.predict(q))

    for q in commands:
        print("\nCommand:", q)
        print(scorer.predict(q))

    for q in excls:
        print("\nExclamation:", q)
        print(scorer.predict(q))

    for q in requests:
        print("\nRequest:", q)
        print(scorer.predict(q))

    print("Questions")
    for q in questions:
        print("\nQuestion:", q)
        print(scorer.predict(q))

    exit(0)

    parser = BasicQuestionParser()

    data = parser.parse("Why is the sky blue")
    assert data['QuestionIntent'] == "explain"

    data = parser.parse("When was Stephen Hawking born")
    assert data['QuestionIntent'] == "time"

    data = parser.parse("Where is the Large Hadron Collider")
    assert data['QuestionIntent'] == "place"

    data = parser.parse("Who invented quantum physics")
    assert data['QuestionIntent'] == "assign_entity"

    data = parser.parse("Can i eat this?")
    assert data['QuestionIntent'] == "confirm"

    parser = QuestionChunker()
    data = parser.decompose_question(
        "how many countries fought in world war 2")
    assert data == [{'action': 'has value',
                     'object': '2',
                     'subintent': 'confirm',
                     'subject': 'world war',
                     'subquestion': 'world war has value 2'},
                    {'object': 'world war',
                     'subintent': 'relate to thing',
                     'subject': 'number countries',
                     'subquestion': 'relate number countries to world war'}]

    parser = SlotParser()
    data = parser.parse_slots("Portugal was founded in October 5, 1910")
    assert data == {'date': ['october 5 1910'],
                    'entity': [],
                    'location': ['Portugal'],
                    'person': [],
                    'property': [],
                    'quantity': [],
                    'thing': ['founded', 'october 5', 'portugal', '1910']}
