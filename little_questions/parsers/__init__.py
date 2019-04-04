from little_questions.settings import AFFIRMATIONS
from simple_NER.annotators.date import DateTimeNER
from simple_NER.annotators.nltk_ner import NltkNER
from little_questions.utils import normalize
from nltk import word_tokenize, pos_tag


class BasicQuestionParser(object):
    """
    Poor-man's english question parser. Not even close to conclusive, but
    appears to construct some decent w|a queries and responses.

    """

    def __init__(self, lang="en-us"):
        self.lang = lang

    def chunk_question(self, sent):
        sent = self.normalize(sent)
        tagged = pos_tag(word_tokenize(sent))
        #pprint(tagged)
        bucket = []

        # extract question type
        for word, pos in tagged:

            if pos.startswith("W"):
                subquestion = {
                    "subintent": "confirm",
                    "subquestion": "question type is " + word,
                    "object": word,
                    "subject": "question",
                    "action": "information retrieval"
                }
                #bucket.append(subquestion)
        # extract quantities
        qnt = ""
        current_chunk = []
        for word, pos in tagged:
            if pos in ["DT", "POS", ".", "IN", "JJ"]:
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
        # normalize entities
        # what -> thing
        for idx, w in enumerate(tagged):
            if w[1].startswith("W"):
                tagged[idx] = (tagged[idx][0], "NN")
                if w[0].lower() == "when":
                    tagged[idx] = ("date", tagged[idx][1])
                elif w[0].lower().startswith("who"):
                    tagged[idx] = ("person", tagged[idx][1])
                elif w[0].lower().startswith("where"):
                    tagged[idx] = ("location", tagged[idx][1])
                #else:
                #    tagged[idx] = ("thing", tagged[idx][1])
        # I -> user
        for idx, w in enumerate(tagged):
            if w[1] == "PRP" and w[0].lower() == "i":
                tagged[idx] = ("user", "NN")
        # you -> self

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

        # extract actions
        pre_chunk = []
        current_chunk = []
        adjective = ""
        for word, pos in tagged:
            if pos in ["DT", "POS", "."]:
                continue
            if pos.startswith("V") or pos.startswith("IN"):

                if pre_chunk and current_chunk:
                    n1 = " ".join(pre_chunk)
                    n2 = " ".join(current_chunk)
                    subquestion = {
                        "subintent": "select option",
                        "subquestion": " ".join(
                            ["select", n1, "with value", adjective, n2]),
                        "action": adjective,
                        "subject": n1,
                        "object": n2
                    }
                    bucket += [subquestion]
                pre_chunk = current_chunk
                current_chunk = []
                if pos.startswith("V"):
                    adjective = word
            elif pos.startswith("N"):
                current_chunk += [word]

        if pre_chunk and current_chunk:
            n1 = " ".join(pre_chunk)
            n2 = " ".join(current_chunk)
            subquestion = {
                "subintent": "select option",
                "subquestion": " ".join(
                    ["select", n1, "with value", adjective, n2]),
                "action": adjective,
                "subject": n1,
                "object": n2
            }
            bucket += [subquestion]

        for idx, q in enumerate(bucket):
            # lemmatize
            for k in ["subquestion", "action"]:
                if k in bucket[idx]:
                    bucket[idx][k] = normalize(q[k])

            # corner cases cleannup
            if bucket[idx]["subquestion"].startswith("select how with value "):
                bucket[idx]["subquestion"] = bucket[idx][
                    "subquestion"].replace("select how with value ",
                                           "select explanation for how ")
            if bucket[idx]["subquestion"].startswith("select what with value "):
                bucket[idx]["subquestion"] = bucket[idx][
                    "subquestion"].replace("select what with value ",
                                           "select option with value ")

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
                        "subquestion": "question type is select " + bucket[idx]["subject"],
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

    def ner(self, text):
        ents = []
        try:
            for ent in DateTimeNER().extract_entities(text, as_json=True):
                ents.append(ent)
        except IndexError:
            # TODO fix in simple_NER
            pass
        except ValueError:
            # ValueError: time data 'march' does not match format '%B %d %Y'
            # TODO fix in simple_NER
            pass
        for ent in NltkNER().extract_entities(text, as_json=True):
            ents.append(ent)
        return ents

    def normalize(self, text):
        # pos parsing normalization
        text = text.replace(" 's", "'s").replace("''", "") \
            .replace("``", "").strip()
        return text.lower()

    def parse(self, utterance):
        # normalization pre-parsing
        utterance = self.normalize(str(utterance))
        subs = self.chunk_question(utterance)
        ents = self.ner(utterance)
        data = {
            "Question": utterance,
            "QuestionIntent": "unknown",
            "about": {
                "entity": False,
                "location": False,
                "person": False,
                "date": False,
                "quantity": False,
                "property": False,
            },
            "slots": {
                "entity": [],
                "thing": [],
                "location": [],
                "person": [],
                "date": [],
                "quantity": [],
                "property": [],
            },
            "is_affirmation": utterance.split(" ")[0].lower() in AFFIRMATIONS,
            "is_wh": utterance.lower().startswith("wh") or
                     utterance.lower().startswith("how ")}
        # entity extraction
        for e in ents:
            if e["entity_type"] == "GPE":
                data["about"]["location"] = True
                data["QuestionIntent"] = "place"
                data["slots"]["location"] += [e["value"]]
            elif e["entity_type"] == "ORGANIZATION":
                data["about"]["entity"] = True
                data["slots"]["thing"] += [e["value"]]
                data["slots"]["entity"] += [e["value"]]
            elif e["entity_type"] == "PERSON":
                data["about"]["person"] = True
                data["slots"]["person"] += [e["value"]]
                data["slots"]["thing"] += [e["value"]]
                data["slots"]["entity"] += [e["value"]]
            else:
                name = e["entity_type"].lower()
                data["about"][name] = True
                if name not in data["slots"]:
                    data["slots"][name] = []
                data["slots"][name] += [e["value"]]

        # yes no question
        if data["is_affirmation"]:
            data["QuestionIntent"] = "confirm"
            if data["about"]["location"]:
                data["QuestionIntent"] = "confirm_location"
        # question
        if data["is_wh"]:
            data["QuestionIntent"] = "retrieve_information"


            utterance = utterance.lower()
            # when questions
            if utterance.startswith("when"):
                data["about"]["date"] = True
                if data["about"]["person"]:
                    data["QuestionIntent"] = "relate_time_and_entity"
                elif data["about"]["entity"]:
                    data["QuestionIntent"] = "relate_time_and_thing"
                elif data["about"]["location"]:
                    data["QuestionIntent"] = "relate_time_and_place"
                else:
                    data["QuestionIntent"] = "time"
                if len(subs):
                    if subs[-1]["subject"] == "date":
                        data["QuestionIntent"] = subs[-1]["subintent"]
                        data["slots"]["thing"] += [subs[-1]["object"]]
                data["slots"]["date"] = ["?"]
            # who questions
            elif utterance.startswith("who"):
                data["about"]["person"] = True
                data["QuestionIntent"] = "assign_entity"
                if len(subs):
                    if subs[-1]["subject"] == "person":
                        data["slots"]["thing"] += [subs[-1]["object"]]
                data["slots"]["person"] = ["?"]
            # quantity questions
            elif utterance.startswith("how many"):
                data["QuestionIntent"] = "quantity"
                data["slots"]["quantity"] = ["?"]
            # explanation questions
            elif utterance.startswith("how") or utterance.startswith("why"):
                data["QuestionIntent"] = "explain"
            # select option questions
            elif utterance.startswith("which"):
                data["QuestionIntent"] = "select_option"
            # location questions
            elif utterance.startswith("where"):
                data["QuestionIntent"] = "place"
                if len(subs):
                    if subs[-1]["subject"] == "location":
                        data["about"]["location"] = True
                        data["slots"]["thing"] += [subs[-1]["object"]]
                data["slots"]["location"] = ["?"]
            # what questions
            if utterance.startswith("what"):
                if data["about"]["person"]:
                    if data["about"]["location"]:
                        data["QuestionIntent"] = "relate_place_and_entity"
                    else:
                        data["QuestionIntent"] = "describe_attribute"

        # fallback rules
        if data["QuestionIntent"] == "unknown":
            if data["about"]["person"]:
                data["QuestionIntent"] = "assign_entity"
            elif data["about"]["location"]:
                data["QuestionIntent"] = "place"
            elif len(subs):
                data["QuestionIntent"] = subs[-1]["subintent"]
        # slot filling
        for sub in subs:
            if sub.get("action", "") == "has_property":
                data["slots"]["property"] += [sub["object"]]
            if sub.get("subject", ""):
                if not sub["subject"] in data["slots"]["property"]:
                    data["slots"]["thing"] += [sub["subject"]]
            if sub.get("object", ""):
                if not sub["object"] in data["slots"]["property"]:
                    data["slots"]["thing"] += [sub["object"]]

        # cleanup
        cleaned = {"slots": {}}
        for s in data["slots"]:
            # empty strings and reundant slot_names
            data["slots"][s] = [a for a in data["slots"][s] if a and a != s]
            if s not in cleaned["slots"]:
                cleaned["slots"][s] = []
            for a in data["slots"][s]:
                if a not in cleaned["slots"][s]:
                    cleaned["slots"][s] += [a]
        data["slots"] = cleaned["slots"]
        return data


if __name__ == "__main__":
    from pprint import pprint
    from little_questions.data import SAMPLE_QUESTIONS
    import random

    parser = BasicQuestionParser()

    questions = [
        "Who was the first English circumnavigator of the globe",
        "When was Rosa Parks born",
        "Where is the Kalahari desert",
        "Who was president in 1913",
        "What country has the highest arson rate",
        "What city in Florida is Sea World in",
        "What is Portugal most famous for"
    ]
    for q in questions:
        data = parser.parse(q)
        print("Q:", q)
        print("Intent:", data['QuestionIntent'])
        pprint(data)
        loc = parser.chunk_question(q)
        print("Subquestions:")
        pprint(loc)
        print("___")

    questions = SAMPLE_QUESTIONS
    random.shuffle(questions)

    for q in questions[:5]:
        data = parser.parse(q)
        print("Q:", q)
        print("Intent:", data['QuestionIntent'])
        pprint(data)
        loc = parser.chunk_question(q)
        print("Subquestions")
        pprint(loc)
        print("___")
