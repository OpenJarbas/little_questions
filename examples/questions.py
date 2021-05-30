from little_questions import Question, Sentence, Statement, Command, \
    Exclamation, Request
from pprint import pprint

text = "Could you pass me the salt please?"
sentence = Sentence(text)

assert isinstance(sentence, Command)
assert isinstance(sentence, Request)

assert not sentence.is_question
assert sentence.is_command
assert not sentence.is_statement
assert sentence.is_request
assert not sentence.is_exclamation

text = "I like pizza"

sentence = Sentence(text)

assert not sentence.is_question
assert not sentence.is_command
assert sentence.is_statement
assert isinstance(sentence, Statement)

text = "Open the pod bay doors"

sentence = Sentence(text)
pprint(sentence.score)
assert not sentence.is_question
assert sentence.is_command
assert not sentence.is_statement
assert isinstance(sentence, Command)

text = "What a nice dog you have there!"
sentence = Sentence(text)
assert isinstance(sentence, Exclamation)

text = "I want you to buy bitcoin"

sentence = Sentence(text)

assert not sentence.is_question
assert sentence.is_statement


text = "who made you"
question = Question(text)

assert question.is_question
assert isinstance(question, Sentence)
assert question.pretty_label == "individual (Human)"
assert question.main_type == "HUM"
assert question.secondary_type == "ind"

text = "when will the world end"
question = Question(text)
assert question.pretty_label == "date (Numeric)"

text = "how fast can an elephant run"
question = Question(text)
assert question.pretty_label == "speed (Numeric)"

text = "why are fire trucks red"
question = Question(text)

assert question.pretty_label == "reason (Description)"

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
    question = Question(q)
    print("Q:", q)
    print(question.sentence_type, question.pretty_label)
    print("____")

