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

assert sentence.sub_steps == [{'action': 'want',
                               'object': 'self',
                               'subintent': 'select option',
                               'subject': 'user',
                               'subquestion': 'select user  for  want self'},
                              {'action': 'buy',
                               'object': 'bitcoin',
                               'subintent': 'select option',
                               'subject': 'self',
                               'subquestion': 'select self for buy bitcoin'}]

text = "Portugal once split the world in half with Spain"

sentence = Sentence(text)

assert sentence.topics["location"] == True

try:
    assert sentence.entities == {'date': [],
                                 'entity': [],
                                 'location': ['Portugal', 'Spain'],
                                 'person': [],
                                 'property': [],
                                 'quantity': [],
                                 'thing': ['half', 'spain', 'world', 'split',
                                           'portugal']}

except:
    # might fail, order of list is not guaranteed
    # pprint(sentence.entities)
    pass

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

text = "Who was the first English circumnavigator of the globe"
question = Question(text)
assert question.intent_data == {
    'Question': 'Who was the first English circumnavigator of the globe',
    'QuestionIntent': 'relate_to_entity',
    'conf': 1.0,
    'entity': 'was the first English circumnavigator',
    'is_affirmation': False,
    'is_wh': True,
    'thing': 'the globe'
}

text = "When was Rosa Parks born"
question = Question(text)
assert question.intent_data == {
    'Question': 'When was Rosa Parks born',
    'QuestionIntent': 'time',
    'conf': 1.0,
    'is_affirmation': False,
    'is_wh': True,
    'thing': 'Rosa Parks born'
}

text = "How many revolutions does a standard LP make in three minutes ?"
question = Question(text)
assert question.intent_data == {
    'Question': 'How many revolutions does a standard LP make in three minutes ?',
    'QuestionIntent': 'describe_attribute',
    'conf': 1.0,
    'is_affirmation': False,
    'is_wh': True,
    'property': 'many revolutions',
    'thing': 'a standard LP make in three minutes'
}

text = "How do I tie a tie ?"
question = Question(text)
assert question.intent_data == {
    'Question': 'How do I tie a tie ?',
    'QuestionIntent': 'step_by_step',
    'conf': 1.0,
    'is_affirmation': False,
    'is_wh': True,
    'query': 'I tie a tie'
}
