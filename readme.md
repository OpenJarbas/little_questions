# Little Questions

Classify and Parse questions

A decent question answering system needs to know what users are talking about

- [Little Questions](#little-questions)
  * [Install](#install)
  * [Usage](#usage)
    + [Sentences](#sentences)
    + [Statements](#statements)
    + [Exclamations](#exclamations)
    + [Commands](#commands)
    + [Requests](#requests)
    + [Questions](#questions)
 
## Install

Available on pip

    pip install little_questions
    
from source

    git clone https://github.com/JarbasAl/little_questions
    cd little_questions
    pip install .


## Usage

So what is a question? A question is a sentence, but a sentence might not be a question

To understand if a sentence is a question we want to:

- classify it as question, command or statement
- extract the entities the sentence is about
- decompose it in individual actions
- classify the question according to answer type
- classify the question according to action type


### Sentences

Sentences might not be questions, in this case we probably want to send them
 to an intent parsing step in our pipeline and take an action


```python
from little_questions import Sentence

text = "I want you to buy bitcoin"

sentence = Sentence(text)

assert not sentence.is_question


```
A sentence might be about different topics, basic topic analysis and entity 
extraction is done with [simple_NER](https://github.com/JarbasAl/simple_NER)

```python
from little_questions import Sentence

text = "Portugal once split the world in half with Spain"

sentence = Sentence(text)

assert sentence.topics["location"] == True

assert sentence.entities == {'date': [],
                             'entity': [],
                             'location': ['Portugal', 'Spain'],
                             'person': [],
                             'property': [],
                             'quantity': [],
                             'thing': ['half', 'spain', 'world', 'split',
                                       'portugal']}
                                       
```

In order to act, you need to understand what the sentence is about, POS_TAG 
parsing is done to chunk the sentence in manageable actions

```python
from little_questions import Sentence

text = "I want you to buy bitcoin"

sentence = Sentence(text)

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

```

Sentence objects will change their class to the corresponding classification
 automatically, you also have access to individual classification scores
 
```python
from little_questions import Sentence, Statement, Command
text = "I like pizza"

sentence = Sentence(text)
assert isinstance(sentence, Statement)

text = "Open the pod bay doors"
assert isinstance(sentence, Command)

assert sentence.score == {'command': 0.8571428571428571,
                         'exclamation': 0.4714285714285714,
                         'question': 0.8000000000000002,
                         'request': 0.29999999999999993,
                         'statement': 0.5715285714285714}

```

### Statements

Sentences might also be statements that don't expect any action and only 
convey information

A statement is defined as having a structure in which there is typically a Subject,
followed by a verb and then a further unit such as a Direct Object.

    Jimmy loves his dog,
    The government will make an announcement at noon,
    She reads two newspapers every day
 
 
```python
from little_questions import Sentence, Statement
text = "I like Mycroft"

sentence = Sentence(text)

assert not sentence.is_question
assert not sentence.is_command
assert sentence.is_statement
assert isinstance(sentence, Statement)
```

### Exclamations

Exclamations grammatically have a structure that involves the words what a or how,

    What a nice person you are!
    What a beautiful painting!,
    How clever you are!,
    How wonderful that is!

Notice that the Subject goes before the verb in How clever you are! If this 
were a question we would have How clever are you?
        
```python
from little_questions import Sentence, Exclamation

text = "How clever you are!"
sentence = Sentence(text)

assert not sentence.is_question
assert not sentence.is_command
assert not sentence.is_statement
assert not sentence.is_request
assert sentence.is_exclamation
assert isinstance(sentence, Exclamation)
```

### Commands

Most likely you only care about questions vs commands classification

Commands also have a special structure in that they typically lack a Subject.

    Eat your dinner
    Be quiet
    Open the pod bay doors

Not all imperative sentences are orders or commands. They can be social 
expressions.

    Have a nice day.
    Get well soon.
    Help yourselves to coffee.

```python
from little_questions import Sentence, Command

text = "Do your homework"
sentence = Sentence(text)

assert not sentence.is_question
assert sentence.is_command
assert not sentence.is_statement
assert not sentence.is_request
assert not sentence.is_exclamation
assert isinstance(sentence, Command)

```

### Requests

We can make a request, which is a type of command, sound more polite by using the interrogative.

    Would you feed the dog, please.
    Would you mind shutting the door.
    Could I have that now, thank you.
    
```python
from little_questions import Sentence, Command, Request

text = "Could you pass me the salt please?"
sentence = Sentence(text)


assert isinstance(sentence, Command)
assert isinstance(sentence, Request)
assert not sentence.is_question
assert sentence.is_command
assert not sentence.is_statement
assert sentence.is_request
assert not sentence.is_exclamation

```
     

### Questions

Questions can have two patterns. Some can have ‘yes’ or ‘no’ as an answer.
    
    Do you like Paris?,
    Can you speak Russian?
    Will you marry me?

Alternatively, they have a pattern that asks an ‘open’ question which can 
have any number of answers

    What did you have for breakfast?
    Which newspaper do you read?
    Who is your favourite actor?
            
In addition to the sub-steps and tagged entities, if we do indeed have a 
question we want to classify it, a classifier will let us know what the answer is about

```python
from little_questions import Question, Sentence

text = "who made you"
question = Sentence(text)

assert question.is_question
assert isinstance(question, Question)
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
```

We also want to know what kind of action we need to take to answer a question

NOTE: namespaces are a work in progress, taxonomy of intents might change

```python
from little_questions import Question

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

```


