# Little Questions

Classify and Parse questions

A decent question answering system needs to know what users are talking about


## Install

    pip install little_questions

## Usage

So what is a question? A question is a sentence, but a sentence might not be a
question

To understand if a sentence is a question we want to:

- classify it as question, command or statement
- extract the entities the sentence is about
- decompose it in individual actions
- classify the question according to answer type
- classify the question according to action type

### Questions

Questions can have two patterns. Some can have ‘yes’ or ‘no’ as an answer.

    Do you like Paris?,
    Can you speak Russian?
    Will you marry me?

Alternatively, they have a pattern that asks an ‘open’ question which can have
any number of answers

    What did you have for breakfast?
    Which newspaper do you read?
    Who is your favourite actor?

In addition to the sub-steps and tagged entities, if we do indeed have a
question we want to classify it, a classifier will let us know what the answer
is about

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

### Sentences

Sentences might not be questions, in this case we probably want to send them to
an intent parsing step in our pipeline and take an action

```python
from little_questions import Sentence

text = "I want you to buy bitcoin"

sentence = Sentence(text)

assert not sentence.is_question


```

### Statements

Sentences might also be statements that don't expect any action and only convey
information

A statement is defined as having a structure in which there is typically a
Subject, followed by a verb and then a further unit such as a Direct Object.

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

Exclamations grammatically have a structure that involves the words what a or
how,

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

We can make a request, which is a type of command, sound more polite by using
the interrogative.

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
   