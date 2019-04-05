# Little Questions

Classify and Parse questions

A decent question answering system needs to know what users are talking about

- [Little Questions](#little-questions)
  * [Install](#install)
  * [Anatomy of a question](#anatomy-of-a-question)
    + [Commands](#commands)
    + [Statements](#statements)
    + [Questions](#questions)
  * [Classification](#classification)
    + [Models](#models)
      - [Classification of sentence type](#classification-of-sentence-type)
      - [Classification of main label](#classification-of-main-label)
      - [Classification of main + secondary label](#classification-of-main---secondary-label)
  * [Question Intent](#question-intent)
    + [Basic Parsing](#basic-parsing)
    + [Regex Parsing](#regex-parsing)
      - [Sample output](#sample-output)
    + [Neural Parsing](#neural-parsing)
      - [Output Comparison](#output-comparison)
 
## Install

Available on pip

    pip install little_questions
    
from source

    git clone https://github.com/JarbasAl/little_questions
    cd little_questions
    pip install .

## Anatomy of a question

So what is a question? A question is a sentence, but a sentence might not be a question

To understand if a sentence is a question we want to:

- classify it as question, command or statement
- extract the entities the sentence is about
- decompose it in individual actions
- classify the question according to answer type
- classify the question according to action type

### Commands

Sentences might be a command, in this case we probably want to send them to 
an intent parsing step in our pipeline and take an action


```python
from little_questions import Command

text = "I want you to buy bitcoin"

sentence = Command(text)

assert not sentence.is_question

```
A sentence might be about different topics, basic topic analysis and entity 
extraction is done with [simple_NER](https://github.com/JarbasAl/simple_NER)

```python
from little_questions import Command

text = "Portugal once split the world in half with Spain"

sentence = Command(text)

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
from little_questions import Command

text = "I want you to buy bitcoin"

sentence = Command(text)

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
### Statements

Sentences might also be statements that don't expect any action and only 
convey information, classifying statements vs commands is a work in progress
 (need data!) and currently unreliable
 
Logic to detect imperative sentences also a work in progress, once accuracy 
improves Command and Statement will be split into it's own class
 
```python
from little_questions import Command
text = "I like pizza"

sentence = Command(text)

assert not sentence.is_question
assert not sentence.is_command
assert sentence.is_statement

text = "Open the pod bay doors"

sentence = Command(text)

assert not sentence.is_question
try:
    assert sentence.is_command  # SHOULD BE TRUE but isn't
    assert not sentence.is_statement
except:
    print("command vs statement needs work!")
```
### Questions

In addition to the sub-steps and tagged entities, if we do indeed have a 
question we want to classify it, a classifier will let us know what the answer is about

```python
from little_questions import Question, Command

text = "who made you"
question = Question(text)

assert question.is_question
assert isinstance(question, Command)
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

NOTE: namespaces are a work in progress, taxonomy might change

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

## Classification

Training classifiers with [this data](http://cogcomp.org/Data/QA/QC/)

There are 6 main labels

* ABBR - answer is an abbreviation
* DESC - answer is a description of something
* ENTY - answer is an entity/thing
* HUM - answer is a human
* LOC - answer is a location
* NUM - answer is numeric

Best accuracy model will always be used for DEFAULT_CLASSIFIER

```python
from little_questions.classifiers import QuestionClassifier
from little_questions.classifiers import MainQuestionClassifier

classifier = QuestionClassifier()
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "HUM:ind"

classifier = MainQuestionClassifier()
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "HUM"

```

### Models

For model accuracy baseline the following features are extracted

- CountVectorizer, n_gram range (1,2)
- TfidfVectorizer, n_gram range (1,2), lemmatized input
- Word2Vec
- PosTagVectorizer

you need to consider speed/memory/performance trade offs and decide which classifier is best for you

NOTE: optimal pipeline/features and hyperparameters under investigation

#### Classification of sentence type

* Passive Aggressive - Accuracy: 0.8666666666666667
* Linear SVC - Accuracy: 0.8666666666666667
* Decision Tree - Accuracy: 0.8666666666666667
* Perceptron - Accuracy: 0.7333333333333333
* Ridge - Accuracy: 0.6666666666666666
* SGD - Accuracy: 0.5333333333333333
* AdaBoost - Accuracy: 0.4666666666666667

#### Classification of main label

* Linear SVC - Accuracy: 0.902
* Ridge - Accuracy: 0.896
* Logistic Regression - Accuracy: 0.894
* SGD - Accuracy: 0.888
* Passive Aggressive - Accuracy: 0.882
* Naive Bayes - Accuracy: 0.81
* Perceptron - Accuracy: 0.872
* Gradient Boosting - Accuracy: 0.858
* Random Forest - Accuracy: 0.798
* Decision Tree - Accuracy: 0.784
* AdaBoost - Accuracy: 0.592

#### Classification of main + secondary label

* Linear SVC - Accuracy: 0.838
* Passive Aggressive - Accuracy: 0.804
* Ridge - Accuracy: 0.834
* SGD - Accuracy: 0.802
* Logistic Regression - Accuracy: 0.794
* Gradient Boosting - Accuracy: 0.776
* Perceptron - Accuracy: 0.766
* Decision Tree - Accuracy: 0.666
* Random Forest - Accuracy: 0.636
* ExtraTree - Accuracy: 0.548
* Naive Bayes - Accuracy: 0.53
* AdaBoost - Accuracy: 0.22

You can test specific classifiers

```python
from little_questions.classifiers.passive_agressive import PassiveAggressiveQuestionClassifier
classifier = PassiveAggressiveQuestionClassifier()

from little_questions.classifiers.gradboost import GradientBoostingQuestionClassifier
classifier = GradientBoostingQuestionClassifier()

from little_questions.classifiers.svm import SVCQuestionClassifier
classifier = SVCQuestionClassifier()

from little_questions.classifiers.logreg import LogRegQuestionClassifier
classifier = LogRegQuestionClassifier()

from little_questions.classifiers.ridge import RidgeQuestionClassifier
classifier = RidgeQuestionClassifier()

from little_questions.classifiers.sgd import SGDQuestionClassifier
classifier = SGDQuestionClassifier()

from little_questions.classifiers.forest import ForestQuestionClassifier
classifier = ForestQuestionClassifier()
    
from little_questions.classifiers.tree import TreeQuestionClassifier
classifier = TreeQuestionClassifier()

from little_questions.classifiers.perceptron import PerceptronQuestionClassifier
classifier = PerceptronQuestionClassifier()

from little_questions.classifiers.naive import NaiveQuestionClassifier
classifier = NaiveQuestionClassifier()


# train / load
train = True
if train:
    t, tt = classifier.load_data()
    classifier.train(t, tt)
    classifier.save()
else:
    classifier.load()
    
# test
X_test, y_test = classifier.load_test_data()
preds = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)
matrix = confusion_matrix(y_test, preds)

```


## Question Intent

Questions Intents are extracted according to rules

Feel free to submit PRs, specially for new languages!

Questions are classified as one of these intents, this will be fine tuned and improved over time

Intents may disappear or be renamed at any time, wait until version 1.0 if 
you are going to depend on this being static!

A better intent taxonomy will be developed, this a rough first version to 
validate the approach

    ['example', 'relate_time_and_thing', 'confirm_location', 'select_option', 
    'step_by_step', 'query_past_action', 'distance', 'age', 'describe_attribute', 
    'confirm', 'query_action', 'explanation', 'retrieve_information', 
    'causal_attribute', 'relate_time_and_place', 'quantity', 'responsible_entity', 
    'duration', 'place', 'defend', 'common_attributes', 'relate_to_entity', 
    'hability_check', 'relate_place_and_thing', 'relate_attributes', 
    'property_check', 'advice', 'time', 'unique_attributes', 'assign_entity']

### Basic Parsing

this can barely be called parsing, but by checking how the sentence starts 
we immediately get some information, e.g. "when" is about time

```python
from little_questions.parsers import BasicQuestionParser

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
```

Entity recognition can also be done for any text

```python
from little_questions.parsers import SlotParser

parser = SlotParser()

data = parser.parse_slots("Portugal was founded in October 5, 1910")
assert data == {'date': ['october 5 1910'],
                'entity': [],
                'location': ['Portugal'],
                'person': [],
                'property': [],
                'quantity': [],
                'thing': ['founded', 'october 5', 'portugal', '1910']}
```

And text can be chunked in subquestions

```python
from little_questions.parsers import QuestionChunker

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
```

### Regex Parsing

This is an early version, rules can be found and expanded [here](little_questions/res/en-us)

Questions are parsed using [Padaos](https://github.com/MycroftAI/padaos), a dead simple regex calc_intent parser
 
```python
from little_questions.parsers.rules import RegexQuestionParser
from little_questions.data import SAMPLE_QUESTIONS
from pprint import pprint
import random

parser = RegexQuestionParser()
print(parser.intents)

questions = SAMPLE_QUESTIONS
random.shuffle(questions)

for q in questions:
    data = parser.parse(q)
    print("Q:", q)
    print("Intent:", data['QuestionIntent'])
    pprint(data)
    print("___")
```

#### Sample output

        Q: what was the bridge of san luis rey made of ?

        Intent: describe_attribute
        {'Question': 'what was bridge of san luis rey made of ?',
         'QuestionIntent': 'describe_attribute',
         'property': 'bridge',
         'thing': 'san luis rey made of'}
        ___
        Q: is this for electric stoves ?
        
        Intent: confirm
        {'Question': 'is this for electric stoves ?',
         'QuestionIntent': 'confirm',
         'statement': 'for electric stoves'}
        ___
        Q: what college produced the most winning super bowl quarterbacks ?
        
        Intent: retrieve_information
        {'Question': 'what college produced most winning super bowl quarterbacks ?',
         'QuestionIntent': 'retrieve_information',
         'thing': 'college produced most winning super bowl quarterbacks'}
        ___
        Q: can more than one water source be tested ?
        
        Intent: hability_check
        {'Question': 'can more than one water source be tested ?',
         'QuestionIntent': 'hability_check',
         'hability': 'tested',
         'thing': 'more than one water source'}
        
        

### Neural Parsing

Questions are parsed using [Padatious](https://github.com/MycroftAI/padatious), An efficient and agile neural network intent parser
 
Same rules as before are used

```python
from little_questions.parsers.neural import NeuralQuestionParser
from little_questions.data import SAMPLE_QUESTIONS
from pprint import pprint
import random

parser = NeuralQuestionParser()
questions = SAMPLE_QUESTIONS
random.shuffle(questions)

for q in questions:
    data = parser.parse(q)
    print("Q:", q)
    print("Intent:", data['QuestionIntent'])
    pprint(data)
    print("___")
```


#### Output Comparison

Here is a comparison of questions were both parsers disagree

    Q: What planet is known as the `` red '' planet ?

    Intent: describe_attribute
    Intent Data: {'property': 'planet is known', 'thing': "red '' planet"}
    Neural Intent: select_option
    Neural Data: {'query': "known as the `` red '' planet", 'option': 'planet', 'conf': 1.0}
    ___
    Q: How cold should a refrigerator be ?
    
    Intent: unknown
    Intent Data: {}
    Neural Intent: relate_time_and_attribute
    Neural Data: {'thing': 'cold should a refrigerator be', 'conf': 0.852873758548952}
    ___
    Q: Why in tennis are zero points called love ?
    
    Intent: unknown
    Intent Data: {}
    Neural Intent: explanation
    Neural Data: {'statement': 'tennis are zero points called love', 'conf': 0.8961664203704263}
    ___
    Q: In Poland , where do most people live ?
    
    Intent: unknown
    Intent Data: {}
    Neural Intent: place
    Neural Data: {'property': 'most people live', 'thing': 'in poland ,', 'conf': 0.9561917913508412}
    ___
    Q: Who was the second man to walk on the moon ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'was second man to walk', 'location': 'moon'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'walk on the moon', 'conf': 1.0, 'entity': 'was the second man'}
    ___
    Q: What is the cause of endangered species ?
    
    Intent: describe_attribute
    Intent Data: {'thing': 'endangered species', 'property': 'cause'}
    Neural Intent: causal_attribute
    Neural Data: {'conf': 1.0, 'effect': 'of endangered species'}
    ___
    Q: Who leads the star ship Enterprise in Star Trek ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'leads star ship enterprise', 'location': 'star trek'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'Star Trek', 'conf': 1.0, 'entity': 'leads the star ship Enterprise'}
    ___
    Q: About how many soldiers died in World War II ?
    
    Intent: quantity
    Intent Data: {'thing': 'soldiers died in world war ii'}
    Neural Intent: retrieve_information
    Neural Data: {'thing': 'how many soldiers died in world war ii', 'conf': 0.9361642176569306}
    ___
    Q: What makes thunder ?
    
    Intent: retrieve_information
    Intent Data: {'thing': 'makes thunder'}
    Neural Intent: relate_attributes
    Neural Data: {'property': 'thunder', 'conf': 1.0}
    ___
    Q: Who wrote the bestselling Missionary Travels and Researches in South Africa , published in 1857 ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'wrote bestselling missionary travels and researches', 'location': 'south africa , published in 1857'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'South Africa , published in 1857', 'conf': 1.0, 'entity': 'wrote the bestselling Missionary Travels and Researches'}
    ___
    Q: What makes Black Hills , South Dakota a tourist attraction ?
    
    Intent: retrieve_information
    Intent Data: {'thing': 'makes black hills , south dakota tourist attraction'}
    Neural Intent: relate_attributes
    Neural Data: {'property': 'black hills , south dakota a tourist attraction', 'conf': 1.0}
    ___
    Q: Who played the title role in My Favorite Martian ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'played title role', 'location': 'my favorite martian'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'My Favorite Martian', 'conf': 1.0, 'entity': 'played the title role'}
    ___
    Q: What makes you fat ?
    
    Intent: retrieve_information
    Intent Data: {'thing': 'makes you fat'}
    Neural Intent: relate_attributes
    Neural Data: {'property': 'you fat', 'conf': 1.0}
    ___
    Q: On average , how long time does it take to type a screenplay ?
    
    Intent: describe_attribute
    Intent Data: {'thing': 'it take to type screenplay', 'property': 'long time'}
    Neural Intent: time
    Neural Data: {'thing': 'does it take to type a screenplay', 'conf': 0.7373458696442745}
    ___
    Q: Who 's the only president buried in Washington
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 's only president buried', 'location': 'washington'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'Washington', 'conf': 1.0, 'entity': 's the only president buried'}
    ___
    Q: Name the story by Chris Van Allsburg in the which a boy tries to become a great sailor ?
    
    Intent: example
    Intent Data: {'thing': 'story by chris van allsburg', 'property': 'boy tries to become great sailor'}
    Neural Intent: select_option
    Neural Data: {'conf': 1.0, 'option': 'Name the story by Chris Van Allsburg in the', 'query': 'a boy tries to become a great sailor'}
    ___

