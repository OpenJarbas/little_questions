# Little Questions

Classify and Parse questions

## Parsing

Questions are classified as one of these intents

    ['example', 'relate_time_and_thing', 'confirm_location', 'select_option', 
    'step_by_step', 'query_past_action', 'distance', 'age', 'describe_attribute', 
    'confirm', 'query_action', 'explanation', 'retrieve_information', 
    'causal_attribute', 'relate_time_and_place', 'quantity', 'responsible_entity', 
    'duration', 'place', 'defend', 'common_attributes', 'relate_to_entity', 
    'hability_check', 'relate_place_and_thing', 'relate_attributes', 
    'property_check', 'advice', 'time', 'unique_attributes', 'assign_entity']


Usage


```python
from little_questions.parsers import BasicQuestionParser
from little_questions.data import SAMPLE_QUESTIONS
from pprint import pprint
import random

parser = BasicQuestionParser()
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

sample output

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
        ___
        Q: does this pick up dust without blowing it out the bag ?
        
        Intent: confirm
        {'Question': 'does this pick up dust without blowing it out bag ?',
         'QuestionIntent': 'confirm',
         'statement': 'pick up dust without blowing it out bag'}
        ___
        Q: who was the first jockey to ride two triple crown winners ?
        
        Intent: relate_to_entity
        {'Question': 'who was first jockey to ride two triple crown winners ?',
         'QuestionIntent': 'relate_to_entity',
         'entity': 'was first jockey',
         'thing': 'ride two triple crown winners'}
        ___
        Q: who seized power from milton obote in 1971 ?
        
        Intent: relate_to_entity
        {'Question': 'who seized power from milton obote in 1971 ?',
         'QuestionIntent': 'relate_to_entity',
         'entity': 'seized power from milton obote',
         'thing': '1971'}
        ___
        Q: who is terrence malick ?
        
        Intent: assign_entity
        {'Question': 'who is terrence malick ?',
         'QuestionIntent': 'assign_entity',
         'statement': 'is terrence malick'}
        ___
        Q: what are the 7 articles of the constitution ?
        
        Intent: describe_attribute
        {'Question': 'what are 7 articles of constitution ?',
         'QuestionIntent': 'describe_attribute',
         'property': '7 articles',
         'thing': 'constitution'}
        ___
        Q: how is the election of a new pope announced to the world ?
        
        Intent: step_by_step
        {'Question': 'how is election of new pope announced to world ?',
         'QuestionIntent': 'step_by_step',
         'query': 'election new pope announced world'}
        ___
        Q: name the largest country in south america .
        
        Intent: example
        {'Question': 'name largest country in south america .',
         'QuestionIntent': 'example',
         'thing': 'largest country in south america'}
        ___
                
        
## Classification

Training classifiers with [this data](http://cogcomp.org/Data/QA/QC/)

Still experimenting and finetuning parameters, consider this unstable

* SGD - Accuracy: 0.752 
* Logistic Regression - Accuracy: 0.7
* Naive Bayes - Accuracy: 0.518

Best model will always be used for DEFAULT_CLASSIFIER

```python
from little_questions.classifiers import QuestionClassifier
from little_questions.settings import DEFAULT_CLASSIFIER

classifier = QuestionClassifier().load(DEFAULT_CLASSIFIER)
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "HUM:ind"
```

You can also test specific classifiers

```python
from little_questions.classifiers.logreg import LogRegQuestionClassifier
classifier = LogRegQuestionClassifier().load()

from little_questions.classifiers.naive import NaiveQuestionClassifier
classifier = NaiveQuestionClassifier().load()

from little_questions.classifiers.sgd import SGDQuestionClassifier
classifier = SGDQuestionClassifier().load()
```