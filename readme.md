# Little Questions

Classify and Parse questions

A decent question answering system needs to know what users are talking about

- [Little Questions](#little-questions)
  * [Parsing](#parsing)
    + [Question Intent](#question-intent)
      - [Sample output](#sample-output)
  * [Classification](#classification)
    + [Models](#models)
    + [Taxonomy](#taxonomy)
    
    
## Parsing

Questions are parsed using Padaos, a dead simple regex intent parser

This is an early version, rules can be found and expanded [here](little_questions/res/en-us)

Feel free to submit PRs, specially for new languages!

### Question Intent

Questions are classified as one of these intents, this will be fine tuned and improved over time

Intents may disappear or be renamed at any time, wait until version 1.0 if 
you are going to depend on this being static!

    ['example', 'relate_time_and_thing', 'confirm_location', 'select_option', 
    'step_by_step', 'query_past_action', 'distance', 'age', 'describe_attribute', 
    'confirm', 'query_action', 'explanation', 'retrieve_information', 
    'causal_attribute', 'relate_time_and_place', 'quantity', 'responsible_entity', 
    'duration', 'place', 'defend', 'common_attributes', 'relate_to_entity', 
    'hability_check', 'relate_place_and_thing', 'relate_attributes', 
    'property_check', 'advice', 'time', 'unique_attributes', 'assign_entity']

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

There are 6 main labels

* ABBR - answer is an abbreviation
* DESC - answer is a description of something
* ENTY - answer is an entity/thing
* HUM - answer is a human
* LOC - answer is a location
* NUM - answer is numeric

### Models

Still experimenting and fine tuning parameters and features, consider this 
unstable

My aim is to have the best model for each algorithm as a baseline, no 
assumption is made about real world usage, you need to consider 
speed/memory/performance trade offs and decide which classifier is best for you

Classification of main label

* Passive Agressive - Accuracy: 0.878
* Linear SVC - Accuracy: 0.868
* Ridge - Accuracy: 0.868
* Gradient Boosting - Accuracy: 0.858
* Perceptron - Accuracy: 0.856
* Random Forest - Accuracy: 0.85
* Logistic Regression - Accuracy: 0.85
* SGD - Accuracy: 0.826
* Decision Tree - Accuracy: 0.792
* Naive Bayes - Accuracy: 0.782
* Extra Tree - Accuracy: 0.768

Classification of main + secondary label

* Passive Agressive - Accuracy: 0.792
* Gradient Boosting - Accuracy: 0.786
* Linear SVC - Accuracy: 0.782
* Ridge - Accuracy: 0.768
* SGD - Accuracy: 0.752 
* Random Forest - Accuracy: 0.75
* Perceptron - Accuracy: 0.728
* Decision Tree - Accuracy: 0.7
* Logistic Regression - Accuracy: 0.7
* ExtraTree - Accuracy: 0.654
* Naive Bayes - Accuracy: 0.518



Best accuracy model will always be used for DEFAULT_CLASSIFIER

```python
from little_questions.classifiers import QuestionClassifier
from little_questions.classifiers import SimpleQuestionClassifier

classifier = QuestionClassifier().load()
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "HUM:ind"

classifier = SimpleQuestionClassifier().load()
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "HUM"

```

You can also test specific classifiers

NOTE: the number of bundled trained models will eventually be pruned and 
moved elsewhere

```python
from little_questions.classifiers.passive_agressive import PassiveAggressiveQuestionClassifier
classifier = PassiveAggressiveQuestionClassifier().load()

from little_questions.classifiers.gradboost import GradientBoostingQuestionClassifier
classifier = GradientBoostingQuestionClassifier().load()

from little_questions.classifiers.svm import SVCQuestionClassifier
classifier = SVCQuestionClassifier().load()

from little_questions.classifiers.logreg import LogRegQuestionClassifier
classifier = LogRegQuestionClassifier().load()

from little_questions.classifiers.ridge import RidgeQuestionClassifier
classifier = RidgeQuestionClassifier().load()

from little_questions.classifiers.sgd import SGDQuestionClassifier
classifier = SGDQuestionClassifier().load()

from little_questions.classifiers.forest import ForestQuestionClassifier
# NOTE: random forest models are almost 400MB, and not included in this repo
train = True
classifier = ForestQuestionClassifier()
if train:
    t, tt = classifier.load_data()
    classifier.train(t, tt)
    classifier.save()
else:
    classifier.load()
    
from little_questions.classifiers.tree import TreeQuestionClassifier
classifier = TreeQuestionClassifier().load()

from little_questions.classifiers.perceptron import PerceptronQuestionClassifier
classifier = PerceptronQuestionClassifier().load()

from little_questions.classifiers.naive import NaiveQuestionClassifier
classifier = NaiveQuestionClassifier().load()


```


###  Taxonomy

Main labels are further specialized in secondary labels

* ABBREVIATION	abbreviation


      abb	        abbreviation
      exp	        expression abbreviated
  
  
* ENTITY	entities


      animal	    animals
      body	        organs of body
      color	        colors
      creative	    inventions, books and other creative pieces
      currency	    currency names
      dis.med.	    diseases and medicine
      event	        events
      food	        food
      instrument	musical instrument
      lang	        languages
      letter	    letters like a-z
      other	        other entities
      plant	        plants
      product	    products
      religion	    religions
      sport	        sports
      substance	    elements and substances
      symbol	    symbols and signs
      technique	    techniques and methods
      term	        equivalent terms
      vehicle	    vehicles
      word	        words with a special property
  
* DESCRIPTION	description and abstract concepts


      definition	definition of sth.
      description	description of sth.
      manner	    manner of an action
      reason	    reasons
  
  
* HUMAN	human beings


      group	        a group or organization of persons
      ind	        an individual
      title	        title of a person
      description	description of a person
      
      
* LOCATION	locations


      city	        cities
      country	    countries
      mountain	    mountains
      other	        other locations
      state	        states
  
  
* NUMERIC	numeric values


      code	        postcodes or other codes
      count	        number of sth.
      date	        dates
      distance	    linear measures
      money	        prices
      order	        ranks
      other	        other numbers
      period	    the lasting time of sth.
      percent	    fractions
      speed	        speed
      temp	        temperature
      size	        size, area and volume
      weight	    weight
      
