# Little Questions

Classify and Parse questions

A decent question answering system needs to know what users are talking about

- [Little Questions](#little-questions)
  * [Classification](#classification)
    + [Models](#models)
  * [Question Intent](#question-intent)
    + [Regex Parsing](#regex-parsing)
      - [Sample output](#sample-output)
    + [Neural Parsing](#neural-parsing)
      - [Output Comparison](#output-comparison)
 
    
## Classification

Training classifiers with [this data](http://cogcomp.org/Data/QA/QC/)

There are 6 main labels

* ABBR - answer is an abbreviation
* DESC - answer is a description of something
* ENTY - answer is an entity/thing
* HUM - answer is a human
* LOC - answer is a location
* NUM - answer is numeric

Still experimenting and fine tuning parameters and features, consider this 
unstable

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

### Models

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


## Question Intent

Questions Intents are extracted according to rules

This is an early version, rules can be found and expanded [here](little_questions/res/en-us)

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

### Regex Parsing

Questions are parsed using [Padaos](https://github.com/MycroftAI/padaos), a dead simple regex intent parser
 
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
                
        

### Neural Parsing

Questions are parsed using [Padatious](https://github.com/MycroftAI/padatious), An efficient and agile neural network  intent parser
 
You need an extra install step in order to use this

    pip install fann2==1.0.7
    pip install padatious==0.4.5
    
    
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
    Q: Who delivered his last newscast on March 6 , 1981 ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'delivered his last newscast', 'location': 'march 6 , 1981'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'March 6 , 1981', 'conf': 1.0, 'entity': 'delivered his last newscast'}
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
    Q: Who was the first American in space ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'was first american', 'location': 'space'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'space', 'conf': 1.0, 'entity': 'was the first American'}
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
    Q: Who portrayed portly criminologist Carl Hyatt on Checkmate ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'portrayed portly criminologist carl hyatt', 'location': 'checkmate'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'Checkmate', 'conf': 1.0, 'entity': 'portrayed portly criminologist Carl Hyatt'}
    ___
    Q: Who won the Nobel Peace Prize in 1991 ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'won nobel peace prize', 'location': '1991'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': '1991', 'conf': 1.0, 'entity': 'won the Nobel Peace Prize'}
    ___
    Q: Who portrayed `` the man without a face '' in the movie of the same name ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'portrayed `` man without face', 'location': 'movie of same name'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'the movie of the same name', 'conf': 1.0, 'entity': 'portrayed `` the man without a face'}
    ___
    Q: Who portrayed Prewett in From Here to Eternity ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'portrayed prewett', 'location': 'from here to eternity'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'From Here to Eternity', 'conf': 1.0, 'entity': 'portrayed Prewett'}
    ___
    Q: Who played the title role in I Was a Teenage Werewolf ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'played title role', 'location': 'i was teenage werewolf'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'I Was a Teenage Werewolf', 'conf': 1.0, 'entity': 'played the title role'}
    ___
    Q: Who is the prime minister in Norway ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'is prime minister', 'location': 'norway'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'Norway', 'conf': 1.0, 'entity': 'is the prime minister'}
    ___
    Q: Who is Westview High 's band director in Funky Winkerbean ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': "is westview high 's band director", 'location': 'funky winkerbean'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'Funky Winkerbean', 'conf': 1.0, 'entity': "is Westview High 's band director"}
    ___
    Q: What makes Black Hills , South Dakota a tourist attraction ?
    
    Intent: retrieve_information
    Intent Data: {'thing': 'makes black hills , south dakota tourist attraction'}
    Neural Intent: relate_attributes
    Neural Data: {'property': 'black hills , south dakota a tourist attraction', 'conf': 1.0}
    ___
    Q: Who won two gold medals in skiing in the Olympic Games in Calgary ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'won two gold medals', 'location': 'skiing in olympic games in calgary'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'skiing in the Olympic Games in Calgary', 'conf': 1.0, 'entity': 'won two gold medals'}
    ___
    Q: Who was in Death of a Salesman original movie , not 1985 ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'was', 'location': 'death of salesman original movie , not 1985'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'Death of a Salesman original movie , not 1985', 'conf': 1.0, 'entity': 'was'}
    ___
    Q: Who lived on the shores of the Gitchee Gumee River ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'lived', 'location': 'shores of gitchee gumee river'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'the shores of the Gitchee Gumee River', 'conf': 1.0, 'entity': 'lived'}
    ___
    Q: Who was shot in the back during a Poker game in Deadwood , the Dakota territory ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'was shot', 'location': 'back during poker game in deadwood , dakota territory'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'the back during a Poker game in Deadwood , the Dakota territory', 'conf': 1.0, 'entity': 'was shot'}
    ___
    Q: Who played the title role in The Romantic Englishwoman ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'played title role', 'location': 'the romantic englishwoman'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'The Romantic Englishwoman', 'conf': 1.0, 'entity': 'played the title role'}
    ___
    Q: Who replies `` I know '' to Princess Leia 's confession `` I love you '' in The Empire Strikes Back ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': "replies `` i know '' to princess leia 's confession `` i love you", 'location': 'the empire strikes back'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': "Princess Leia 's confession `` I love you '' in The Empire Strikes Back", 'conf': 1.0, 'entity': 'replies `` I know'}
    ___
    Q: Who played the title role in My Favorite Martian ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'played title role', 'location': 'my favorite martian'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'My Favorite Martian', 'conf': 1.0, 'entity': 'played the title role'}
    ___
    Q: Who is the richest person in the world ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'is richest person', 'location': 'world'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'the world', 'conf': 1.0, 'entity': 'is the richest person'}
    ___
    Q: Who portrayed `` Rosanne Rosanna-Dana '' on the television show `` Saturday Night Live '' ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'portrayed `` rosanne rosanna-dana', 'location': 'television show `` saturday night live'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'the television show `` Saturday Night Live', 'conf': 1.0, 'entity': 'portrayed `` Rosanne Rosanna-Dana'}
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
    Q: Who is Ishmael in Moby Dick ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'is ishmael', 'location': 'moby dick'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'Moby Dick', 'conf': 1.0, 'entity': 'is Ishmael'}
    ___
    Q: Who is currently the most popular singer in the world ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'is currently most popular singer', 'location': 'world'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'the world', 'conf': 1.0, 'entity': 'is currently the most popular singer'}
    ___
    Q: Who is the richest woman in the world ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'is richest woman', 'location': 'world'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'the world', 'conf': 1.0, 'entity': 'is the richest woman'}
    ___
    Q: Name the story by Chris Van Allsburg in the which a boy tries to become a great sailor ?
    
    Intent: example
    Intent Data: {'thing': 'story by chris van allsburg', 'property': 'boy tries to become great sailor'}
    Neural Intent: select_option
    Neural Data: {'conf': 1.0, 'option': 'Name the story by Chris Van Allsburg in the', 'query': 'a boy tries to become a great sailor'}
    ___
    Q: Who portrayed the title character in the film The Day of the Jackal ?
    
    Intent: relate_place_and_entity
    Intent Data: {'entity': 'portrayed title character', 'location': 'film the day of jackal'}
    Neural Intent: relate_to_entity
    Neural Data: {'thing': 'the film The Day of the Jackal', 'conf': 1.0, 'entity': 'portrayed the title character'}
    ___
