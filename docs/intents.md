
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

Questions are parsed using [Padaos](https://github.com/MycroftAI/padaos), a dead simple regex intent parser

```python
from little_questions.parsers.rules import RegexQuestionParser
from train_scripts.data import SAMPLE_QUESTIONS
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
from train_scripts.data import SAMPLE_QUESTIONS
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

