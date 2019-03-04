# Little Questions

Classify and Parse questions

## Parsing


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

        Q: what is nebraska 's most valuable resource ?
        
        Intent: retrieve_information
        {'Question': "what is nebraska's most valuable resource",
         'QuestionIntent': 'retrieve_information',
         'thing': "nebraska's most valuable resource"}
        ___
        Q: who led the opposition when konrad adenauer was chancellor in germany ?
        
        Intent: relate_to_entity
        {'Question': 'who led the opposition when konrad adenauer was chancellor in '
                     'germany',
         'QuestionIntent': 'relate_to_entity',
         'entity': 'led the oppositi',
         'thing': 'when konrad adenauer was chancellor in germany'}
        ___
        Q: are these drip pans dishwasher safe ?
        
        Intent: confirm
        {'Question': 'are these drip pans dishwasher safe',
         'QuestionIntent': 'confirm',
         'statement': 'drip pans dishwasher safe'}
        ___
        Q: when was hurricane hugo ?
        
        Intent: time
        {'Question': 'when was hurricane hugo',
         'QuestionIntent': 'time',
         'thing': 'hurricane hugo'}
        ___
        Q: what 's the american dollar equivalent for 8 pounds in the u.k. ?
        
        Intent: describe_attribute
        {'Question': "what's the american dollar equivalent for 8 pounds in the u.k.",
         'QuestionIntent': 'describe_attribute',
         'property': 's the american dollar equivalent',
         'thing': '8 pounds in the u.k'}
        ___
        Q: does this test forlead ?
        
        Intent: confirm
        {'Question': 'does this test forlead',
         'QuestionIntent': 'confirm',
         'statement': 'test forlead'}
        ___
        Q: what do you call a group of geese ?
        
        Intent: describe_attribute
        {'Question': 'what do you call a group of geese',
         'QuestionIntent': 'describe_attribute',
         'property': 'you call a group',
         'thing': 'geese'}
        ___
        Q: do they come in different sizes ?
        
        Intent: confirm
        {'Question': 'do they come in different sizes',
         'QuestionIntent': 'confirm',
         'statement': 'come in different sizes'}
        ___
        Q: who was credited with saying : `` i never met a man i did n't like '' ?
        
        Intent: relate_to_entity
        {'Question': "who was credited with saying :  i never met a man i did n't "
                     "like''",
         'QuestionIntent': 'relate_to_entity',
         'entity': 'was credited',
         'thing': "saying :  i never met a man i did n't like"}
        ___
        Q: whom did the chicago bulls beat in the 1993 championship ?
        
        Intent: responsible_entity
        {'Question': 'whom did the chicago bulls beat in the 1993 championship',
         'QuestionIntent': 'responsible_entity',
         'thing': 'did the chicago bulls beat in the 1993 championship'}
        ___
        Q: what is `` the bear of beers '' ?
        
        Intent: describe_attribute
        {'Question': "what is  the bear of beers''",
         'QuestionIntent': 'describe_attribute',
         'property': 'the bear',
         'thing': 'beers'}
        ___
        Q: how can you contact play producers and promoters on-line ?
        
        Intent: step_by_step
        {'Question': 'how can you contact play producers and promoters on-line',
         'QuestionIntent': 'step_by_step',
         'query': 'contact play producers and promoters on-line'}
        ___
        Q: what is color ?
        
        Intent: retrieve_information
        {'Question': 'what is color',
         'QuestionIntent': 'retrieve_information',
         'thing': 'color'}
        ___
        Q: what city boasts penn 's landing , on the banks of the delaware river ?
        
        Intent: place
        {'Question': "what city boasts penn's landing  on the banks of the delaware "
                     'river',
         'QuestionIntent': 'place',
         'property': 'river',
         'thing': "boasts penn's landing  on the banks of the delaw"}
        ___
        Q: what countries have the largest areas of forest ?
        
        Intent: relate_attributes
        {'Question': 'what countries have the largest areas of forest',
         'QuestionIntent': 'relate_attributes',
         'property': 'forest',
         'thing': 'countries have the largest areas'}
        ___
        Q: what is eagle 's syndrome styloid process ?
        
        Intent: retrieve_information
        {'Question': "what is eagle's syndrome styloid process",
         'QuestionIntent': 'retrieve_information',
         'thing': "eagle's syndrome styloid process"}
        ___
        Q: what does the name shawn mean ?
        
        Intent: retrieve_information
        {'Question': 'what does the name shawn mean',
         'QuestionIntent': 'retrieve_information',
         'thing': 'the name shawn mean'}
        ___
        Q: what `` magic '' does mandrake employ ?
        
        Intent: describe_attribute
        {'Question': "what  magic'' does mandrake employ",
         'QuestionIntent': 'describe_attribute',
         'property': 'magic',
         'thing': 'mandrake employ'}
        ___
        
        
## Classification

TODO http://cogcomp.org/Data/QA/QC/

