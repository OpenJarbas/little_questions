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

import random

scorer = SentenceScorer()
states = ["Our dog eats any old thing.",
          "We have already won several races.",
          "The dog hasn’t been fed yet.",
          "I am poor",
          "pizza is awesome"]
excls = ["What a nice person you are!",
         "What a beautiful painting!",
         "How clever you are!",
         "How wonderful that is!"]
commands = ["Eat your dinner",
            "Be quiet",
            "Open the door",
            "Name your price",
            "Define evil"]
requests = ["Would you feed the dog, please.",
            "Would you mind shutting the door.",
            "Could I have that now, thank you."]
questions = []

for q in states:
    print("\nStatement:", q)
    print(scorer.predict(q))

for q in commands:
    print("\nCommand:", q)
    print(scorer.predict(q))

for q in excls:
    print("\nExclamation:", q)
    print(scorer.predict(q))

for q in requests:
    print("\nRequest:", q)
    print(scorer.predict(q))

print("Questions")
for q in questions:
    print("\nQuestion:", q)
    print(scorer.predict(q))

classifier = QuestionClassifier()
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "HUM:ind"

classifier = MainQuestionClassifier()
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "HUM"
