from little_questions.classifiers import QuestionClassifier
from little_questions.classifiers import MainQuestionClassifier
from little_questions.classifiers import SentenceClassifier

classifier = QuestionClassifier()
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "HUM:ind"

classifier = MainQuestionClassifier()
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "HUM"

classifier = SentenceClassifier()
question = "who made you"
preds = classifier.predict([question])
assert preds[0] == "question"
