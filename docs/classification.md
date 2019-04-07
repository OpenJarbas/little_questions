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