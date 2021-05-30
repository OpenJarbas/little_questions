from os.path import join, dirname
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix

from little_questions.classifiers import MainQuestionClassifier
from sklearn.model_selection import train_test_split

"""
accuracy - 0.8740740740740741
              precision    recall  f1-score   support

        ABBR       1.00      0.57      0.73        14
        DESC       0.85      0.90      0.88       193
        ENTY       0.81      0.83      0.82       226
         HUM       0.85      0.86      0.86       195
         LOC       0.93      0.89      0.91       141
         NUM       0.95      0.93      0.94       176

    accuracy                           0.87       945
   macro avg       0.90      0.83      0.86       945
weighted avg       0.88      0.87      0.87       945

[[  8   5   1   0   0   0]
 [  0 174  13   1   2   3]
 [  0  12 187  21   4   2]
 [  0   3  21 168   2   1]
 [  0   6   3   5 125   2]
 [  0   4   5   2   1 164]]

"""

DATA_PATH = join(dirname(__file__), "clean_data")
train_data_path = join(DATA_PATH, "raw_questions_0.7.0a1.txt")


clf = MainQuestionClassifier()


x, y = clf.load_data(train_data_path)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.15,
                                                    stratify=y)

clf.train(x_train, y_train)
clf.save("questions6.pkl")

preds = clf.predict(x_test)

accuracy = accuracy_score(y_test, preds)
matrix = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds)

print(report)
