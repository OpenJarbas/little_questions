from os.path import join, dirname
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix

from little_questions.classifiers import MainQuestionClassifier
from sklearn.model_selection import train_test_split


"""
              precision    recall  f1-score   support

        ABBR       1.00      0.78      0.88         9
        DESC       0.86      0.99      0.92       138
        ENTY       0.85      0.81      0.83        94
         HUM       0.94      0.95      0.95        65
         LOC       0.91      0.85      0.88        81
         NUM       0.98      0.90      0.94       113

    accuracy                           0.90       500
   macro avg       0.92      0.88      0.90       500
weighted avg       0.91      0.90      0.90       500

[[  7   2   0   0   0   0]
 [  0 136   2   0   0   0]
 [  0   9  76   4   4   1]
 [  0   0   3  62   0   0]
 [  0   4   7   0  69   1]
 [  0   7   1   0   3 102]]
"""

DATA_PATH = join(dirname(__file__), "clean_data")

clf = MainQuestionClassifier()

train_data_path = join(DATA_PATH, "raw_questions+.txt")

x, y = clf.load_data(train_data_path)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.15,
                                                    random_state=42,
                                                    stratify=y)

clf.train(x_train, y_train)
clf.save("questions6.pkl")

preds = clf.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy)
report = classification_report(y_test, preds)
print(report)
matrix = confusion_matrix(y_test, preds)
print(matrix)
