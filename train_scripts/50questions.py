from os.path import join, dirname
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from little_questions.classifiers import QuestionClassifier

"""
accuracy - 0.8105820105820106
                precision    recall  f1-score   support

      ABBR:abb       0.67      0.67      0.67         3
      ABBR:exp       0.90      0.82      0.86        11
      DESC:def       0.78      0.95      0.86        80
     DESC:desc       0.72      0.79      0.75        42
   DESC:manner       1.00      0.98      0.99        42
   DESC:reason       0.90      0.97      0.93        29
   ENTY:animal       0.67      0.73      0.70        22
     ENTY:body       0.67      0.67      0.67         9
    ENTY:color       1.00      1.00      1.00        10
   ENTY:cremat       0.77      0.65      0.70        31
 ENTY:currency       1.00      1.00      1.00         1
   ENTY:dismed       0.87      0.81      0.84        16
    ENTY:event       1.00      0.56      0.71         9
     ENTY:food       0.55      0.67      0.60        18
   ENTY:instru       1.00      1.00      1.00         1
     ENTY:lang       1.00      1.00      1.00         3
   ENTY:letter       0.50      0.50      0.50         2
    ENTY:other       0.36      0.38      0.37        34
    ENTY:plant       0.67      0.67      0.67         3
  ENTY:product       1.00      0.57      0.73         7
 ENTY:religion       1.00      0.75      0.86         4
    ENTY:sport       1.00      0.67      0.80         9
ENTY:substance       1.00      0.38      0.55         8
   ENTY:symbol       0.71      0.71      0.71         7
 ENTY:techmeth       0.00      0.00      0.00         6
   ENTY:termeq       0.64      0.60      0.62        15
      ENTY:veh       0.75      0.38      0.50         8
     ENTY:word       0.50      0.50      0.50         4
      HUM:desc       1.00      1.00      1.00         7
        HUM:gr       0.74      0.48      0.58        29
       HUM:ind       0.78      0.89      0.83       152
     HUM:title       0.80      0.57      0.67         7
      LOC:city       0.95      0.86      0.90        22
   LOC:country       0.92      1.00      0.96        24
  LOC:landmass       1.00      0.67      0.80         6
     LOC:mount       1.00      1.00      1.00         6
     LOC:other       0.90      0.83      0.86        64
     LOC:state       0.92      1.00      0.96        11
     LOC:water       0.73      1.00      0.84         8
      NUM:code       0.80      1.00      0.89         4
     NUM:count       0.91      0.96      0.94        54
      NUM:date       0.98      1.00      0.99        40
      NUM:dist       1.00      0.50      0.67        12
     NUM:money       0.83      0.50      0.62        10
       NUM:ord       0.88      0.88      0.88         8
     NUM:other       0.56      0.62      0.59         8
      NUM:perc       1.00      0.25      0.40         4
    NUM:period       1.00      0.92      0.96        12
     NUM:speed       0.67      1.00      0.80         4
      NUM:temp       1.00      1.00      1.00         4
   NUM:volsize       1.00      1.00      1.00         6
    NUM:weight       1.00      0.89      0.94         9

      accuracy                           0.81       945
     macro avg       0.83      0.75      0.77       945
  weighted avg       0.82      0.81      0.80       945
"""

DATA_PATH = join(dirname(__file__), "clean_data")
train_data_path = join(DATA_PATH, "raw_questions_0.7.0a1.txt")

clf = QuestionClassifier()

x, y = clf.load_data(train_data_path)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.15,
                                                    stratify=y)

clf.train(x_train, y_train)
clf.save("questions50.pkl")

preds = clf.predict(x_test)
accuracy = accuracy_score(y_test, preds)
matrix = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds)
print(report)