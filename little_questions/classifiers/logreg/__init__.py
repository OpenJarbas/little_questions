from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.logreg import LogRegTextClassifier
from os.path import join


class LogRegQuestionClassifier(QuestionClassifier, LogRegTextClassifier):
    # default pipeline - Accuracy: 0.794
    _accuracy = 0.794
    _report = """
                       precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.75      0.86         8
          DESC:def       0.79      1.00      0.88       123
         DESC:desc       0.33      0.71      0.45         7
       DESC:manner       0.67      1.00      0.80         2
       DESC:reason       1.00      1.00      1.00         6
       ENTY:animal       0.70      0.44      0.54        16
         ENTY:body       1.00      0.50      0.67         2
        ENTY:color       1.00      0.90      0.95        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       1.00      0.33      0.50         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       1.00      0.25      0.40         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
        ENTY:other       0.33      0.33      0.33        12
        ENTY:plant       0.00      0.00      0.00         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       1.00      0.33      0.50        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.50      0.86      0.63         7
          ENTY:veh       1.00      0.25      0.40         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.80      0.67      0.73         6
           HUM:ind       0.84      0.96      0.90        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       1.00      0.78      0.88        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.67      0.67      0.67         3
         LOC:other       0.76      0.88      0.81        50
         LOC:state       0.62      0.71      0.67         7
         NUM:count       0.90      1.00      0.95         9
          NUM:date       0.98      0.96      0.97        47
          NUM:dist       1.00      0.56      0.72        16
         NUM:money       0.50      0.33      0.40         3
         NUM:other       0.86      0.50      0.63        12
          NUM:perc       0.67      0.67      0.67         3
        NUM:period       0.73      1.00      0.84         8
         NUM:speed       1.00      0.50      0.67         6
          NUM:temp       1.00      0.20      0.33         5
        NUM:weight       1.00      0.25      0.40         4
    
         micro avg       0.79      0.79      0.79       500
         macro avg       0.72      0.61      0.62       500
      weighted avg       0.81      0.79      0.77       500
        """
    _matrix = """
        [[  1   0   0 ...   0   0   0]
         [  0   6   2 ...   0   0   0]
         [  0   0 123 ...   0   0   0]
         ...
         [  0   0   2 ...   3   0   0]
         [  0   0   1 ...   0   1   0]
         [  0   0   1 ...   0   0   1]]

        """


class LogRegMainQuestionClassifier(MainQuestionClassifier,
                                   LogRegTextClassifier):
    # default pipeline - Accuracy: 0.894
    _accuracy = 0.894
    _report = """
                   precision    recall  f1-score   support
    
            ABBR       1.00      0.78      0.88         9
            DESC       0.84      0.99      0.91       138
            ENTY       0.84      0.80      0.82        94
             HUM       0.92      0.92      0.92        65
             LOC       0.90      0.86      0.88        81
             NUM       1.00      0.88      0.93       113
    
       micro avg       0.89      0.89      0.89       500
       macro avg       0.92      0.87      0.89       500
    weighted avg       0.90      0.89      0.89       500
        """
    _matrix = """
        [[  7   2   0   0   0   0]
         [  0 136   2   0   0   0]
         [  0  11  75   4   4   0]
         [  0   1   3  60   1   0]
         [  0   3   8   0  70   0]
         [  0   9   1   1   3  99]]

        """


class LogRegSentenceClassifier(SentenceClassifier, LogRegTextClassifier):
    # default pipeline - Accuracy: 0
    _accuracy = 0
    _report = """
                         precision    recall  f1-score   support

        """
    _matrix = """
      

        """


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_lr"
    clf = LogRegQuestionClassifier(name)
    name = "main_questions_lr"
    clf = LogRegMainQuestionClassifier(name)
    name = "sentences_lr"
    sent_clf = LogRegSentenceClassifier(name)
    if search:
        print("MAIN_LABEL : SECONDARY_LABEL")
        best_score, best_pipeline = best_pipeline(clf)
        print("BEST:", best_pipeline, "ACCURACY:", best_score)
        print("MAIN LABEL")
        best_score, best_pipeline = best_pipeline(main_clf)
        print("BEST:", best_pipeline, "ACCURACY:", best_score)
        print("QUESTION/SENTENCE")
        best_score, best_pipeline = best_pipeline(sent_clf)
        print("BEST:", best_pipeline, "ACCURACY:", best_score)
        exit(0)

    train_data_path = join(DATA_PATH, "questions.txt")
    test_data_path = join(DATA_PATH, "questions_test.txt")
    if train:
        t, t_label = clf.load_data(train_data_path)
        clf.train(t, t_label)
        clf.save()
    else:
        clf.load()
    from sklearn.metrics import accuracy_score, classification_report, \
        confusion_matrix

    X_test, y_test = clf.load_test_data(test_data_path)
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(accuracy)
    report = classification_report(y_test, preds)
    print(report)
    matrix = confusion_matrix(y_test, preds)
    print(matrix)
