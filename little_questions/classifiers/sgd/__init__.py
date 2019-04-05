from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.sgd import SGDTextClassifier
from os.path import join


class SGDQuestionClassifier(QuestionClassifier, SGDTextClassifier):
    # default pipeline - Accuracy: 0.8666666666666667
    _accuracy = 0.802
    _report = """
                      precision    recall  f1-score   support
    
          ABBR:abb       0.20      1.00      0.33         1
          ABBR:exp       0.30      0.88      0.45         8
          DESC:def       0.97      0.98      0.98       123
         DESC:desc       0.46      0.86      0.60         7
       DESC:manner       0.67      1.00      0.80         2
       DESC:reason       1.00      0.83      0.91         6
       ENTY:animal       0.86      0.38      0.52        16
         ENTY:body       1.00      0.50      0.67         2
        ENTY:color       0.83      1.00      0.91        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       1.00      0.67      0.80         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.60      0.75      0.67         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
        ENTY:other       0.50      0.33      0.40        12
        ENTY:plant       1.00      0.20      0.33         5
      ENTY:product       0.50      0.25      0.33         4
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       0.86      0.40      0.55        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.47      1.00      0.64         7
          ENTY:veh       0.67      0.50      0.57         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       1.00      0.33      0.50         6
           HUM:ind       0.88      0.95      0.91        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.79      0.83      0.81        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.67      0.67      0.67         3
         LOC:other       0.85      0.70      0.77        50
         LOC:state       0.58      1.00      0.74         7
         NUM:count       1.00      0.89      0.94         9
          NUM:date       0.98      1.00      0.99        47
          NUM:dist       1.00      0.56      0.72        16
         NUM:money       0.17      0.67      0.27         3
           NUM:ord       0.00      0.00      0.00         0
         NUM:other       1.00      0.42      0.59        12
          NUM:perc       1.00      0.67      0.80         3
        NUM:period       0.62      1.00      0.76         8
         NUM:speed       1.00      0.67      0.80         6
          NUM:temp       1.00      0.60      0.75         5
       NUM:volsize       0.00      0.00      0.00         0
        NUM:weight       1.00      0.50      0.67         4
    
         micro avg       0.80      0.80      0.80       500
         macro avg       0.69      0.64      0.62       500
      weighted avg       0.86      0.80      0.80       500
        """
    _matrix = """
            [[  1   0   0 ...   0   0   0]
             [  0   7   1 ...   0   0   0]
             [  0   2 121 ...   0   0   0]
             ...
             [  0   0   0 ...   3   0   0]
             [  0   0   0 ...   0   0   0]
             [  0   0   0 ...   0   0   2]]

    """


class SGDMainQuestionClassifier(MainQuestionClassifier, SGDTextClassifier):
    # default pipeline - Accuracy: 0.888
    _accuracy = 0.888
    _report = """
                     precision    recall  f1-score   support
    
            ABBR       0.58      0.78      0.67         9
            DESC       0.88      0.99      0.93       138
            ENTY       0.88      0.67      0.76        94
             HUM       0.92      0.94      0.93        65
             LOC       0.82      0.91      0.87        81
             NUM       0.98      0.91      0.94       113
    
       micro avg       0.89      0.89      0.89       500
       macro avg       0.84      0.87      0.85       500
    weighted avg       0.89      0.89      0.89       500
        """
    _matrix = """
            [[  7   2   0   0   0   0]
             [  1 136   1   0   0   0]
             [  3  10  63   5  11   2]
             [  0   0   2  61   2   0]
             [  1   1   5   0  74   0]
             [  0   6   1   0   3 103]]

        """


class SGDSentenceClassifier(SentenceClassifier, SGDTextClassifier):
    # default pipeline - Accuracy: 0.5333333333333333
    _accuracy = 0.5333333333333333
    _report = """
                    precision    recall  f1-score   support
    
         command       0.00      0.00      0.00         5
        question       0.67      0.80      0.73         5
       statement       0.44      0.80      0.57         5
    
       micro avg       0.53      0.53      0.53        15
       macro avg       0.37      0.53      0.43        15
    weighted avg       0.37      0.53      0.43        15
        """
    _matrix = """
            [[0 1 4]
             [0 4 1]
             [0 1 4]]

        """


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_sgd"
    clf = SGDQuestionClassifier(name)
    name = "main_questions_sgd"
    main_clf = SGDMainQuestionClassifier(name)
    name = "sentences_sgd"
    sent_clf = SGDSentenceClassifier(name)
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
