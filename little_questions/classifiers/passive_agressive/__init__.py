from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.passive_agressive import \
    PassiveAggressiveTextClassifier
from os.path import join


class PassiveAggressiveQuestionClassifier(QuestionClassifier,
                                          PassiveAggressiveTextClassifier):
    # default pipeline - Accuracy: 0.804
    _accuracy = 0.804
    _report = """
                  precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       0.86      0.75      0.80         8
          DESC:def       0.81      0.99      0.89       123
         DESC:desc       0.45      0.71      0.56         7
       DESC:manner       1.00      1.00      1.00         2
       DESC:reason       1.00      0.83      0.91         6
       ENTY:animal       0.67      0.38      0.48        16
         ENTY:body       0.67      1.00      0.80         2
        ENTY:color       1.00      0.90      0.95        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       1.00      0.33      0.50         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.75      0.75      0.75         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
       ENTY:letter       0.00      0.00      0.00         0
        ENTY:other       0.42      0.42      0.42        12
        ENTY:plant       1.00      0.20      0.33         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       1.00      0.33      0.50        15
     ENTY:techmeth       0.20      1.00      0.33         1
       ENTY:termeq       0.54      1.00      0.70         7
          ENTY:veh       1.00      0.25      0.40         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.75      0.50      0.60         6
           HUM:ind       0.93      0.95      0.94        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.93      0.78      0.85        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       1.00      0.67      0.80         3
         LOC:other       0.77      0.82      0.80        50
         LOC:state       0.60      0.86      0.71         7
         NUM:count       0.82      1.00      0.90         9
          NUM:date       1.00      0.98      0.99        47
          NUM:dist       0.90      0.56      0.69        16
         NUM:money       0.50      0.33      0.40         3
         NUM:other       1.00      0.42      0.59        12
          NUM:perc       0.50      0.67      0.57         3
        NUM:period       0.67      1.00      0.80         8
         NUM:speed       1.00      0.67      0.80         6
          NUM:temp       1.00      0.80      0.89         5
        NUM:weight       0.75      0.75      0.75         4
    
         micro avg       0.80      0.80      0.80       500
         macro avg       0.70      0.65      0.64       500
      weighted avg       0.83      0.80      0.79       500
        """
    _matrix = """
        [[  1   0   0 ...   0   0   0]
         [  0   6   2 ...   0   0   0]
         [  0   1 122 ...   0   0   0]
         ...
         [  0   0   1 ...   4   0   0]
         [  0   0   0 ...   0   4   0]
         [  0   0   0 ...   0   0   3]]

        """


class PassiveAggressiveMainQuestionClassifier(MainQuestionClassifier,
                                              PassiveAggressiveTextClassifier):
    # default pipeline - Accuracy: 0.882
    _accuracy = 0.882
    _report = """
                  precision    recall  f1-score   support
    
            ABBR       1.00      0.78      0.88         9
            DESC       0.82      0.99      0.89       138
            ENTY       0.84      0.72      0.78        94
             HUM       0.90      0.95      0.93        65
             LOC       0.90      0.85      0.87        81
             NUM       0.99      0.88      0.93       113
    
       micro avg       0.88      0.88      0.88       500
       macro avg       0.91      0.86      0.88       500
    weighted avg       0.89      0.88      0.88       500
        """
    _matrix = """
        [[  7   2   0   0   0   0]
         [  0 136   2   0   0   0]
         [  0  15  68   7   4   0]
         [  0   1   2  62   0   0]
         [  0   3   8   0  69   1]
         [  0   9   1   0   4  99]]


        """


class PassiveAggressiveSentenceClassifier(SentenceClassifier,
                                          PassiveAggressiveTextClassifier):
    # default pipeline - Accuracy: 0.8666666666666667
    _accuracy = 0.8666666666666667
    _report = """
                  precision    recall  f1-score   support
    
         command       1.00      0.80      0.89         5
        question       0.80      0.80      0.80         5
       statement       0.83      1.00      0.91         5
    
       micro avg       0.87      0.87      0.87        15
       macro avg       0.88      0.87      0.87        15
    weighted avg       0.88      0.87      0.87        15
        """
    _matrix = """
        [[4 1 0]
         [0 4 1]
         [0 0 5]]


        """


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_pa"
    clf = PassiveAggressiveQuestionClassifier(name)
    name = "main_questions_pa"
    main_clf = PassiveAggressiveMainQuestionClassifier(name)
    name = "sentences_pa"
    sent_clf = PassiveAggressiveSentenceClassifier(name)

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
