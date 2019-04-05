from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.forest import ForestTextClassifier
from os.path import join


class ForestQuestionClassifier(QuestionClassifier, ForestTextClassifier):
    # default pipeline - Accuracy: 0.636
    _accuracy = 0.636
    _report = """
                           precision    recall  f1-score   support
        
              ABBR:abb       1.00      1.00      1.00         1
              ABBR:exp       1.00      1.00      1.00         8
              DESC:def       0.65      1.00      0.79       123
             DESC:desc       0.18      0.43      0.25         7
           DESC:manner       0.11      1.00      0.20         2
           DESC:reason       0.80      0.67      0.73         6
           ENTY:animal       1.00      0.06      0.12        16
             ENTY:body       0.00      0.00      0.00         2
            ENTY:color       1.00      0.70      0.82        10
         ENTY:currency       1.00      0.33      0.50         6
           ENTY:dismed       0.00      0.00      0.00         2
            ENTY:event       0.00      0.00      0.00         2
             ENTY:food       0.00      0.00      0.00         4
           ENTY:instru       0.00      0.00      0.00         1
             ENTY:lang       0.00      0.00      0.00         2
            ENTY:other       0.33      0.17      0.22        12
            ENTY:plant       0.00      0.00      0.00         5
          ENTY:product       0.00      0.00      0.00         4
            ENTY:sport       0.00      0.00      0.00         1
        ENTY:substance       0.00      0.00      0.00        15
         ENTY:techmeth       1.00      1.00      1.00         1
           ENTY:termeq       0.33      0.14      0.20         7
              ENTY:veh       0.00      0.00      0.00         4
              HUM:desc       1.00      0.67      0.80         3
                HUM:gr       0.00      0.00      0.00         6
               HUM:ind       0.50      0.95      0.66        55
             HUM:title       0.00      0.00      0.00         1
              LOC:city       1.00      0.50      0.67        18
           LOC:country       0.50      0.67      0.57         3
             LOC:mount       1.00      0.33      0.50         3
             LOC:other       0.67      0.72      0.69        50
             LOC:state       0.75      0.43      0.55         7
             NUM:count       0.75      1.00      0.86         9
              NUM:date       1.00      0.87      0.93        47
              NUM:dist       0.50      0.06      0.11        16
             NUM:money       0.00      0.00      0.00         3
             NUM:other       1.00      0.42      0.59        12
              NUM:perc       0.00      0.00      0.00         3
            NUM:period       0.67      0.25      0.36         8
             NUM:speed       0.00      0.00      0.00         6
              NUM:temp       0.00      0.00      0.00         5
            NUM:weight       0.00      0.00      0.00         4
        
             micro avg       0.64      0.64      0.64       500
             macro avg       0.42      0.34      0.34       500
          weighted avg       0.61      0.64      0.57       500
        """
    _matrix = """
        [[  1   0   0 ...   0   0   0]
         [  0   8   0 ...   0   0   0]
         [  0   0 123 ...   0   0   0]
         ...
         [  0   0   3 ...   0   0   0]
         [  0   0   3 ...   0   0   0]
         [  0   0   1 ...   0   0   0]]


        """


class ForestMainQuestionClassifier(MainQuestionClassifier,
                                     ForestTextClassifier):
    # default pipeline - Accuracy: 0.798
    _accuracy = 0.798
    _report = """
                   precision    recall  f1-score   support
    
            ABBR       1.00      0.89      0.94         9
            DESC       0.79      0.99      0.87       138
            ENTY       0.62      0.71      0.66        94
             HUM       0.81      0.86      0.84        65
             LOC       0.84      0.67      0.74        81
             NUM       1.00      0.69      0.82       113
    
       micro avg       0.80      0.80      0.80       500
       macro avg       0.84      0.80      0.81       500
    weighted avg       0.82      0.80      0.80       500
        """
    _matrix = """
        [[  8   1   0   0   0   0]
         [  0 136   2   0   0   0]
         [  0  15  67   6   6   0]
         [  0   1   6  56   2   0]
         [  0   8  16   3  54   0]
         [  0  12  17   4   2  78]]

        """


class ForestSentenceClassifier(SentenceClassifier, ForestTextClassifier):
    # default pipeline - Accuracy: ?
    _accuracy = 0
    _report = """
        """
    _matrix = """

        """


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_forest"
    clf = ForestQuestionClassifier(name)
    name = "main_questions_forest"
    main_clf = ForestMainQuestionClassifier(name)
    name = "sentences_forest"
    sent_clf = ForestSentenceClassifier(name)
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
