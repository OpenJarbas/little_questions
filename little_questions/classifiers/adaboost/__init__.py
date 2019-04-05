from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.adaboost import AdaBoostTextClassifier
from os.path import join


class AdaBoostQuestionClassifier(QuestionClassifier, AdaBoostTextClassifier):
    # default pipeline - Accuracy: 0.22
    _accuracy = 0.22
    _report = """
                        precision    recall  f1-score   support
        
              ABBR:abb       0.00      0.00      0.00         1
              ABBR:exp       0.00      0.00      0.00         8
              DESC:def       0.00      0.00      0.00       123
             DESC:desc       0.00      0.00      0.00         7
           DESC:manner       0.00      0.00      0.00         2
           DESC:reason       0.00      0.00      0.00         6
           ENTY:animal       0.00      0.00      0.00        16
             ENTY:body       0.00      0.00      0.00         2
            ENTY:color       1.00      0.80      0.89        10
         ENTY:currency       0.00      0.00      0.00         6
           ENTY:dismed       0.00      0.00      0.00         2
            ENTY:event       0.00      0.00      0.00         2
             ENTY:food       0.00      0.00      0.00         4
           ENTY:instru       1.00      1.00      1.00         1
             ENTY:lang       0.00      0.00      0.00         2
            ENTY:other       0.00      0.00      0.00        12
            ENTY:plant       0.00      0.00      0.00         5
          ENTY:product       0.00      0.00      0.00         4
            ENTY:sport       0.00      0.00      0.00         1
        ENTY:substance       0.00      0.00      0.00        15
         ENTY:techmeth       0.00      0.00      0.00         1
           ENTY:termeq       0.00      0.00      0.00         7
              ENTY:veh       0.00      0.00      0.00         4
              HUM:desc       0.00      0.00      0.00         3
                HUM:gr       0.00      0.00      0.00         6
               HUM:ind       0.94      0.80      0.86        55
             HUM:title       0.00      0.00      0.00         1
              LOC:city       0.00      0.00      0.00        18
           LOC:country       0.00      0.00      0.00         3
             LOC:mount       0.00      0.00      0.00         3
             LOC:other       0.11      1.00      0.21        50
             LOC:state       0.00      0.00      0.00         7
             NUM:count       1.00      0.78      0.88         9
              NUM:date       0.00      0.00      0.00        47
              NUM:dist       0.00      0.00      0.00        16
             NUM:money       0.00      0.00      0.00         3
             NUM:other       0.00      0.00      0.00        12
              NUM:perc       0.00      0.00      0.00         3
            NUM:period       0.00      0.00      0.00         8
             NUM:speed       0.00      0.00      0.00         6
              NUM:temp       0.00      0.00      0.00         5
            NUM:weight       0.00      0.00      0.00         4
        
             micro avg       0.22      0.22      0.22       500
             macro avg       0.10      0.10      0.09       500
          weighted avg       0.15      0.22      0.15       500
        """
    _matrix = """
        [[0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]
         ...
         [0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]
         [0 0 0 ... 0 0 0]]

        """


class AdaBoostMainQuestionClassifier(MainQuestionClassifier,
                                     AdaBoostTextClassifier):
    # default pipeline - Accuracy: 0.592
    _accuracy = 0.592
    _report = """
                      precision    recall  f1-score   support
        
                ABBR       1.00      0.78      0.88         9
                DESC       0.65      0.72      0.68       138
                ENTY       0.36      0.76      0.49        94
                 HUM       0.75      0.68      0.71        65
                 LOC       0.76      0.32      0.45        81
                 NUM       0.98      0.42      0.59       113
        
           micro avg       0.59      0.59      0.59       500
           macro avg       0.75      0.61      0.63       500
        weighted avg       0.71      0.59      0.60       500
        """
    _matrix = """
        [[  7   2   0   0   0   0]
         [  0 100  37   0   0   1]
         [  0  15  71   5   3   0]
         [  0   5  16  44   0   0]
         [  0  16  35   4  26   0]
         [  0  16  38   6   5  48]]


        """


class AdaBoostSentenceClassifier(SentenceClassifier, AdaBoostTextClassifier):
    # default pipeline - Accuracy: 0.4666666666666667
    _accuracy = 0.4666666666666667
    _report = """
                         precision    recall  f1-score   support

             command       0.00      0.00      0.00         5
            question       0.67      0.40      0.50         5
           statement       0.42      1.00      0.59         5
        
           micro avg       0.47      0.47      0.47        15
           macro avg       0.36      0.47      0.36        15
        weighted avg       0.36      0.47      0.36        15

        """
    _matrix = """
        [[0 1 4]
         [0 2 3]
         [0 0 5]]

        """


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_adaboost"
    clf = AdaBoostQuestionClassifier(name)
    name = "main_questions_adaboost"
    main_clf = AdaBoostMainQuestionClassifier(name)
    name = "sentences_adaboost"
    sent_clf = AdaBoostSentenceClassifier(name)
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
