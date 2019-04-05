from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.perceptron import PerceptronTextClassifier
from os.path import join


class PerceptronQuestionClassifier(QuestionClassifier,
                                   PerceptronTextClassifier):
    # default pipeline - Accuracy: 0.766
    _accuracy = 0.766
    _report = """
                    precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.62      0.77         8
          DESC:def       0.71      1.00      0.83       123
         DESC:desc       0.71      0.71      0.71         7
       DESC:manner       1.00      1.00      1.00         2
       DESC:reason       1.00      0.83      0.91         6
       ENTY:animal       0.73      0.50      0.59        16
         ENTY:body       1.00      0.50      0.67         2
        ENTY:color       1.00      1.00      1.00        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       0.67      0.33      0.44         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.00      0.00      0.00         4
       ENTY:instru       0.50      1.00      0.67         1
         ENTY:lang       1.00      0.50      0.67         2
        ENTY:other       0.50      0.25      0.33        12
        ENTY:plant       1.00      0.40      0.57         5
      ENTY:product       0.00      0.00      0.00         4
     ENTY:religion       0.00      0.00      0.00         0
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       0.80      0.27      0.40        15
       ENTY:symbol       0.00      0.00      0.00         0
     ENTY:techmeth       0.25      1.00      0.40         1
       ENTY:termeq       0.50      0.71      0.59         7
          ENTY:veh       0.67      0.50      0.57         4
         ENTY:word       0.00      0.00      0.00         0
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.33      0.67      0.44         6
           HUM:ind       1.00      0.91      0.95        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.82      0.78      0.80        18
       LOC:country       0.75      1.00      0.86         3
         LOC:mount       1.00      0.67      0.80         3
         LOC:other       0.86      0.76      0.81        50
         LOC:state       0.67      0.57      0.62         7
         NUM:count       1.00      1.00      1.00         9
          NUM:date       0.92      0.96      0.94        47
          NUM:dist       0.89      0.50      0.64        16
         NUM:money       0.33      0.33      0.33         3
         NUM:other       0.86      0.50      0.63        12
          NUM:perc       1.00      0.33      0.50         3
        NUM:period       0.62      1.00      0.76         8
         NUM:speed       0.80      0.67      0.73         6
          NUM:temp       1.00      0.20      0.33         5
        NUM:weight       0.00      0.00      0.00         4
    
         micro avg       0.77      0.77      0.77       500
         macro avg       0.62      0.54      0.54       500
      weighted avg       0.78      0.77      0.75       500
        """
    _matrix = """
        [[  1   0   0 ...   0   0   0]
         [  0   5   3 ...   0   0   0]
         [  0   0 123 ...   0   0   0]
         ...
         [  0   0   1 ...   4   0   0]
         [  0   0   1 ...   1   1   0]
         [  0   0   1 ...   0   0   0]]

        """


class PerceptronMainQuestionClassifier(MainQuestionClassifier,
                                       PerceptronTextClassifier):
    # default pipeline - Accuracy: 0.872
    _accuracy = 0.872
    _report = """
                  precision    recall  f1-score   support
    
            ABBR       0.80      0.89      0.84         9
            DESC       0.92      0.97      0.94       138
            ENTY       0.88      0.56      0.69        94
             HUM       0.81      0.95      0.87        65
             LOC       0.82      0.90      0.86        81
             NUM       0.90      0.94      0.92       113
    
       micro avg       0.87      0.87      0.87       500
       macro avg       0.85      0.87      0.85       500
    weighted avg       0.87      0.87      0.87       500
        """
    _matrix = """
        [[  8   1   0   0   0   0]
         [  0 134   3   0   0   1]
         [  2   5  53  14  12   8]
         [  0   0   2  62   1   0]
         [  0   2   2   1  73   3]
         [  0   4   0   0   3 106]]

        """


class PerceptronSentenceClassifier(SentenceClassifier,
                                   PerceptronTextClassifier):
    # default pipeline - Accuracy: 0.7333333333333333
    _accuracy = 0.7333333333333333
    _report = """
                  precision    recall  f1-score   support
    
         command       0.75      0.60      0.67         5
        question       0.67      0.80      0.73         5
       statement       0.80      0.80      0.80         5
    
       micro avg       0.73      0.73      0.73        15
       macro avg       0.74      0.73      0.73        15
    weighted avg       0.74      0.73      0.73        15
        """
    _matrix = """
       [[3 2 0]
         [0 4 1]
         [1 0 4]]

        """


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_perceptron"
    clf = PerceptronQuestionClassifier(name)
    name = "main_questions_perceptron"
    main_clf = PerceptronMainQuestionClassifier(name)
    name = "sentences_perceptron"
    sent_clf = PerceptronSentenceClassifier(name)
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
