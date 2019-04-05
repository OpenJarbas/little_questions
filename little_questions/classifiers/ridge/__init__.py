from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.ridge import RidgeTextClassifier
from os.path import join


class RidgeQuestionClassifier(QuestionClassifier,  RidgeTextClassifier):
    # default pipeline - Accuracy: 0.834
    _accuracy = 0.834
    _report = """
                       precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.62      0.77         8
          DESC:def       0.77      1.00      0.87       123
         DESC:desc       1.00      0.71      0.83         7
       DESC:manner       0.67      1.00      0.80         2
       DESC:reason       1.00      1.00      1.00         6
       ENTY:animal       0.91      0.62      0.74        16
         ENTY:body       0.67      1.00      0.80         2
        ENTY:color       1.00      1.00      1.00        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       0.83      0.83      0.83         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       1.00      0.75      0.86         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
        ENTY:other       0.45      0.42      0.43        12
        ENTY:plant       1.00      0.20      0.33         5
      ENTY:product       1.00      0.25      0.40         4
        ENTY:sport       1.00      1.00      1.00         1
    ENTY:substance       1.00      0.40      0.57        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.67      0.86      0.75         7
          ENTY:veh       1.00      0.50      0.67         4
          HUM:desc       1.00      0.67      0.80         3
            HUM:gr       0.80      0.67      0.73         6
           HUM:ind       0.90      0.96      0.93        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.93      0.78      0.85        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.67      0.67      0.67         3
         LOC:other       0.78      0.80      0.79        50
         LOC:state       0.58      1.00      0.74         7
         NUM:count       0.82      1.00      0.90         9
          NUM:date       1.00      0.98      0.99        47
          NUM:dist       1.00      0.56      0.72        16
         NUM:money       1.00      0.33      0.50         3
         NUM:other       1.00      0.42      0.59        12
          NUM:perc       0.60      1.00      0.75         3
        NUM:period       0.80      1.00      0.89         8
         NUM:speed       1.00      0.83      0.91         6
          NUM:temp       1.00      0.80      0.89         5
        NUM:weight       0.80      1.00      0.89         4
    
         micro avg       0.83      0.83      0.83       500
         macro avg       0.81      0.71      0.73       500
      weighted avg       0.85      0.83      0.82       500
       """
    _matrix = """
       [[  1   0   0 ...   0   0   0]
         [  0   5   3 ...   0   0   0]
         [  0   0 123 ...   0   0   0]
         ...
         [  0   0   1 ...   5   0   0]
         [  0   0   0 ...   0   4   0]
         [  0   0   0 ...   0   0   4]]

       """


class RidgeMainQuestionClassifier(MainQuestionClassifier, RidgeTextClassifier):
    # default pipeline - Accuracy: 0.896
    _accuracy = 0.896
    _report = """
                   precision    recall  f1-score   support
    
            ABBR       1.00      0.78      0.88         9
            DESC       0.82      0.99      0.90       138
            ENTY       0.87      0.76      0.81        94
             HUM       0.94      0.94      0.94        65
             LOC       0.90      0.88      0.89        81
             NUM       1.00      0.90      0.95       113
    
       micro avg       0.90      0.90      0.90       500
       macro avg       0.92      0.87      0.89       500
    weighted avg       0.90      0.90      0.90       500
       """
    _matrix = """
           
    [[  7   2   0   0   0   0]
     [  0 136   2   0   0   0]
     [  0  15  71   3   5   0]
     [  0   1   2  61   1   0]
     [  0   4   6   0  71   0]
     [  0   7   1   1   2 102]]

       """


class RidgeSentenceClassifier(SentenceClassifier,  RidgeTextClassifier):
    # default pipeline - Accuracy: 0.6666666666666666
    _accuracy = 0.6666666666666666
    _report = """
                  precision    recall  f1-score   support
    
         command       1.00      0.20      0.33         5
        question       0.80      0.80      0.80         5
       statement       0.56      1.00      0.71         5
    
       micro avg       0.67      0.67      0.67        15
       macro avg       0.79      0.67      0.62        15
    weighted avg       0.79      0.67      0.62        15
       """
    _matrix = """
        [[1 1 3]
         [0 4 1]
         [0 0 5]]
       """


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_ridge"
    clf = RidgeQuestionClassifier(name)
    name = "main_questions_ridge"
    main_clf = RidgeMainQuestionClassifier(name)
    name = "sentences_ridge"
    sent_clf = RidgeSentenceClassifier(name)
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
