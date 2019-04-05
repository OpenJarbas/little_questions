from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.gradboost import \
    GradientBoostingTextClassifier
from little_questions.settings import DATA_PATH
from os.path import join


class GradientBoostingQuestionClassifier(QuestionClassifier,
                                         GradientBoostingTextClassifier):
    # default pipeline - Accuracy: 0.776
    _accuracy = 0.776
    _report = """
                    precision    recall  f1-score   support
    
          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.75      0.86         8
          DESC:def       0.88      0.98      0.93       123
         DESC:desc       0.80      0.57      0.67         7
       DESC:manner       0.67      1.00      0.80         2
       DESC:reason       1.00      0.83      0.91         6
       ENTY:animal       1.00      0.50      0.67        16
         ENTY:body       0.00      0.00      0.00         2
        ENTY:color       1.00      1.00      1.00        10
     ENTY:currency       0.83      0.83      0.83         6
       ENTY:dismed       0.50      0.50      0.50         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       1.00      0.25      0.40         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       0.00      0.00      0.00         2
        ENTY:other       0.16      0.67      0.26        12
        ENTY:plant       0.00      0.00      0.00         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       1.00      1.00      1.00         1
    ENTY:substance       0.83      0.33      0.48        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.83      0.71      0.77         7
          ENTY:veh       1.00      0.25      0.40         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.33      0.17      0.22         6
           HUM:ind       0.67      0.95      0.78        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.94      0.83      0.88        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       1.00      0.67      0.80         3
         LOC:other       0.84      0.74      0.79        50
         LOC:state       0.67      0.86      0.75         7
         NUM:count       0.69      1.00      0.82         9
          NUM:date       1.00      0.96      0.98        47
          NUM:dist       0.89      0.50      0.64        16
         NUM:money       0.00      0.00      0.00         3
         NUM:other       1.00      0.08      0.15        12
          NUM:perc       1.00      0.67      0.80         3
        NUM:period       0.75      0.75      0.75         8
         NUM:speed       1.00      0.83      0.91         6
          NUM:temp       1.00      0.60      0.75         5
       NUM:volsize       0.00      0.00      0.00         0
        NUM:weight       0.80      1.00      0.89         4
    
         micro avg       0.78      0.78      0.78       500
         macro avg       0.70      0.60      0.61       500
      weighted avg       0.81      0.78      0.76       500
        """
    _matrix = """
        [[  1   0   0 ...   0   0   0]
         [  0   6   1 ...   0   0   0]
         [  0   0 121 ...   0   0   0]
         ...
         [  0   0   1 ...   3   0   0]
         [  0   0   0 ...   0   0   0]
         [  0   0   0 ...   0   0   4]]

        """


class GradientBoostingMainQuestionClassifier(MainQuestionClassifier,
                                             GradientBoostingTextClassifier):
    # default pipeline - Accuracy: 0.858
    _accuracy = 0.858
    _report = """
                  precision    recall  f1-score   support
    
            ABBR       0.89      0.89      0.89         9
            DESC       0.90      0.96      0.93       138
            ENTY       0.67      0.79      0.72        94
             HUM       0.88      0.88      0.88        65
             LOC       0.86      0.77      0.81        81
             NUM       1.00      0.85      0.92       113
    
       micro avg       0.86      0.86      0.86       500
       macro avg       0.87      0.85      0.86       500
    weighted avg       0.87      0.86      0.86       500
        """
    _matrix = """
        [[  8   0   1   0   0   0]
         [  0 132   6   0   0   0]
         [  0   5  74   8   7   0]
         [  1   1   5  57   1   0]
         [  0   5  14   0  62   0]
         [  0   4  11   0   2  96]]

        """


class GradientBoostingSentenceClassifier(SentenceClassifier,
                                         GradientBoostingTextClassifier):
    # default pipeline - Accuracy: ?
    _accuracy = 0
    _report = """
           
        """
    _matrix = """

        """


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_gradboost"
    clf = GradientBoostingQuestionClassifier(name)
    name = "main_questions_gradboost"
    main_clf = GradientBoostingMainQuestionClassifier(name)
    name = "sentences_gradboost"
    sent_clf = GradientBoostingSentenceClassifier(name)
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
