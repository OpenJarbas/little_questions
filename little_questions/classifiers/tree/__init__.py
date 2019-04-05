from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.tree import ExtraTreeTextClassifier, \
    TreeTextClassifier
from os.path import join


class TreeQuestionClassifier(QuestionClassifier, TreeTextClassifier):
    # default pipeline - Accuracy: 0.666
    _accuracy = 0.666
    _report = """
                   precision    recall  f1-score   support
    
          ABBR:abb       0.50      1.00      0.67         1
          ABBR:exp       0.86      0.75      0.80         8
          DESC:def       0.90      0.92      0.91       123
         DESC:desc       0.21      0.43      0.29         7
       DESC:manner       0.50      1.00      0.67         2
       DESC:reason       0.71      0.83      0.77         6
       ENTY:animal       0.21      0.19      0.20        16
         ENTY:body       0.50      0.50      0.50         2
        ENTY:color       1.00      0.90      0.95        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       0.00      0.00      0.00         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.29      0.50      0.36         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      0.50      0.67         2
       ENTY:letter       0.00      0.00      0.00         0
        ENTY:other       0.19      0.25      0.21        12
        ENTY:plant       0.00      0.00      0.00         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       1.00      1.00      1.00         1
    ENTY:substance       0.75      0.20      0.32        15
     ENTY:techmeth       1.00      1.00      1.00         1
       ENTY:termeq       0.11      0.14      0.12         7
          ENTY:veh       0.00      0.00      0.00         4
         ENTY:word       0.00      0.00      0.00         0
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.17      0.33      0.22         6
           HUM:ind       0.77      0.91      0.83        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       1.00      0.67      0.80        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.33      0.33      0.33         3
         LOC:other       0.71      0.60      0.65        50
         LOC:state       0.60      0.43      0.50         7
          NUM:code       0.00      0.00      0.00         0
         NUM:count       0.80      0.89      0.84         9
          NUM:date       0.93      0.91      0.92        47
          NUM:dist       0.70      0.44      0.54        16
         NUM:money       0.33      0.33      0.33         3
         NUM:other       0.50      0.42      0.45        12
          NUM:perc       0.33      0.33      0.33         3
        NUM:period       0.60      0.75      0.67         8
         NUM:speed       1.00      0.17      0.29         6
          NUM:temp       0.00      0.00      0.00         5
       NUM:volsize       0.00      0.00      0.00         0
        NUM:weight       0.50      0.25      0.33         4
    
         micro avg       0.67      0.67      0.67       500
         macro avg       0.47      0.44      0.44       500
      weighted avg       0.70      0.67      0.67       500
        """
    _matrix = """
        [[  1   0   0 ...   0   0   0]
         [  0   6   1 ...   0   0   0]
         [  0   1 113 ...   1   0   0]
         ...
         [  0   0   1 ...   0   0   0]
         [  0   0   0 ...   0   0   0]
         [  0   0   0 ...   0   0   1]]

        """


class TreeMainQuestionClassifier(MainQuestionClassifier, TreeTextClassifier):
    # default pipeline - Accuracy: 0.784
    _accuracy = 0.784
    _report = """
                      precision    recall  f1-score   support
        
                ABBR       0.73      0.89      0.80         9
                DESC       0.84      0.92      0.88       138
                ENTY       0.65      0.56      0.61        94
                 HUM       0.68      0.83      0.75        65
                 LOC       0.80      0.73      0.76        81
                 NUM       0.88      0.81      0.84       113
        
           micro avg       0.78      0.78      0.78       500
           macro avg       0.76      0.79      0.77       500
        weighted avg       0.78      0.78      0.78       500
        """
    _matrix = """
        [[  8   1   0   0   0   0]
         [  3 127   5   1   1   1]
         [  0  13  53  14   7   7]
         [  0   0   8  54   2   1]
         [  0   4   8   6  59   4]
         [  0   6   7   4   5  91]]

        """


class TreeSentenceClassifier(SentenceClassifier, TreeTextClassifier):
    # default pipeline - Accuracy: 0.8666666666666667
    _accuracy = 0.8666666666666667
    _report = """
                    precision    recall  f1-score   support
    
         command       1.00      0.80      0.89         5
        question       1.00      0.80      0.89         5
       statement       0.71      1.00      0.83         5
    
       micro avg       0.87      0.87      0.87        15
       macro avg       0.90      0.87      0.87        15
    weighted avg       0.90      0.87      0.87        15
    """
    _matrix = """
    [[4 0 1]
     [0 4 1]
     [0 0 5]]
    
    """


class ExtraTreeQuestionClassifier(QuestionClassifier, ExtraTreeTextClassifier):
    # default pipeline - Accuracy: 0.548
    _accuracy = 0.548
    _report = """
                   precision    recall  f1-score   support
    
          ABBR:abb       0.00      0.00      0.00         1
          ABBR:exp       0.60      0.75      0.67         8
          DESC:def       0.81      0.87      0.84       123
         DESC:desc       0.17      0.29      0.21         7
       DESC:manner       0.33      1.00      0.50         2
       DESC:reason       0.21      0.50      0.30         6
       ENTY:animal       0.33      0.31      0.32        16
         ENTY:body       0.00      0.00      0.00         2
        ENTY:color       1.00      0.60      0.75        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       0.67      0.33      0.44         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.17      0.25      0.20         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       0.00      0.00      0.00         2
       ENTY:letter       0.00      0.00      0.00         0
        ENTY:other       0.18      0.17      0.17        12
        ENTY:plant       0.00      0.00      0.00         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       0.00      0.00      0.00         1
    ENTY:substance       0.00      0.00      0.00        15
       ENTY:symbol       0.00      0.00      0.00         0
     ENTY:techmeth       0.00      0.00      0.00         1
       ENTY:termeq       0.00      0.00      0.00         7
          ENTY:veh       0.00      0.00      0.00         4
         ENTY:word       0.00      0.00      0.00         0
          HUM:desc       0.50      1.00      0.67         3
            HUM:gr       0.00      0.00      0.00         6
           HUM:ind       0.64      0.80      0.71        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       0.35      0.39      0.37        18
       LOC:country       0.08      0.33      0.13         3
         LOC:mount       0.50      0.67      0.57         3
         LOC:other       0.61      0.54      0.57        50
         LOC:state       0.00      0.00      0.00         7
         NUM:count       0.44      0.78      0.56         9
          NUM:date       0.82      0.66      0.73        47
          NUM:dist       0.78      0.44      0.56        16
         NUM:money       0.40      0.67      0.50         3
         NUM:other       0.67      0.17      0.27        12
          NUM:perc       0.00      0.00      0.00         3
        NUM:period       0.67      0.25      0.36         8
         NUM:speed       0.00      0.00      0.00         6
          NUM:temp       1.00      0.40      0.57         5
        NUM:weight       0.00      0.00      0.00         4
    
         micro avg       0.55      0.55      0.55       500
         macro avg       0.28      0.29      0.26       500
      weighted avg       0.56      0.55      0.54       500
        """
    _matrix = """
        [[  0   1   0 ...   0   0   0]
         [  0   6   2 ...   0   0   0]
         [  0   0 107 ...   0   0   0]
         ...
         [  0   0   0 ...   0   0   0]
         [  0   0   0 ...   0   2   0]
         [  0   0   0 ...   0   0   0]]

        """


class ExtraTreeMainQuestionClassifier(MainQuestionClassifier,
                                      ExtraTreeTextClassifier):
    # default pipeline - Accuracy: ?
    _accuracy = 0
    _report = """
                         precision    recall  f1-score   support

        """
    _matrix = """

        """


class ExtraTreeSentenceClassifier(SentenceClassifier, ExtraTreeTextClassifier):
    # default pipeline - Accuracy: ?
    _accuracy = 0
    _report = """
                         precision    recall  f1-score   support

               
        """
    _matrix = """
       

        """


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_tree"
    clf = TreeQuestionClassifier(name)
    name = "main_questions_tree"
    main_clf = TreeMainQuestionClassifier(name)
    name = "sentences_tree"
    sent_clf = TreeSentenceClassifier(name)
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
