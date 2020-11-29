from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, find_best_pipeline, \
    BaseClassifier

from os.path import join

from sklearn.svm import LinearSVC


class LinearSVCTextClassifier(BaseClassifier):
    def __init__(self, name="linear_svc"):
        super().__init__(name)

    @property
    def classifier_class(self):
        return LinearSVC()

    @property
    def parameters(self):
        return {'features__text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'features__text__tfidf__use_idf': (True, False),
                'clf__probability': (True, False),
                'clf__shrinking': (True, False),
                'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid',
                                'precomputed'),
                'clf__decision_function_shape': ('ovo', 'ovr')}


class LinearSVCQuestionClassifier(QuestionClassifier, LinearSVCTextClassifier):
    # lemmatized text, count vectorizer ngram(1,2) - Accuracy: 0.836
    # lemmatized text, tfidf vectorizer ngram(1,2) - Accuracy: 0.822
    # postag one hot encoder - Accuracy: 0.802
    # default pipeline - Accuracy: 0.838
    _accuracy = 0.838
    _report = """
                     precision    recall  f1-score   support

          ABBR:abb       1.00      1.00      1.00         1
          ABBR:exp       1.00      0.88      0.93         8
          DESC:def       0.83      1.00      0.91       123
         DESC:desc       0.56      0.71      0.63         7
       DESC:manner       1.00      1.00      1.00         2
       DESC:reason       1.00      1.00      1.00         6
       ENTY:animal       0.83      0.62      0.71        16
         ENTY:body       1.00      0.50      0.67         2
        ENTY:color       1.00      1.00      1.00        10
       ENTY:cremat       0.00      0.00      0.00         0
     ENTY:currency       1.00      0.83      0.91         6
       ENTY:dismed       0.00      0.00      0.00         2
        ENTY:event       0.00      0.00      0.00         2
         ENTY:food       0.60      0.75      0.67         4
       ENTY:instru       1.00      1.00      1.00         1
         ENTY:lang       1.00      1.00      1.00         2
        ENTY:other       0.38      0.42      0.40        12
        ENTY:plant       1.00      0.20      0.33         5
      ENTY:product       0.00      0.00      0.00         4
        ENTY:sport       0.50      1.00      0.67         1
    ENTY:substance       1.00      0.33      0.50        15
     ENTY:techmeth       0.50      1.00      0.67         1
       ENTY:termeq       0.55      0.86      0.67         7
          ENTY:veh       1.00      0.50      0.67         4
          HUM:desc       1.00      1.00      1.00         3
            HUM:gr       0.80      0.67      0.73         6
           HUM:ind       0.95      0.98      0.96        55
         HUM:title       0.00      0.00      0.00         1
          LOC:city       1.00      0.78      0.88        18
       LOC:country       1.00      1.00      1.00         3
         LOC:mount       0.67      0.67      0.67         3
         LOC:other       0.76      0.84      0.80        50
         LOC:state       0.70      1.00      0.82         7
         NUM:count       0.90      1.00      0.95         9
          NUM:date       1.00      1.00      1.00        47
          NUM:dist       0.90      0.56      0.69        16
         NUM:money       1.00      0.33      0.50         3
         NUM:other       1.00      0.42      0.59        12
          NUM:perc       0.50      0.67      0.57         3
        NUM:period       0.67      1.00      0.80         8
         NUM:speed       1.00      0.67      0.80         6
          NUM:temp       1.00      0.80      0.89         5
        NUM:weight       0.80      1.00      0.89         4

         micro avg       0.84      0.84      0.84       500
         macro avg       0.75      0.70      0.69       500
      weighted avg       0.85      0.84      0.82       500
    """
    _matrix = """
    [[  1   0   0 ...   0   0   0]
     [  0   7   1 ...   0   0   0]
     [  0   0 123 ...   0   0   0]
     ...
     [  0   0   1 ...   4   0   0]
     [  0   0   0 ...   0   4   0]
     [  0   0   0 ...   0   0   4]]

    """


class LinearSVCMainQuestionClassifier(MainQuestionClassifier,
                                      LinearSVCTextClassifier):
    # default pipeline - Accuracy: 0.902
    _accuracy = 0.902
    _report = """
                     precision    recall  f1-score   support

            ABBR       1.00      0.78      0.88         9
            DESC       0.84      0.99      0.91       138
            ENTY       0.85      0.80      0.82        94
             HUM       0.94      0.95      0.95        65
             LOC       0.92      0.85      0.88        81
             NUM       0.99      0.90      0.94       113

       micro avg       0.90      0.90      0.90       500
       macro avg       0.92      0.88      0.90       500
    weighted avg       0.91      0.90      0.90       500
    """
    _matrix = """
    [[  7   2   0   0   0   0]
     [  0 136   2   0   0   0]
     [  0  12  75   4   3   0]
     [  0   0   3  62   0   0]
     [  0   4   7   0  69   1]
     [  0   7   1   0   3 102]]

    """


class LinearSVCSentenceClassifier(SentenceClassifier, LinearSVCTextClassifier):
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
    from sklearn.metrics import accuracy_score, classification_report, \
        confusion_matrix

    train = True
    search = True
    name = "questions_svc"
    clf = LinearSVCQuestionClassifier(name)
    name = "main_questions_svc"
    main_clf = LinearSVCMainQuestionClassifier(name)
    name = "sentences_svc"
    sent_clf = LinearSVCSentenceClassifier(name)
    if search:
        print("MAIN_LABEL : SECONDARY_LABEL")
        best_score, best_pipeline = find_best_pipeline(clf)
        print("BEST:", best_pipeline, "ACCURACY:", best_score)
        print("MAIN LABEL")
        best_score, best_pipeline = find_best_pipeline(main_clf)
        print("BEST:", best_pipeline, "ACCURACY:", best_score)
        print("QUESTION/SENTENCE")
        best_score, best_pipeline = find_best_pipeline(sent_clf)
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
    X_test, y_test = clf.load_test_data(test_data_path)
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(accuracy)
    report = classification_report(y_test, preds)
    print(report)
    matrix = confusion_matrix(y_test, preds)
    print(matrix)
