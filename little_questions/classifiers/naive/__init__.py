from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from little_questions.classifiers.pipelines import default_pipeline2
from text_classifikation.classifiers.naive import NaiveTextClassifier
from os.path import join


class NaiveQuestionClassifier(QuestionClassifier, NaiveTextClassifier):
    # default pipeline 2 - Accuracy: 0.53
    _accuracy = 0.53
    _report = """
                           precision    recall  f1-score   support
        
              ABBR:abb       0.00      0.00      0.00         1
              ABBR:exp       0.00      0.00      0.00         8
              DESC:def       0.80      1.00      0.89       123
             DESC:desc       0.75      0.43      0.55         7
           DESC:manner       0.67      1.00      0.80         2
           DESC:reason       1.00      0.17      0.29         6
           ENTY:animal       0.00      0.00      0.00        16
             ENTY:body       0.00      0.00      0.00         2
            ENTY:color       0.00      0.00      0.00        10
         ENTY:currency       0.00      0.00      0.00         6
           ENTY:dismed       0.00      0.00      0.00         2
            ENTY:event       0.00      0.00      0.00         2
             ENTY:food       0.00      0.00      0.00         4
           ENTY:instru       0.00      0.00      0.00         1
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
               HUM:ind       0.25      1.00      0.41        55
             HUM:title       0.00      0.00      0.00         1
              LOC:city       1.00      0.06      0.11        18
           LOC:country       1.00      0.33      0.50         3
             LOC:mount       0.00      0.00      0.00         3
             LOC:other       0.61      0.74      0.67        50
             LOC:state       0.00      0.00      0.00         7
             NUM:count       0.33      1.00      0.50         9
              NUM:date       1.00      0.68      0.81        47
              NUM:dist       0.00      0.00      0.00        16
             NUM:money       0.00      0.00      0.00         3
             NUM:other       0.00      0.00      0.00        12
              NUM:perc       0.00      0.00      0.00         3
            NUM:period       1.00      0.12      0.22         8
             NUM:speed       0.00      0.00      0.00         6
              NUM:temp       0.00      0.00      0.00         5
            NUM:weight       0.00      0.00      0.00         4
        
             micro avg       0.53      0.53      0.53       500
             macro avg       0.20      0.16      0.14       500
          weighted avg       0.47      0.53      0.44       500
        """
    _matrix = """
        
        [[  0   0   0 ...   0   0   0]
         [  0   0   8 ...   0   0   0]
         [  0   0 123 ...   0   0   0]
         ...
         [  0   0   1 ...   0   0   0]
         [  0   0   1 ...   0   0   0]
         [  0   0   0 ...   0   0   0]]

        """

    @property
    def pipeline(self):
        # no w2v, can't have negative values
        return [
            ('features', default_pipeline2),
            ('clf', self.classifier_class)
        ]


class NaiveMainQuestionClassifier(MainQuestionClassifier, NaiveTextClassifier):
    # default pipeline2 - Accuracy: 0.81
    _accuracy = 0.81
    _report = """
                      precision    recall  f1-score   support
        
                ABBR       0.00      0.00      0.00         9
                DESC       0.81      0.98      0.89       138
                ENTY       0.63      0.76      0.69        94
                 HUM       0.84      0.89      0.87        65
                 LOC       0.86      0.79      0.83        81
                 NUM       0.99      0.68      0.81       113
        
           micro avg       0.81      0.81      0.81       500
           macro avg       0.69      0.68      0.68       500
        weighted avg       0.82      0.81      0.80       500
        """
    _matrix = """
        [[  0   8   1   0   0   0]
         [  0 135   2   0   0   1]
         [  0  12  71   7   4   0]
         [  0   0   5  58   2   0]
         [  0   2  13   2  64   0]
         [  0   9  21   2   4  77]]


        """

    @property
    def pipeline(self):
        # no w2v, can't have negative values
        return [
            ('features', default_pipeline2),
            ('clf', self.classifier_class)
        ]


class NaiveSentenceClassifier(SentenceClassifier, NaiveTextClassifier):
    # default pipeline - Accuracy: ?
    _accuracy = 0
    _report = """
                         precision    recall  f1-score   support

          
        """
    _matrix = """


        """

    @property
    def pipeline(self):
        # no w2v, can't have negative values
        return [
            ('features', default_pipeline2),
            ('clf', self.classifier_class)
        ]


if __name__ == '__main__':
    train = True
    search = False
    name = "questions_naive"
    clf = NaiveQuestionClassifier(name)
    name = "main_questions_naive"
    main_clf = NaiveMainQuestionClassifier(name)
    name = "sentences_naive"
    sent_clf = NaiveSentenceClassifier(name)
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
