from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.adaboost import AdaBoostTextClassifier
from os.path import join


class AdaBoostQuestionClassifier(QuestionClassifier, AdaBoostTextClassifier):
    pass


class AdaBoostMainQuestionClassifier(MainQuestionClassifier,
                                     AdaBoostTextClassifier):
    pass


class AdaBoostSentenceClassifier(SentenceClassifier, AdaBoostTextClassifier):
    pass


if __name__ == '__main__':
    train = True
    search = True
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
    clf.evaluate_model(test_data_path)
