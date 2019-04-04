from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.ridge import RidgeTextClassifier
from os.path import join


class RidgeQuestionClassifier(QuestionClassifier,  RidgeTextClassifier):
    pass


class RidgeMainQuestionClassifier(MainQuestionClassifier, RidgeTextClassifier):
    pass


class RidgeSentenceClassifier(SentenceClassifier,  RidgeTextClassifier):
    pass


if __name__ == '__main__':
    train = True
    search = True
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
    clf.evaluate_model(test_data_path)
