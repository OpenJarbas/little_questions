from little_questions.settings import DATA_PATH
from little_questions.classifiers import QuestionClassifier, \
    MainQuestionClassifier, SentenceClassifier, best_pipeline
from text_classifikation.classifiers.tree import ExtraTreeTextClassifier, \
    TreeTextClassifier
from os.path import join


class TreeQuestionClassifier(QuestionClassifier, TreeTextClassifier):
    pass


class TreeMainQuestionClassifier(MainQuestionClassifier, TreeTextClassifier):
    pass


class TreeSentenceClassifier(SentenceClassifier, TreeTextClassifier):
    pass


class ExtraTreeQuestionClassifier(QuestionClassifier, ExtraTreeTextClassifier):
    pass


class ExtraTreeMainQuestionClassifier(MainQuestionClassifier,
                                      ExtraTreeTextClassifier):
    pass


class ExtraTreeSentenceClassifier(SentenceClassifier, ExtraTreeTextClassifier):
    pass


if __name__ == '__main__':
    train = True
    search = True
    name = "questions_xtratree"
    clf = ExtraTreeQuestionClassifier(name)
    name = "main_questions_xtratree"
    main_clf = ExtraTreeMainQuestionClassifier(name)
    name = "sentences_xtratree"
    sent_clf = ExtraTreeSentenceClassifier(name)
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
