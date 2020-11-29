from little_questions.classifiers.svm import \
    LinearSVCMainQuestionClassifier, LinearSVCQuestionClassifier, \
    LinearSVCSentenceClassifier, find_best_pipeline

# see little_questions/classifiers/pipelines.py for builtin pipeline options

# kwargs for find_best_pipeline
# outfolder=None,
# save_all=False,
# skip_existing=True,
# save_best=True,
# weights={"f1": 0.5, "acc": 0.15, "recall": 0.2, "precision": 0.15}

outfolder = "models"

print("MAIN LABEL")
main_clf = LinearSVCMainQuestionClassifier("main_questions_svc")
best_score, best_pipeline = find_best_pipeline(main_clf,
                                               outfolder=outfolder,
                                               save_all=False)
print("BEST:", best_pipeline, "WEIGHTED_SCORE:", best_score)

print("MAIN_LABEL : SECONDARY_LABEL")
clf = LinearSVCQuestionClassifier("questions_svc")
best_score, best_pipeline = find_best_pipeline(clf,
                                               outfolder=outfolder,
                                               save_all=False)
print("BEST:", best_pipeline, "WEIGHTED_SCORE:", best_score)

print("QUESTION/SENTENCE")
sent_clf = LinearSVCSentenceClassifier("sentences_svc")
best_score, best_pipeline = find_best_pipeline(sent_clf,
                                               outfolder=outfolder,
                                               save_all=False)
print("BEST:", best_pipeline, "WEIGHTED_SCORE:", best_score)
