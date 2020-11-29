from sklearn.metrics import classification_report, confusion_matrix
from little_questions.classifiers.svm import LinearSVCQuestionClassifier
from little_questions.settings import DATA_PATH
from os.path import join

train = True
model_path = "my_model.pkl"


# default pipeline will be used,
#
# default_pipeline = FeatureUnion([
#     ("cv", pipeline__cv2),
#     ("w2v", pipeline__w2v),
#     ("tfidf", pipeline__lemma_tfidf),
#     ("postag", pipeline__postag)
# ])
#
# change this by subclassing
# see little_questions/classifiers/pipelines.py for builtin options
#
#     @property
#     def pipeline(self):
#         return [
#             ('features', default_pipeline),
#             ('clf', self.classifier_class)
#         ]
#
clf = LinearSVCQuestionClassifier("questions_svc")


train_data_path = join(DATA_PATH, "questions.txt")
test_data_path = join(DATA_PATH, "questions_test.txt")
if train:
    t, t_label = clf.load_data(train_data_path)
    clf.train(t, t_label)
    clf.save(model_path)
else:
    clf.load(model_path)

X_test, y_test = clf.load_test_data(test_data_path)
preds = clf.predict(X_test)

report = classification_report(y_test, preds)
print(report)

matrix = confusion_matrix(y_test, preds)
print(matrix)
