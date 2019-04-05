from text_classifikation.classifiers.pipelines import \
    default_pipelines as pipelines, generate_unions

from sklearn.pipeline import Pipeline, FeatureUnion
from little_questions.classifiers.features import DictTransformer, \
    NeuralDictTransformer
from sklearn.feature_extraction import DictVectorizer


pipeline__intent = Pipeline([('dict', DictTransformer()),
                             ('dict_vec', DictVectorizer())])

pipeline__neuralintent = Pipeline([('dict', NeuralDictTransformer()),
                                   ('dict_vec', DictVectorizer())])

from text_classifikation.classifiers.pipelines import pipeline__postag, \
    pipeline__w2v, pipeline__cv2, pipeline__lemma_tfidf

default_pipeline = FeatureUnion([
        ("cv", pipeline__cv2),
        ("w2v", pipeline__w2v),
        ("tfidf", pipeline__lemma_tfidf),
        ("postag", pipeline__postag)
    ])
default_pipeline2 = FeatureUnion([
        ("cv", pipeline__cv2),
        ("tfidf", pipeline__lemma_tfidf),
        ("postag", pipeline__postag)
    ])
# pipelines["intentdict"] = pipeline__intent
# pipelines["neuralintentdict"] = pipeline__neuralintent

_independent_components = {
    "cv": ("cv", "cv2", "cv3", "cv_lemma", "cv2_lemma", "cv3_lemma"),
    "tfidf": ("tfidf", "tfidf2", "tfidf3", "tfidf_lemma", "tfidf2_lemma",
              "tfidf3_lemma"),
    "w2v": ("w2v", "w2v_lemma"),
    "postag": ("postag", ),
    "ner": ("ner", ),
    #"intentdict": ("intentdict", "neuralintentdict")
}

pipeline_unions = generate_unions(pipelines, _independent_components)
