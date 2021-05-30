from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from little_questions.features import LemmatizerTransformer, \
    POSTaggerVectorizer, NERVectorizer


pipeline__ner = Pipeline([
    ('ner', NERVectorizer())
])

pipeline__postag = Pipeline([
    ('pos_tag', POSTaggerVectorizer())
])

pipeline__cv2 = Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 2)))
])

pipeline__lemma_tfidf = Pipeline([
    ("lemma", LemmatizerTransformer()),
    ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
])

default_pipeline = FeatureUnion([
    ("cv2", pipeline__cv2),
    ('ner', pipeline__ner),
    ('tfidf_lemma', pipeline__lemma_tfidf),
    ("postag", pipeline__postag)
])
