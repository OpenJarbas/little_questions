from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from little_questions.classifiers.features import WordFeaturesVectorizer
from little_questions.classifiers.lang.en.features import *
from little_questions.classifiers.lang.pt.features import *
from little_questions.classifiers.lang.es.features import *
from little_questions.classifiers.lang.ca.features import *
from little_questions.classifiers.lang.fr.features import *
from little_questions.classifiers.lang.de.features import *
from little_questions.classifiers.lang.it.features import *
from little_questions.classifiers.lang.en import SentenceScorerEN

# TODO identify the best pipeline for each language


# lazy loaded as needed, avoid loading all langs at once"
LANG2PIPELINE = {
    "en": None,
    "pt": None,
    "es": None,
    "ca": None,
    "fr": None,
    "de": None,
    "it": None,
    # works for all languages...
    "default": None,
    "naive": None,
    "naive_en": None,
}


def get_pipeline_naive():
    global LANG2PIPELINE
    if not LANG2PIPELINE["naive"]:
        LANG2PIPELINE["naive"] = FeatureUnion([
            ("word_feats", WordFeaturesVectorizer())
        ])
    return LANG2PIPELINE["naive"]


def get_pipeline_naive_en():
    global LANG2PIPELINE
    if not LANG2PIPELINE["naive"]:
        LANG2PIPELINE["naive"] = FeatureUnion([
            ("word_feats", WordFeaturesVectorizer()),
            ("postag", POSTaggerVectorizerEN())
        ])
    return LANG2PIPELINE["naive"]


def get_pipeline_default():
    global LANG2PIPELINE
    if not LANG2PIPELINE["default"]:
        LANG2PIPELINE["default"] = FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            ("word_feats", WordFeaturesVectorizer()),
            ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
        ])
    return LANG2PIPELINE["default"]


def get_pipeline_en():
    global LANG2PIPELINE
    if not LANG2PIPELINE["en"]:
        LANG2PIPELINE["en"] = FeatureUnion([
            ("question_feats", QuestionFeaturesVectorizerEN()),
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerEN()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ])),
            ("postag", POSTaggerVectorizerEN())
        ])
    return LANG2PIPELINE["en"]


def get_pipeline_es():
    global LANG2PIPELINE
    if not LANG2PIPELINE["es"]:
        LANG2PIPELINE["es"] = FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            #  ("word_feats", WordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerES()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ])),
            ("postag", POSTaggerVectorizerES())
        ])
    return LANG2PIPELINE["es"]


def get_pipeline_pt():
    global LANG2PIPELINE
    if not LANG2PIPELINE["pt"]:
        LANG2PIPELINE["pt"] = FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            #     ("word_feats", WordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerPT()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ])),
            ("postag", POSTaggerVectorizerPT())
        ])
    return LANG2PIPELINE["pt"]


def get_pipeline_fr():
    global LANG2PIPELINE
    if not LANG2PIPELINE["fr"]:
        LANG2PIPELINE["fr"] = FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            ("word_feats", WordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerFR()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ]))
        ])
    return LANG2PIPELINE["fr"]


def get_pipeline_ca():
    global LANG2PIPELINE
    if not LANG2PIPELINE["ca"]:
        LANG2PIPELINE["ca"] = FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            #    ("word_feats", WordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerCA()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ])),
            ("postag", POSTaggerVectorizerCA())
        ])
    return LANG2PIPELINE["ca"]


def get_pipeline_de():
    global LANG2PIPELINE
    if not LANG2PIPELINE["de"]:
        LANG2PIPELINE["de"] = FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            ("word_feats", WordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerDE()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ]))
        ])
    return LANG2PIPELINE["de"]


def get_pipeline_it():
    global LANG2PIPELINE
    if not LANG2PIPELINE["it"]:
        LANG2PIPELINE["it"] = FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            ("word_feats", WordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerIT()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ]))
        ])
    return LANG2PIPELINE["it"]


def get_pipeline(lang="default"):
    lang = lang.replace("_small", "")
    if lang.startswith("en"):
        return get_pipeline_en()
    if lang.startswith("pt"):
        return get_pipeline_pt()
    if lang.startswith("es"):
        return get_pipeline_es()
    if lang.startswith("fr"):
        return get_pipeline_fr()
    if lang.startswith("ca"):
        return get_pipeline_ca()
    if lang.startswith("it"):
        return get_pipeline_it()
    if lang.startswith("de"):
        return get_pipeline_de()
    return get_pipeline_default()
