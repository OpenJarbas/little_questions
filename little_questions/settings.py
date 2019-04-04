from os.path import dirname, join

# begin of sentence indicators for Yes/No questions pre parsing
AFFIRMATIONS = ["do", "would", "it's", "is", "will", "does", "can", "has",
                "could", "are", "should", "have", "has", "did"]

MODELS_PATH = join(dirname(__file__), "models")
DATA_PATH = join(dirname(__file__), "data")
RESOURCES_PATH = join(dirname(__file__), "res")
INTENT_CACHE_PATH = join(MODELS_PATH, 'intent_cache')

DEFAULT_CLASSIFIER = "passive_agressive_model.pkl"
DEFAULT_MAIN_CLASSIFIER = "passive_agressive_main_model.pkl"
DEFAULT_SENTENCE_CLASSIFIER = "logreg_sentence_model.pkl"