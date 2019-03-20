from os.path import dirname, join
import spacy

nlp = spacy.load('en')

# https://drive.google.com/uc?id=1saFGKezSFgH-5YjsQX_yiWes41xqbHrf&export=download
GLOVE_PATH = join(dirname(__file__), "data", "glove.txt")

# begin of sentence indicators for Yes/No questions pre parsing
AFFIRMATIONS = ["do", "would", "it's", "is", "will", "does", "can", "has",
                "could", "are", "should", "have", "has", "did"]

MODELS_PATH = join(dirname(__file__), "models")
DATA_PATH = join(dirname(__file__), "data")
RESOURCES_PATH = join(dirname(__file__), "res")

DEFAULT_CLASSIFIER = "passive_agressive_model.pkl"
DEFAULT_SIMPLE_CLASSIFIER = "passive_agressive_main_model.pkl"