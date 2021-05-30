from os.path import dirname, join

# begin of sentence indicators for Yes/No questions pre parsing
AFFIRMATIONS = ["would", "is", "will", "does", "can", "has",
                "could", "are", "should", "have", "has", "did"]

MODELS_PATH = join(dirname(__file__), "models")

DEFAULT_CLASSIFIER = "questions50.pkl"
DEFAULT_MAIN_CLASSIFIER = "questions6.pkl"

ALL_POS_TAGS = ['NNPS', '--', '.', 'POS', 'RB', 'UH', 'SYM', '(', 'JJR', 'WDT',
                'PRP', 'NNS', 'JJS', '$', 'JJ', 'IN', 'EX', 'CC', 'NN', 'MD',
                '``', ',', 'RBR', ':', 'PDT', 'WP', 'RP', 'WP$', 'TO', 'VBP',
                'WRB', 'VB', 'VBG', 'VBN', ')', 'DT', "''", 'PRP$', 'VBZ',
                'VBD', 'FW', 'LS', 'CD', 'NNP', 'RBS']
