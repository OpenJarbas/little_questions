from nltk import word_tokenize
import JarbasModelZoo
from little_questions.models import LANG2MODEL


def load_ca_tagger():
    return JarbasModelZoo.load_model("nltk_cess_cat_udep_brill_tagger")


def pos_tag_ca(tokens, tagger=None):
    tagger = tagger or load_ca_tagger()
    if isinstance(tokens, str):
        tokens = word_tokenize(tokens)
    postagged = tagger.tag(tokens)

    return postagged
