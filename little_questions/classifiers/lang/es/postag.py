from nltk import word_tokenize
import JarbasModelZoo

def load_es_tagger():
    return JarbasModelZoo.load_model("nltk_cess_esp_udep_brill_tagger")


def pos_tag_es(tokens, tagger=None):
    tagger = tagger or load_es_tagger()
    if isinstance(tokens, str):
        tokens = word_tokenize(tokens)
    postagged = tagger.tag(tokens)

    return postagged
