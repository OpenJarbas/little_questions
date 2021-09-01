from little_questions.classifiers.lang.pt.tokenize import word_tokenize_pt
import JarbasModelZoo

def load_pt_tagger():
    return JarbasModelZoo.load_model("nltk_floresta_macmorpho_brill_tagger")


def pos_tag_pt(tokens, tagger=None):
    tagger = tagger or load_pt_tagger()
    if isinstance(tokens, str):
        tokens = word_tokenize_pt(tokens)
    postagged = tagger.tag(tokens)

    # HACK this fixes some know failures from postag
    # this is not sustainable but important cases can be added at any time
    # PRs + unittests welcome!
    DETS = ["a", "á", "o", "ós", "aos", "ao"]
    for idx, (w, t) in enumerate(postagged):
        #  ('á', 'NOUN'), ('Maria', 'NOUN')
        if w.lower() in DETS and t == "NOUN":
            postagged[idx] = (w, "DET")

    return postagged
