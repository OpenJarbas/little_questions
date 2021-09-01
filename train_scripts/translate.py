from os.path import join, dirname, isfile

from google_tx import google_translator

DATA_PATH = join(dirname(__file__), "clean_data")


def translate(dest, translator, lang_tgt='pt', lang_src="en",
              src_data=join(DATA_PATH, "raw_questions_0.7.0a1.txt")):
    questions = []
    tx = []
    with open(src_data) as f:
        questions += f.read().split("\n")

    if isfile(join(DATA_PATH, dest)):
        with open(join(DATA_PATH, dest)) as f:
            tx = f.read().split("\n")

    for q in questions[len(tx):]:
        label = q.split(" ")[0]
        translate_text = " ".join(q.split(" ")[1:])
        try:
            translate_text = translator.translate(translate_text,
                                                  lang_tgt=lang_tgt,
                                                  lang_src=lang_src)
            tx.append(label + " " + translate_text)
            print(">>", q)
            print(label + " " + translate_text)
        except Exception as e:
            print(e)
            break

    with open(join(DATA_PATH, dest), "w") as f:
        f.write("\n".join(tx))


translate("raw_questions_DE_googtx0.7.0a1.txt",
          lang_tgt='de', lang_src="en", translator=google_translator(),
          src_data=join(DATA_PATH, "raw_questions_0.7.0a1.txt"))
translate("raw_questions_PT_googtx0.7.0a1.txt",
          lang_tgt='pt', lang_src="en", translator=google_translator(),
          src_data=join(DATA_PATH, "raw_questions_0.7.0a1.txt"))
translate("raw_questions_ES_googtx0.7.0a1.txt",
          lang_tgt='es', lang_src="en", translator=google_translator(),
          src_data=join(DATA_PATH, "raw_questions_0.7.0a1.txt"))
translate("raw_questions_IT_googtx0.7.0a1.txt",
          lang_tgt='it', lang_src="en", translator=google_translator(),
          src_data=join(DATA_PATH, "raw_questions_0.7.0a1.txt"))
translate("raw_questions_CA_googtx0.7.0a1.txt",
          lang_tgt='ca', lang_src="en", translator=google_translator(),
          src_data=join(DATA_PATH, "raw_questions_0.7.0a1.txt"))
