from little_questions.parsers import BasicQuestionParser
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from os.path import join, exists
from little_questions.settings import MODELS_PATH, DATA_PATH, nlp, AFFIRMATIONS
import json
from pprint import pprint

_parser = BasicQuestionParser()

# categories = parse_labels()
categories = ['ABBR:abb', 'ABBR:exp', 'DESC:def', 'DESC:desc', 'DESC:manner',
              'DESC:reason', 'ENTY:animal', 'ENTY:body', 'ENTY:color',
              'ENTY:cremat', 'ENTY:currency', 'ENTY:dismed', 'ENTY:event',
              'ENTY:food', 'ENTY:instru', 'ENTY:lang', 'ENTY:letter',
              'ENTY:other', 'ENTY:plant', 'ENTY:product', 'ENTY:religion',
              'ENTY:sport', 'ENTY:substance', 'ENTY:symbol', 'ENTY:techmeth',
              'ENTY:termeq', 'ENTY:veh', 'ENTY:word', 'HUM:desc', 'HUM:gr',
              'HUM:ind', 'HUM:title', 'LOC:city', 'LOC:country', 'LOC:mount',
              'LOC:other', 'LOC:state', 'NUM:code', 'NUM:count', 'NUM:date',
              'NUM:dist', 'NUM:money', 'NUM:ord', 'NUM:other', 'NUM:perc',
              'NUM:period', 'NUM:speed', 'NUM:temp', 'NUM:volsize',
              'NUM:weight']
label_encoder = LabelEncoder()
label_encoder.fit(categories)


def _parse(text, debug=False):
    global _parser
    dict_feats = _parser.parse(text)
    text = nlp(text)
    s_feature = {
        'tag': "",
        'is_wh': False,
        'is_affirmation': False
    }
    for token in text:
        if token.text.lower().startswith('wh'):
            s_feature['is_wh'] = True
        if token.text.lower() in AFFIRMATIONS:
            s_feature['is_affirmation'] = True
        s_feature['tag'] = token.tag_
        break
    s_feature.update(dict_feats)

    if debug or s_feature['QuestionIntent'] == "unknown":
        print("Q:", text)
        print("Intent:", s_feature['QuestionIntent'])
        pprint(s_feature)
        print("___")

    return s_feature


def extract_json_features(questions, dump=True):
    print("calculating question intents")
    if ":" in questions[0].split(" ")[0]:
        # CAT:label question
        dicts = []
        for idx, q in enumerate(questions):
            words = q.split(" ")
            q = " ".join(words[1:])
            d = _parse(q)
            d["label"] = words[0]
            dicts += [d]
    else:
        dicts = [_parse(q) for q in questions]
    if dump:
        with open(join(DATA_PATH, 'dict_features.json'), "w") as write_file:
            json.dump(dicts, write_file, indent=4)
    return dicts


def train_vectorizers(questions_file=join(DATA_PATH, "questions.txt"),
                      dump=True):
    print("Training vectorizing models")

    print("reading questions file", questions_file)
    with open(questions_file) as f:
        questions = f.readlines()
    print("loaded", len(questions), "questions")
    dicts = extract_json_features(questions, dump=dump)

    if ":" in questions[0].split(" ")[0]:
        # CAT:label question
        for idx, q in enumerate(questions):
            words = q.split(" ")
            questions[idx] = " ".join(words[1:])

    print("fitting DictVectorizer")
    special_vect = DictVectorizer(sparse=False)
    special_vect.fit_transform(dicts)
    if dump:
        print('Dumping Dict Vectorizer to models directory')
        joblib.dump(special_vect, join(MODELS_PATH, 'dict_vect.pkl'))

    # Get TfIdf Vector
    print("fitting TfidfVectorizer")
    tfidf_vect = TfidfVectorizer(ngram_range=[1, 2], encoding='utf-8')
    tfidf_vect.fit_transform(questions).toarray()
    if dump:
        print('Dumping TfIdf vectorizer to models directory')
        joblib.dump(tfidf_vect, join(MODELS_PATH, 'tfidf_vect.pkl'))
    return special_vect, tfidf_vect


if exists(join(MODELS_PATH, 'dict_vect.pkl')) and exists(
        join(MODELS_PATH, 'tfidf_vect.pkl')):
    print('Loading vectorizing models')
    dict_vect = joblib.load(join(MODELS_PATH, 'dict_vect.pkl'))
    tfidf_vect = joblib.load(join(MODELS_PATH, 'tfidf_vect.pkl'))
else:

    dict_vect, tfidf_vect = train_vectorizers()


def featurize(sentence):
    global dict_vect, tfidf_vect
    parse_data = _parse(sentence)
    spec_features = dict_vect.transform(parse_data)
    tfidf_features = tfidf_vect.transform([sentence]).toarray()

    return np.concatenate((spec_features, tfidf_features), axis=1)


def parse_labels(questions_path=join(DATA_PATH, "questions.txt")):
    with open(questions_path) as f:
        sentences = [s for s in f.readlines()]
    labels = []
    for s in sentences:
        lbl = s.split(" ")[0]
        if lbl not in labels:
            labels += [lbl]
    return sorted(labels)


if __name__ == "__main__":
    # re-train
    train_vectorizers()
