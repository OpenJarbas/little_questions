from functools import lru_cache
import math
from typing import Iterable, List

from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.porter import PorterStemmer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
import gensim.downloader as api

stemmer = PorterStemmer()
from os.path import isfile, join
from little_questions.settings import DATA_PATH, MODELS_PATH


@lru_cache(maxsize=1024)
def stem(word: str) -> str:
    """stemming words is not cheap, so use a cache decorator"""
    return stemmer.stem(word)


def tokenizer(sentence: str) -> List[str]:
    """use gensim's `simple_preprocess` and `STOPWORDS` list"""
    return [stem(token) for token in simple_preprocess(sentence) if
            token not in STOPWORDS]


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """https://en.wikipedia.org/wiki/Cosine_similarity"""
    num = np.dot(v1, v2)
    d1 = np.dot(v1, v1)
    d2 = np.dot(v2, v2)

    if d1 > 0.0 and d2 > 0.0:
        return num / math.sqrt(d1 * d2)
    else:
        return 0.0


class WordTwoVec(object):
    """
    a wrapper for gensim.Word2Vec with added functionality to embed phrases
    """
    # TODO default to Glove?
    def __init__(self, model_file=join(MODELS_PATH, 'question2vec.bin')):
        if model_file and isfile(model_file):
            self.model = Word2Vec.load(model_file)
        else:
            # download the model and return as object ready for use
            self.model = api.load(model_file)

    @staticmethod
    def models():
        # model names from https://github.com/RaRe-Technologies/gensim-data
        info = api.info()  # show info about available models/datasets
        return info

    def embed(self, words: Iterable[str]) -> np.ndarray:
        """given a list of words, find their vector embeddings and return the vector mean"""
        # first find the vector embedding for each word
        vectors = [self.model[word] for word in words if word in self.model]

        if vectors:
            # if there are vector embeddings, take the vector average
            return np.average(vectors, axis=0)
        else:
            # otherwise just return a zero vector
            return np.zeros(self.model.vector_size)

    def cosine_similarity(self, question_stem: str, choice_text: str) -> float:
        """how good is the choice for this question?"""
        question_words = {word for word in tokenizer(question_stem)}
        choice_words = {word for word in tokenizer(choice_text) if
                        word not in question_words}
        return cosine_similarity(self.embed(question_words),
                                 self.embed(choice_words))


def train_question_vectors(questions_path=join(DATA_PATH, "questions.txt"),
                           model_path=join(MODELS_PATH, 'question2vec.bin')):
    # train model
    with open(questions_path) as f:
        sentences = [s.split(" ")[1:] for s in f.readlines()]
    model = Word2Vec(sentences, min_count=1)
    # save model
    model.save(model_path)
    return model


if __name__ == "__main__":
    model = train_question_vectors()
    # summarize the loaded model
    print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words)
    # access vector for one word
    print(model['what'])