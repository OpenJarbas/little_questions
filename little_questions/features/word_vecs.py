from gensim.models.word2vec import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Word2VecVectorizer(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            size=100,
            alpha=0.025,
            window=5,
            min_count=1,
            max_vocab_size=None,
            sample=0.001,
            seed=1,
            workers=3,
            min_alpha=0.0001,
            sg=0,
            hs=0,
            negative=5,
            cbow_mean=1,
            iter=5,
            null_word=0,
            trim_rule=None,
            sorted_vocab=1,
            batch_words=10000,
            compute_loss=False
    ):
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.cbow_mean = cbow_mean
        self.iter = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.compute_loss = compute_loss
        self.model = Word2Vec()

    def fit(self, sentences, y=None):
        self.model = Word2Vec(
            sentences,
            size=self.size,
            alpha=self.alpha,
            window=self.window,
            min_count=self.min_count,
            max_vocab_size=self.max_vocab_size,
            sample=self.sample,
            seed=self.seed,
            workers=self.workers,
            min_alpha=self.min_alpha,
            sg=self.sg,
            hs=self.hs,
            negative=self.negative,
            cbow_mean=self.cbow_mean,
            iter=self.iter,
            null_word=self.null_word,
            trim_rule=self.trim_rule,
            sorted_vocab=self.sorted_vocab,
            batch_words=self.batch_words
        )
        return self

    def transform_sentence(self, sentence):
        vectors_list = [self.model[word] for word in sentence if
                        word in self.model.wv.vocab]
        if vectors_list:
            return np.mean(vectors_list, axis=0)
        else:
            return np.repeat(0, self.model.vector_size)

    def transform(self, sentences):
        return np.array(
            [list(self.transform_sentence(sentence)) for sentence in
             sentences])

    def fit_transform(self, sentences, y=None):
        self.fit(sentences)
        return self.transform(sentences)

