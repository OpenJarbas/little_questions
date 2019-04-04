from little_questions.settings import AFFIRMATIONS
from little_questions.parsers import BasicQuestionParser
from little_questions.parsers.neural import NeuralQuestionParser
from sklearn.base import BaseEstimator, TransformerMixin


class DictTransformer(BaseEstimator, TransformerMixin):
    """ transofmr a list of sentences into a list of dicts """
    parser = BasicQuestionParser()

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []
        for sent in X:
            sent = sent.strip().lower().replace("``", "").replace("''", "")\
                .replace(" '", "'")
            dict_feats = self.parser.parse(sent)
            s_feature = {
                'is_affirmation': sent.split(" ")[0] in AFFIRMATIONS,
                'is_wh': sent.startswith('wh')
            }
            s_feature.update(dict_feats)
            feats.append(s_feature)

        return feats


class NeuralDictTransformer(DictTransformer):
    """ transofmr a list of sentences into a list of dicts """
    parser = NeuralQuestionParser()

