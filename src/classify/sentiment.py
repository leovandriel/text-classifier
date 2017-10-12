import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.insert(0, './src/grds')

from encoder import Model
import numpy as np

min_line_length = 50


class Sentiment:
    def __init__(self):
        print('loading LSTM model')
        self.model = Sentiment.load_model()
        self.coefs = Sentiment.load_coefs()

    def load_model():
        os.chdir('src/grds')
        model = Model()
        os.chdir('../..')
        return model

    def load_coefs():
        coefs = np.load('model/coefs.npy')
        return coefs

    def run_transformer(self, lines):
        transformed = self.model.transform(lines)
        return transformed

    def normalize(t):
        return 100 / (1 + np.exp(-(t)))

    def infer_sentiment(self, lines):
        transformed = self.run_transformer(lines)
        logistic = Sentiment.normalize(np.dot(transformed, self.coefs))
        single = Sentiment.normalize(transformed[:, 2388])
        return logistic, single

    def annotate(self, lines, logistic, single):
        annotated = [
            lines[i] + (' [%02.0f/%02.0f]' % (logistic[i], single[i])
                        if len(lines[i]) > min_line_length else '')
            for i in range(0, len(lines))
        ]
        return annotated

    def weighted_sentiment(lines, sent):
        weights = [(len(s) if len(s) > min_line_length else 0) for s in lines]
        weights /= np.sum(weights) + 1e-6
        weighted = np.sum(sent * weights)
        return weighted

    def mean_sentiment(lines, sent):
        weights = [(1 if len(s) > min_line_length else 0) for s in lines]
        weights /= np.sum(weights) + 1e-6
        weighted = np.sum(sent * weights)
        return weighted
