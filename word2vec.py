import logging
import os

from gensim.models.word2vec import Word2Vec, Text8Corpus

from util.utils import get_logger

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = get_logger(__name__)


class Word2VecModel:
    def __init__(self):
        self.default_model_path = 'text8.model'
        self.model = None

    def get_model(self):
        if self.model:
            return self.model
        logger.info('Start loading word2vec model...')
        word2Vec = Word2Vec()
        self.model = word2Vec.load_word2vec_format(os.getenv('WORD2VEC_MODEL_PATH', self.default_model_path),
                                                   binary=True)
        # self.model = word2Vec.load(os.getenv('WORD2VEC_MODEL_PATH', self.default_model_path))
        self.model.init_sims(replace=True)
        logger.info('Finish loading word2vec model...')
        return self.model


if __name__ == '__main__':
    # test word2vec
    text8Corpus = Text8Corpus(fname='/home/diepdt/Documents/word2vec/text8')
    model = Word2Vec(text8Corpus, workers=4)
    model.save('text8.model')
    model.most_similar(['apple'])
