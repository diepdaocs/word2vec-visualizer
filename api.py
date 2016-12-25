from flask import Flask, request
from flask_restplus import Api, Resource
from util.utils import get_logger, is_number
from word2vec import Word2VecModel

logger = get_logger(__name__)

word2VecModel = Word2VecModel()

app = Flask(__name__)
api = Api(app, doc='/doc/', version='1.0', title='Content Insights with Deep Learning')

ns_word2vec = api.namespace('word2vec', 'Word2Vec')


@ns_word2vec.route('/most_similar')
class MostSimilarResource(Resource):
    @api.doc(params={'word': 'Word',
                     'topn': 'Number of return words, default is 10'})
    @api.response(200, 'Success')
    def get(self):
        """Get most similar words"""
        result = {
            'error': False,
            'message': '',
            'similar': []
        }
        topn = request.values.get('topn', '')
        if not topn or not topn.strip():
            topn = '10'

        if not is_number(topn):
            result = {
                'error': True,
                'message': "'topn' must be a number",
                'similar': []
            }
            return result

        word = request.values.get('word', '').strip()
        if not word:
            result['error'] = True
            result['message'] = 'Empty word'
            return result

        model = word2VecModel.get_model()
        try:
            result['similar'] = model.similar_by_word(word=word, topn=int(topn))
        except Exception as ex:
            logger.error(ex)
            result['error'] = True
            result['message'] = ex.message
            return result

        return result


@ns_word2vec.route('/analogy')
class SimilarResource(Resource):
    @api.doc(params={'positive_words': 'Positive words (separate by comma)',
                     'negative_words': 'Negative words (separate by comma)'})
    @api.response(200, 'Success')
    def get(self):
        """Get analogy words"""
        result = {
            'error': False,
            'message': '',
            'similar': []
        }
        pos_words = request.values.get('positive_words', '')
        pos_words = [u.strip() for u in pos_words.split(',') if u]
        neg_words = request.values.get('negative_words', '')
        neg_words = [u.strip() for u in neg_words.split(',') if u]
        if not pos_words and not neg_words:
            result['error'] = True
            result['message'] = 'Empty positive and negative words'
            return result

        model = word2VecModel.get_model()
        try:
            result['similar'] = model.most_similar(positive=pos_words, negative=neg_words)
        except Exception as ex:
            logger.error(ex)
            result['error'] = True
            result['message'] = ex.message
            return result
        return result


@ns_word2vec.route('/n_similarity')
class NSimilarResource(Resource):
    @api.doc(params={'words_set_1': 'Words set 1 (separate by comma)',
                     'words_set_2': 'Words set 2 (separate by comma)'})
    @api.response(200, 'Success')
    def get(self):
        """Compute cosine similarity between two sets of words"""
        result = {
            'error': False,
            'message': '',
            'similarity': 0
        }
        words_set_1 = request.values.get('words_set_1', '')
        words_set_1 = [u.strip() for u in words_set_1.split(',') if u]
        words_set_2 = request.values.get('words_set_2', '')
        words_set_2 = [u.strip() for u in words_set_2.split(',') if u]
        if not words_set_1 or not words_set_2:
            result['error'] = True
            result['message'] = 'Some words set is empty'
            return result

        model = word2VecModel.get_model()
        try:
            result['similarity'] = model.n_similarity(words_set_1,     words_set_2)
        except Exception as ex:
            logger.error(ex)
            result['error'] = True
            result['message'] = ex.message
            return result
        return result


@ns_word2vec.route('/similarity')
class SimilarityResource(Resource):
    @api.doc(params={'first_word': 'First word',
                     'second_word': 'Second word'})
    @api.response(200, 'Success')
    def get(self):
        """Check similarity between words"""
        result = {
            'error': False,
            'message': '',
            'similarity': []
        }
        first_word = request.values.get('first_word', '').strip()
        second_word = request.values.get('second_word', '').strip()
        if not first_word or not second_word:
            result['error'] = True
            result['message'] = 'Empty first word and second word'
            return result

        model = word2VecModel.get_model()
        try:
            result['similar'] = model.similarity(first_word, second_word)
        except Exception as ex:
            logger.error(ex)
            result['error'] = True
            result['message'] = "word '%s' not in vocabulary" % ex.message
            return result

        return result


@ns_word2vec.route('/doesnt_match')
class CheckerResource(Resource):
    @api.doc(params={'sentence': 'Sentence that want to check logic'})
    @api.response(200, 'Success')
    def get(self):
        """Get words that don't make sense in a sentence"""
        result = {
            'error': False,
            'message': '',
            'similar': []
        }
        sentence = request.values.get('sentence', '').strip()
        if not sentence:
            result['error'] = True
            result['message'] = 'Empty sentence'
            return result

        model = word2VecModel.get_model()
        try:
            result['doesnt_match'] = model.doesnt_match(w.strip() for w in sentence.split())
        except Exception as ex:
            logger.error(ex)
            result['error'] = True
            result['message'] = ex.message
            return result

        return result
