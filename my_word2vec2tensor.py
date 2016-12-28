import gensim
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

"""
To visualize your embeddings, there are 3 things you need to do:

1) Setup a 2D tensor variable(s) that holds your embedding(s).

embedding_var = tf.Variable(....)
2) Periodically save your embeddings in a LOG_DIR.

saver = tf.train.Saver()
saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)
3) Associate metadata with your embedding.

from tensorflow.contrib.tensorboard.plugins import projector
# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.train.SummaryWriter(LOG_DIR)

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

# Saves a configuration file that TensorBoard will read during startup.
projector.visualize_embeddings(summary_writer, config)
After running your model and training your embeddings, run TensorBoard and point it to the LOG_DIR of the job.

tensorboard --logdir=LOG_DIR
"""

# Config

# word2vec_model_path = 'text8.model'
# model = gensim.models.Word2Vec.load(word2vec_model_path)
word2vec_model_path = '/root/word2vec/models/GoogleNews-vectors-negative300.bin'
model = gensim.models.Word2Vec.load_word2vec_format(word2vec_model_path, binary=True)

tensor_filename = './projector/prefix'
tensor_tsv = tensor_filename + '_tensor.tsv'
tensor_metadata_tsv = tensor_filename + '_metadata.tsv'

voc_num = 10000
dim = 300

# Loading Word2Vec model to tensor and tensor metadata
w2v = np.zeros((voc_num, dim))
with open(tensor_tsv, 'w+') as file_vector:
    with open(tensor_metadata_tsv, 'w+') as file_metadata:
        for idx, word in enumerate(model.index2word[:voc_num]):
            w2v[idx] = model[word]
            file_metadata.write(word.encode('utf-8') + '\n')
            vector_row = '\t'.join(map(str, model[word]))
            file_vector.write(vector_row + '\n')


# Step 1 (Define the model without training)
sess = tf.InteractiveSession()
embedding = tf.Variable(w2v, trainable=False, name=tensor_filename)
tf.global_variables_initializer().run()

# Step 2 (Save embedding)
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter('./projector', sess.graph)

# Step 3 (Link metadata with tensor)
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = tensor_filename
embed.metadata_path = tensor_metadata_tsv
# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(summary_writer, config)
saver.save(sess, './projector/prefix_model.ckpt', global_step=voc_num)
