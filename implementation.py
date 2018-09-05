import tensorflow as tf
import re
from string import punctuation


# BATCH_SIZE = 128
# MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider, also called as number of steps
# EMBEDDING_SIZE = 50  # Dimensions for each word vector

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider, also called as number of steps
EMBEDDING_SIZE = 50  # Dimensions for each word vector, is also the number of lstm cells

# stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
#                   'there', 'about', 'once', 'during', 'out', 'very', 'having',
#                   'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
#                   'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
#                   'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
#                   'each', 'the', 'themselves', 'below', 'are', 'we',
#                   'these', 'your', 'his', 'through', 'don', 'me', 'were',
#                   'her', 'more', 'himself', 'this', 'down', 'should', 'our',
#                   'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
#                   'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
#                   'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
#                   'yourselves', 'then', 'that', 'because', 'what', 'over',
#                   'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
#                   'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
#                   'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
#                   'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
#                   'how', 'further', 'was', 'here', 'than'})

stop_words = set({'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
    'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 
    'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
    'won', "won't", 'wouldn', "wouldn't"})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """

    # remove space, new line, tab, punctuation
    review = review.lower().replace("<br />", " ")
    review = re.sub(r"[0-9{}\s\n\t]+".format(punctuation), " ", review)
    words = review.lower().split(" ")

    processed_review = ""

    for word in words:
      if word not in stop_words:
        processed_review += " " + word

    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    # number of lstm layers
    num_layers = 2
    # number of class: postive and negative
    num_class = 2
    # learning rate
    learning_rate = 0.01



    # input data, emdedded data already 
    input_data = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")
    # labels
    labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 2], name="labels")
    # drop out rate
    dropout_keep_prob = tf.placeholder_with_default(0.6, shape=[], name="dropout_keep_prob")
    #dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name="dropout_keep_prob")


    '''
    ########################### basic lstm ###########################
    #lstm cell
    #cell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE)
    cell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE, initializer=tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32), forget_bias=1.0)
    # dropout input and output
    #cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    # multi layers
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)])
    # output and state
    output, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    # weight
    weight = tf.get_variable(name="weight", shape=[EMBEDDING_SIZE, num_class], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32))
    #bias
    bias = tf.get_variable(name="bias",shape=[num_class], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
    '''

    
    ########################### bi-direction lstm ###########################
    # multi layers
    output = input_data
    for layer in range(num_layers): 
      #lstm forward cell and lstm backward cell
      fwcell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE, initializer=tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32), forget_bias=1.0)
      bwcell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE, initializer=tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32), forget_bias=1.0)
      # dropout input and output
      fwcell = tf.contrib.rnn.DropoutWrapper(fwcell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob)
      bwcell = tf.contrib.rnn.DropoutWrapper(bwcell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob)
      # outputs and state
      outputs, _ = tf.nn.bidirectional_dynamic_rnn(fwcell, bwcell, output,scope='BLSTM_'+ str(layer), dtype=tf.float32)
      output = tf.concat(outputs,2)
    # weight
    weight = tf.get_variable(name="weight", shape=[2*EMBEDDING_SIZE, num_class], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32))
    # bias
    bias = tf.get_variable(name="bias",shape=[num_class], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
    


    # reshape output and get the last value 
    output = tf.transpose(output, [1, 0, 2])
    last_value = tf.gather(output, int(output.get_shape()[0]) - 1)
    prediction = tf.matmul(last_value, weight) + bias
    # correct prediction
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    # accuracy: calculate the mean value (tf.cast: convert a data type to another data type)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    # use softmax cross entropy to compute loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name="loss")
    # use adam optimizer model
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
