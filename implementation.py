import tensorflow as tf
import re
from string import punctuation
from random import *


# BATCH_SIZE = 128
# MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider, also called as number of steps
# EMBEDDING_SIZE = 50  # Dimensions for each word vector

BATCH_SIZE = 16
MAX_WORDS_IN_REVIEW = 200 # Maximum length of a review to consider, also called as number of steps
EMBEDDING_SIZE = 50  # Dimensions for each word vector, is also the number of lstm cells

# number of cells
hidden_size = 16
# number of lstm layers
num_layers = 2
# number of class: postive and negative
num_class = 2
# drop out
dropout = 0.4

stop_words = set({"a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "actually", 
    "added", "adj", "after", "afterwards", "again", "against", "ah", "all", "almost", "alone", "along", "already", "also", 
    "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anymore", "anyone", 
    "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "aren't", "arent", "arise", 
    "around", "as", "aside", "ask", "asks", "asking", "at", "auth", "available", "away", "awfully", "b", "back", "be", "became", 
    "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", 
    "behind", "being", "believe", "below", "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", 
    "by", "c", "ca", "came", "can", "cannot", "can't", "cant", "cause", "causes", "certain", "certainly", "co", "com", "come", 
    "comes", "contain", "containing", "contains", "contained", "could", "couldnt", "d", "date", "did", "didn't", "didnt", 
    "different", "do", "does", "doesn't", "doesnt", "doing", "done", "don't", "dont", "down", "downwards", "due", "during", 
    "e", "each", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", 
    "especially", "et", "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", 
    "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former", 
    "formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get", "gets", "getting", "give", 
    "given", "gives", "giving", "go", "goes", "gone", "going", "got", "gotten", "h", "had", "happens", "hardly", "has", "hasn't", 
    "hasnt", "have", "haven't", "havent", "having", "he", "hed", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", 
    "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how", "howbeit", "however", 
    "hundred", "i", "id", "ie", "if", "i'll", "ill", "im", "immediate", "immediately", "importance", "important", "in", "inc", 
    "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "isnt", "it", "itd", "it'll", 
    "itll", "its", "itself", "i've", "ive", "j", "just", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", 
    "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "letting", "lets", "likely", 
    "line", "little", "'ll", "ll", "look", "looking", "looks", "ltd", "m", "made", "mainly", "make", "makes", "many", "may", "maybe", 
    "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", 
    "mostly", "mr", "mrs", "much", "mug", "must", "my", "myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", 
    "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", 
    "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained", 
    "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", "ord", 
    "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", 
    "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", 
    "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", 
    "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "re", "readily", "really", "recent", "recently", "ref", 
    "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", 
    "right", "run", "s", "said", "same", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", 
    "seems", "seen", "self", "selves", "seven", "several", "shall", "she", "shed", "she'll", "shell", "shes", "should", "shouldn't", 
    "shouldnt", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", 
    "slightly", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", 
    "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", 
    "such", "sufficiently", "suggest", "sup", "sure", "t", "take", "taken", "taking", "tell", "tends", "th", "than", "thank", "thanks", 
    "thanx", "that", "that'll", "thatll", "thats", "that've", "thatve", "the", "their", "theirs", "them", "themselves", "then", "thence", 
    "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "therell", "thereof", "therere", "theres", "thereto", 
    "thereupon", "there've", "thereve", "these", "they", "theyd", "they'll", "theyll", "theyre", "they've", "theyve", "think", "this", "those", 
    "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til", "tip", "to", "together", "too", "took", 
    "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", 
    "unlike", "unlikely", "until", "unto", "up", "upon", "ups", "us", "use", "used", "uses", "using", "usually", "v", "value", "various", 
    "'ve", "ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasn't", "wasnt", "way", "we", "wed", "welcome", 
    "we'll", "went", "were", "weren't", "werent", "we've", "weve", "what", "whatever", "what'll", "whatll", "whats", "when", "whence", 
    "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", 
    "whither", "who", "whod", "whoever", "whole", "who'll", "wholl", "whom", "whomever", "whos", "whose", "why", "widely", "willing", "wish", 
    "with", "within", "without", "won't", "wont", "words", "world", "would", "wouldnt", "www", "x", "y", "yes", "yet", "you", "youd", "you'll", 
    "youll", "your", "youre", "yours", "yourself", "yourselves", "you've", "youve", "z", "zero", "br"})

linked_verbs = set({"am", "is", "isn't", "are", "aren't", "was", "wasn't", "were", "weren't", "be", "been", "being", "act", "acts", "acted", 
  "appear", "appears", "appeared", "became", "become", "becomes", "come", "came", "comes","did", "do", "does", "done", "fall", "falls", "feel", 
  "feels", "fell", "felt",  "get", "got", "gets", "getting", "go", "going", "goes", "grew", "grow", "grows", "grown", "had", "has", "have", 
  "indicate", "indicates", "indicated", "keep", "keeps", "kept", "look", "looks", "looked", "prove", "proves", "remain", "remains", "remained", 
  "seem", "seemed", "seems", "smell", "smells", "smelt", "sound", "sounds","stay", "stays", "stayed", "taste", "tastes", "tasted", "turn", 
  "turns", "turned", "wax", "waxed", "waxes", "went"})

link_words = set({'based on', 'in the light of', 'for one thing', 'beneath', 'further', 'in order to', 'given that', 'due to', 'all of a sudden', 'around', 
  'consequently', 'similarly', 'to repeat', 'by the time', 'notably', 'sooner or later', 'comparatively', 'although', 'in particular', 'over', 
  'indeed', 'on this side', 'particularly', 'up to the present time', 'because of', 'most compelling evidence', 'prior to', 'on the positive side', 
  'furthermore', 'hence', 'in front of', 'especially', 'for this reason', 'by and large', 'ultimately', 'in view of', 'in time', 'now', 'to be sure', 
  'with attention to', 'last', 'to sum up', 'by the same token', 'on the whole', 'different from', 'on the condition ', 'ordinarily', 'in spite of', 
  'as well as', 'immediately', 'at the present time', 'must be remembered', 'markedly', 'between', 'provided that', 'as a result', 'in the hope that', 
  'significantly', 'not only', 'to put it another way', 'for the most part', 'obviously', 'until now', 'equally important', 'from time to time', 
  'even though','even so', 'later', 'secondly', 'overall', 'in the event that', 'in either case', 'forthwith', 'suddenly', 'for instance', 'surely', 
  'in reality', 'granted', 'thirdly', 'above all', 'certainly', 'after all', 'adjacent to', 'in the background', 'despite', 'as shown above', 'presently', 
  'in the center of', 'so that', 'on the contrary', 'during', 'at this instant', 'to summarize', 'as soon as', 'uniquely', 'until', 'after', 'till', 
  'altogether', 'surprisingly', 'amid', 'in fact', 'for example', 'to put it differently', 'first thing to remember', 'unlike', 'in a moment', 'finally', 
  'about', 'in the first place', 'specifically', 'henceforth', 'eventually', 'be that as it may', 'in effect', 'in essence', 'at lest', 'in a word', 
  'next', 'to point out', 'first', 'point often overlooked', 'not to mention', 'all in all', 'meanwhile', 'across', 'when', 'including', 'equally', 
  'chiefly', 'then again', 'albeit', 'as has been noted', 'in the same fashion', 'in the meantime', 'although this may be true', 'likewise', 'truly', 
  'quickly', 'whenever', 'even though', 'before', 'occasionally', 'inasmuch as', 'as can be seen', 'as a matter of fact', 'under', 'with this intention', 
  'as an illustration', 'straightaway', 'to the left', 'nevertheless', 'in due time', 'in the long run', 'in the middle', 'important to realize', 
  'by all means', 'to the right', 'to demonstrate', 'in other words', 'to begin with', 'coupled with', 'on balance', 'together with', 'to emphasize', 
  'on the other hand', 'in that case', 'identically', 'with this in mind', 'behind', 'to say nothing of', 'for fear that', 'in conclusion', 'beyond', 
  'correspondingly', 'so long as', 'once', 'near', 'in short', 'shortly', 'below', 'in contrast', 'as much as', 'in the distance', 'in the foreground', 
  'opposite to ', 'in this case', 'another key point', 'since', 'nearby', 'beside', 'to explain', 'among', 'regardless', 'nonetheless', 'in case', 
  'now that', 'at the same time', 'for the purpose of', 'on the negative side', 'formerly', 'however', 'frequently', 'in brief', 'third', 'besides', 
  'in the same way', 'that is to say', 'here and there', 'down', 'up', 'alongside', 'then', 'otherwise', 'but also', 'without delay', 'thereupon', 'also', 
  'generally speaking', 'in general', 'given these points', 'moreover', 'under those circumstances', 'in addition', 'explicitly', 'in the final analysis', 
  'in any event', 'conversely', 'firstly', 'again', 'all things considered', 'definitely', 'instantly', 'above', 'and yet', 'expressly', 'as long as', 
  'to the end that', 'additionally', 'owing to', 'accordingly', 'in like manner', 'second', 'namely', 'in summary', 'of course', 'notwithstanding', 'in detail', 
  'to clarify', 'to enumerate'})

deny_phrases = set({"instead of", "rather than", "other than", "no longer", "in no way"})

deny_words = set({"cannot", "can't", "cant", "didn't", "didnt", "doesn't", "doesnt", "don't", "dont", "hasn't", "hasnt", "haven't", 
    "havent", "isn't", "isnt", "shouldn't", "shouldnt", "weren't", "werent", "without", "won't", "wont", "aren", "aren't", "arent", 
    "wasn't", "wasnt", "hardly", "never", "few", "little", "either", "neither", "no", "nobody", "non", "none", "nowhere", "nothing", 
    "nor", "seldom", "rarely", "merely", "scarcely", "barely"})

abbr_words = set({"2f4u", "4yeo", "fyeo", "aamof", "ack", "afaik", "afair", "afk", "aka", "b2k", "btk", "btt", "btw", "b/c", "c&p", 
    "cu", "cys", "diy", "eobd", "eod", "eom", "eot", "faq", "fack", "fka", "fwiw", "fyi", "jfyi", "ftw", "hf", "hth", "idk", "iirc", 
    "imho", "imo", "imnsho", "iow", "itt", "lol", "dgmw", "mmw", "n/a", "nan", "nntr", "noob", "n00b", "noyb", "nrn", "omg", "op", 
    "ot", "otoh", "pebkac", "pov", "rotfl", "rsvp", "rtfm", "scnr", "sflr", "spoc", "tba", "tbc", "tia", "tgif", "thx", "tnx", "tq", 
    "tyvm", "tyt", "ttyl", "w00t", "wfm", "wrt", "wth", "wtf", "ymmd", "ymmv", "yam", "icymi"})


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

    review = re.sub(r"[0-9{}\n\t\"#$%&()*+,-./:;<=>@[\]^_`{|}~]+", " ", review)

    review = review.lower()

    review = review.replace("?", " ?")
    review = review.replace("!", " !")

    for deny_phrase in deny_phrases:
      review = review.replace(deny_phrase, "not")

    for link_word in link_words:
      review = review.replace(link_word, " ")

    #Sample strengthen
    redundant_words = review.split(" ")
    review_length = len(redundant_words)
    for i in range(review_length):
      if redundant_words[i] in linked_verbs:
        j = 1
        while (i + j) < review_length and j <= 4:
          # repeat the word for 3 times
          redundant_words[i + j] = redundant_words[i + j] + " " + redundant_words[i + j]
          j += 1

    redundant_review = ""
    for word in redundant_words:
      redundant_review += word + " "    

    words = redundant_review.split(" ")
    simplified_words = []
    keep_prob = []
    for word in words:
      if word in deny_words:
        simplified_words.append("not")
        keep_prob.append(random())
      if word == '':
        continue
      if word in linked_verbs:
        continue
      if word in stop_words:
        continue
      if word in abbr_words:
        continue
      else:
        simplified_words.append(word)
        keep_prob.append(random())

    review_length = len(simplified_words)

    processed_review = []
    if review_length > MAX_WORDS_IN_REVIEW:
      sort_keep_prob = keep_prob[:]
      sort_keep_prob = sorted(sort_keep_prob)
      threshold = sort_keep_prob[review_length - MAX_WORDS_IN_REVIEW]

      for i in range(review_length):
        if keep_prob[i] >= threshold:
          processed_review.append(simplified_words[i])
    else:
      for i in range(review_length):
        processed_review.append(simplified_words[i])

    return processed_review



def layer(dropout_keep_prob):
    #GRU cell
    gru_cell = tf.contrib.rnn.GRUCell(hidden_size)
    #cell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE, initializer=tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32), forget_bias=1.0)
    # dropout input and output
    cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=dropout_keep_prob)
    return cell


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

    # input data, emdedded data already 
    input_data = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")
    # labels
    labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, num_class], name="labels")
    # drop out rate
    dropout_keep_prob = tf.placeholder_with_default(1-dropout, shape=[], name="dropout_keep_prob")
    #dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name="dropout_keep_prob")

    # if dropout < 1:
    #   input_data = tf.nn.dropout(input_data, dropout)

    
    ########################### basic lstm ###########################
    # multi layers
    cell = tf.contrib.rnn.MultiRNNCell([layer(dropout_keep_prob) for _ in range(num_layers)], state_is_tuple=True)
    # output and state
    outputs, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    # weight and bias
    weight = tf.Variable(tf.truncated_normal(shape=[hidden_size, num_class],stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_class]))


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
    '''



    # reshape output and get the last value 
    output = tf.transpose(outputs, [1, 0, 2])
    result = tf.gather(output, int(output.get_shape()[0]) - 1)
    logits = tf.matmul(result, weight) + bias
    # correct prediction
    correct = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    # accuracy: calculate the mean value (tf.cast: convert a data type to another data type)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    # use softmax cross entropy to compute loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), name="loss")
    # use adam optimizer model
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
