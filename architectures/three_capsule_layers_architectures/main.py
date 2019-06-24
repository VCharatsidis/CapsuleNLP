from __future__ import division, print_function, unicode_literals
import argparse
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from architectures.three_capsule_layers_architectures.losses import spread_loss, cross_entropy, margin_loss
from architectures.three_capsule_layers_architectures.models import model_a, model_b, model_baseline_cnn
from keras import utils
from utils.nlp_utils import numeric_to_one_hot
from architectures.three_capsule_layers_architectures.capsule_layers import softmax
from architectures import config
import numpy as np
from utils.nlp_utils import tokenize_sentences, read_embedding_list, clear_embedding_list, convert_tokens_to_ids
import json
from utils.create_dataframe import to_dataFrame
from matplotlib import pyplot as plt
import capsule_layers

tf.reset_default_graph()
np.random.seed(0)
tf.set_random_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--embedding_type', type=str, default='static',
                    help='Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')

parser.add_argument('--dataset', type=str, default='resources/train_TREC.txt',
                    help='Options: reuters_multilabel_dataset, MR_dataset, SST_dataset')

parser.add_argument('--loss_type', type=str, default='margin_loss',
                    help='margin_loss, spread_loss, cross_entropy')

parser.add_argument('--model_type', type=str, default='capsule-A',
                    help='CNN, KIMCNN, capsule-A, capsule-B')

parser.add_argument('--has_test', type=int, default=1,
                    help='If data has test, we use it. Otherwise, we use CV on folds')
parser.add_argument('--has_dev', type=int, default=1, help='If data has dev, we use it, otherwise we split from train')

parser.add_argument('--num_epochs', type=int, default=120, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')

parser.add_argument('--use_orphan', type=bool, default='True', help='Add orphan capsule or not')
parser.add_argument('--use_leaky', type=bool, default='False', help='Use leaky-softmax or not')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')  # CNN 0.0005
parser.add_argument('--margin', type=float, default=0.2, help='the initial value for spread loss')



args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent=2))

#
# def load_data(dataset):
#     train, train_label = [], []
#     dev, dev_label = [], []
#     test, test_label = [], []
#
#     f = h5py.File(dataset + '.hdf5', 'r')
#     print('loading data...')
#     print(dataset)
#     print("Keys: %s" % f.keys())
#
#     w2v = list(f['w2v'])
#     train = list(f['train'])
#     train_label = list(f['train_label'])
#     if args.use_orphan:
#         args.num_classes = max(train_label) + 1
#
#     if len(list(f['test'])) == 0:
#         args.has_test = 0
#     else:
#         args.has_test = 1
#         test = list(f['test'])
#         test_label = list(f['test_label'])
#
#     for i, v in enumerate(train):
#         if np.sum(v) == 0:
#             del (train[i])
#             del (train_label[i])
#
#     for i, v in enumerate(test):
#         if np.sum(v) == 0:
#             del (test[i])
#             del (test_label[i])
#
#     train, dev, train_label, dev_label = train_test_split(train, train_label, test_size=0.1, random_state=0)
#     return train, train_label, test, test_label, dev, dev_label, w2v


class BatchGenerator(object):
    """Generate and hold batches."""

    def __init__(self, dataset, label, batch_size, input_size, is_shuffle=True):
        self._dataset = dataset
        self._label = label
        self._batch_size = batch_size
        self._cursor = 0
        self._input_size = input_size

        if is_shuffle:
            index = np.arange(len(self._dataset))
            np.random.shuffle(index)
            self._dataset = np.array(self._dataset)[index]
            self._label = np.array(self._label)[index]
        else:
            self._dataset = np.array(self._dataset)
            self._label = np.array(self._label)

    def next(self):
        if self._cursor + self._batch_size > len(self._dataset):
            self._cursor = 0
        """Generate a single batch from the current cursor position in the data."""
        batch_x = self._dataset[self._cursor: self._cursor + self._batch_size, :]
        batch_y = self._label[self._cursor: self._cursor + self._batch_size]
        self._cursor += self._batch_size
        return batch_x, batch_y


#train, train_label, test, test_label, dev, dev_label, w2v = load_data(args.dataset)



# Load data
print("Loading data...")
train_data = to_dataFrame(config.train_file_path)
test_data = to_dataFrame(config.test_file_path)

list_sentences_train = train_data["application_text"].fillna(config.NAN_WORD).values
list_sentences_test = test_data["application_text"].fillna(config.NAN_WORD).values
y_train = train_data['Class'].values
y_test = test_data['Class'].values


print("Tokenizing sentences in train set...")
tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})
print("Tokenizing sentences in test set...")
tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)


# Embedding
words_dict[config.UNKNOWN_WORD] = len(words_dict)
print("Loading embeddings...")
embedding_list, embedding_word_dict = read_embedding_list(config.embedding_path)
embedding_size = len(embedding_list[0])
print(len(embedding_list))
print(embedding_size)
print(embedding_word_dict)

print("Preparing data...")
embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

embedding_word_dict[config.UNKNOWN_WORD] = len(embedding_word_dict)
embedding_list.append(np.random.uniform(0, 1, embedding_size))
embedding_word_dict[config.END_WORD] = len(embedding_word_dict)
embedding_list.append([-1.] * embedding_size)

embedding_matrix = np.array(embedding_list)

id_to_word = dict((id, word) for word, id in words_dict.items())

train_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_train,
    id_to_word,
    embedding_word_dict,
    config.sentences_length)

test_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_test,
    id_to_word,
    embedding_word_dict,
    config.sentences_length)

X_train = np.array(train_list_of_token_ids)
X_test = np.array(test_list_of_token_ids)

args.vocab_size = len(words_dict.keys())
args.vec_size = embedding_matrix[1].shape[0]
args.max_sent = 20

print("embeding matrix "+str(len(words_dict.keys())))
print('max sent: ', args.max_sent)
print('vocab size: ', args.vocab_size)
print('vec size: ', args.vec_size)


with tf.device('/cpu:0'):
    global_step = tf.train.get_or_create_global_step()


label = ['DESC', 'ENTY', 'ABBR', 'HUM', 'NUM', 'LOC']
number_classes = len(label)
label = map(str, label)
args.max_sent = 20
threshold = 0.5

X = tf.placeholder(tf.int32, [args.batch_size, args.max_sent], name="input_x")
y = tf.placeholder(tf.int64, [args.batch_size, number_classes], name="input_y")

is_training = tf.placeholder_with_default(False, shape=())
learning_rate = tf.placeholder(dtype='float32')
margin = tf.placeholder(shape=(), dtype='float32')

l2_loss = tf.constant(0.0)

w2v = np.array(embedding_matrix, dtype=np.float32)
W1 = tf.Variable(w2v, trainable=False)
X_embedding = tf.nn.embedding_lookup(W1, X)
X_embedding = X_embedding[..., tf.newaxis]

print(X_embedding)

# if args.embedding_type == 'rand':
#     W1 = tf.Variable(tf.random_uniform([args.vocab_size, args.vec_size], -0.25, 0.25), name="Wemb")
#     X_embedding = tf.nn.embedding_lookup(W1, X)
#     X_embedding = X_embedding[..., tf.newaxis]
#
# if args.embedding_type == 'static':
#     W1 = tf.Variable(w2v, trainable=False)
#     X_embedding = tf.nn.embedding_lookup(W1, X)
#     X_embedding = X_embedding[..., tf.newaxis]
#
# if args.embedding_type == 'nonstatic':
#     W1 = tf.Variable(w2v, trainable=True)
#     X_embedding = tf.nn.embedding_lookup(W1, X)
#     X_embedding = X_embedding[..., tf.newaxis]
#
# if args.embedding_type == 'multi-channel':
#     W1 = tf.Variable(w2v, trainable=True)
#     W2 = tf.Variable(w2v, trainable=False)
#     X_1 = tf.nn.embedding_lookup(W1, X)
#     X_2 = tf.nn.embedding_lookup(W2, X)
#     X_1 = X_1[..., tf.newaxis]
#     X_2 = X_2[..., tf.newaxis]
#     X_embedding = tf.concat([X_1, X_2], axis=-1)

tf.logging.info("input dimension:{}".format(X_embedding.get_shape()))

poses, activations = model_b.get_model(X_embedding, number_classes)

# if args.model_type == 'capsule-A':
#     poses, activations = model_a.get_model(X_embedding, config.number_classes)
# if args.model_type == 'capsule-B':
#     poses, activations = model_b.get_model(X_embedding, config.number_classes)
# if args.model_type == 'CNN':
#     poses, activations = model_baseline_cnn.get_model(X_embedding, args.num_classes)

# if args.model_type == 'KIMCNN':
#     poses, activations = baseline_model_kimcnn(X_embedding, args.max_sent, args.num_classes)

#loss = spread_loss(y, activations, margin)
loss = margin_loss(y, activations)

# if args.loss_type == 'spread_loss':
#     loss = spread_loss(y, activations, margin)
# if args.loss_type == 'margin_loss':
#     loss = margin_loss(y, activations)
# if args.loss_type == 'cross_entropy':
#     loss = cross_entropy(y, activations)

y_pred = tf.argmax(activations, axis=1, name="y_proba")
correct = tf.equal(tf.argmax(y, axis=1), y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
gradients, variables = zip(*optimizer.compute_gradients(loss))

grad_check = [tf.check_numerics(g, message='Gradient NaN Found!') for g in gradients if g is not None] + \
             [tf.check_numerics(loss, message='Loss NaN Found')]

with tf.control_dependencies(grad_check):
    training_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)



sess = tf.InteractiveSession()


n_iterations_per_epoch = len(X_train) // args.batch_size
n_iterations_test = len(X_test) // args.batch_size
#n_iterations_dev = len(dev) // args.batch_size

mr_train = BatchGenerator(X_train, y_train, args.batch_size, 0)
#mr_dev = BatchGenerator(dev, dev_label, args.batch_size, 0)
mr_test = BatchGenerator(X_test, y_test, args.batch_size, 0, is_shuffle=False)

best_model = None
best_epoch = 0
best_acc_val = 0.

init = tf.global_variables_initializer()
sess.run(init)

def calc_accuracy(predictions, numeric_test_y):
    preds = np.argmax(predictions, 1)
    result = preds == numeric_test_y
    sum = np.sum(result)
    accuracy = sum / float(len(preds))

    return accuracy

lr = args.learning_rate
m = args.margin
accuracies = []
precisions = []
recalls = []
F1s =[]
for epoch in range(args.num_epochs):
    for iteration in range(1, n_iterations_per_epoch + 1):
        X_batch, y_batch = mr_train.next()
        y_batch = utils.to_categorical(y_batch, number_classes)

        _, loss_train, probs, capsule_pose = sess.run(
            [training_op, loss, activations, poses],
            feed_dict={X: X_batch[:, :args.max_sent],
                       y: y_batch,
                       is_training: True,
                       learning_rate: lr,
                       margin: m}
        )

        print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
            iteration, n_iterations_per_epoch,
            iteration * 100 / n_iterations_per_epoch,
            loss_train),
            end="")

    # loss_vals, acc_vals = [], []
    # for iteration in range(1, n_iterations_test + 1):
    #     X_batch, y_batch = mr_test.next()
    #     y_batch = utils.to_categorical(y_batch, number_classes)
    #
    #     loss_val, acc_val = sess.run(
    #         [loss, accuracy],
    #         feed_dict={X: X_batch[:, :args.max_sent],
    #                    y: y_batch,
    #                    is_training: False,
    #                    margin: m})
    #
    #     loss_vals.append(loss_val)
    #     acc_vals.append(acc_val)
    # loss_val, acc_val = np.mean(loss_vals), np.mean(acc_vals)
    # print("\rEpoch: {}  Val accuracy: {:.1f}%  Loss: {:.4f}".format(epoch + 1, acc_val * 100, loss_val))


    preds_list, y_list = [], []
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = mr_test.next()
        probs = sess.run([activations], feed_dict={X: X_batch[:, :args.max_sent], is_training: False})
        preds_list = preds_list + probs[0].tolist()
        y_list = y_list + y_batch.tolist()

    y_list = np.array(y_list)
    preds_probs = np.array(preds_list)

    exp_max = np.exp(preds_probs - np.max(preds_probs, axis=-1, keepdims=True))
    softmax_preds_probs = exp_max / np.sum(exp_max, axis=-1, keepdims=True)

    acc = calc_accuracy(softmax_preds_probs, y_list)

    preds_probs[np.where(preds_probs >= threshold)] = 1.0
    preds_probs[np.where(preds_probs < threshold)] = 0.0


    y_one_hot = numeric_to_one_hot(y_list, number_classes)
    [precision, recall, F1, support] = precision_recall_fscore_support(y_one_hot, preds_probs, average='samples')
    #acc = accuracy_score(y_one_hot, preds_probs)

    accuracies.append(acc)
    precisions.append(precision)
    recalls.append(recall)
    F1s.append(F1)

    print('\rAccuracy: %.3f' % acc, 'Precision: %.3f' % precision, 'Recall: %.3f' % recall, 'F1: %.3f' % F1)
    if args.model_type == 'CNN' or args.model_type == 'KIMCNN':
        lr = max(1e-6, lr * 0.8)
    if args.loss_type == 'margin_loss':
        m = min(0.9, m + 0.1)


print(max(accuracies))
plt.figure(1)
plt.plot(accuracies, label="Accuracies")
plt.plot(precisions, label="Precisions")
plt.plot(recalls, label="Recalls")
plt.plot(F1s, label="F1s")
plt.legend()

plt.figure(2)

plt.plot(accuracies)
plt.title("Accuracies")

plt.show()