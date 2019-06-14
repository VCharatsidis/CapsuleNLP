import pandas as pd
import config
import numpy as np
from nlp_utils import tokenize_sentences, read_embedding_list, clear_embedding_list, convert_tokens_to_ids
from train import get_model, train_folds
from scipy.stats import rankdata
from create_dataframe import to_dataFrame

# Load data
print("Loading data...")
# train_data = pd.read_csv(config.train_file_path)
# test_data = pd.read_csv(config.test_file_path)
train_data = to_dataFrame(config.train_file_path)
test_data = to_dataFrame(config.test_file_path)

list_sentences_train = train_data["application_text"].fillna(config.NAN_WORD).values
list_sentences_test = test_data["application_text"].fillna(config.NAN_WORD).values
y_train = train_data['Class'].values


print("Tokenizing sentences in train set...")
tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})
print("Tokenizing sentences in test set...")
tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)


# Embedding
words_dict[config.UNKNOWN_WORD] = len(words_dict)
print("Loading embeddings...")
embedding_list, embedding_word_dict = read_embedding_list(config.embedding_path)
embedding_size = len(embedding_list[0])

print("Preparing data...")
embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

embedding_word_dict[config.UNKNOWN_WORD] = len(embedding_word_dict)
embedding_list.append([0.] * embedding_size)
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

get_model_func = lambda: get_model(
    embedding_matrix,
    config.sentences_length,
    config.dropout_rate,
    config.recurrent_units,
    config.dense_size)


print("Starting to train models...")
models = train_folds(X_train, y_train, X_test, config.fold_count, config.batch_size, get_model_func)


base = "C:/Users/chara/Desktop/UvA/project/predictions/"
predict_list = []
for j in range(10):
    predict_list.append(np.load(base + "predictions_001/test_predicts%d.npy" % j))

print("Rank averaging on ", len(predict_list), " files")
predcitions = np.zeros_like(predict_list[0])
for predict in predict_list:
    predcitions = np.add(predcitions.flatten(), rankdata(predict) / predcitions.shape[0])
predcitions /= len(predict_list)



LABELS = ['DESC', 'ENTY', 'ABBR', 'HUM', 'NUM', 'LOC']

#LABELS = ["project_is_approved"]
# submission = pd.read_csv('../input/donorschoose-application-screening/sample_submission.csv')
# submission[LABELS] = predcitions
# submission.to_csv('submission.csv', index=False)