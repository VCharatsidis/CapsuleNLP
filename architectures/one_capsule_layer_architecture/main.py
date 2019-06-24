from architectures import config
import numpy as np
from utils.nlp_utils import tokenize_sentences, read_embedding_list, clear_embedding_list, convert_tokens_to_ids
from architectures.one_capsule_layer_architecture.train import train_model
from architectures.one_capsule_layer_architecture.initial_model import get_model
from utils.create_dataframe import to_dataFrame

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
print(X_train)
print(X_test)

get_model_func = lambda: get_model(embedding_matrix, config.sentences_length)

print("Starting to train model...")
model = train_model(get_model_func(), config.batch_size, X_train, y_train, X_test, y_test)


