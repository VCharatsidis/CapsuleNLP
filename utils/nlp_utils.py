import nltk
import tqdm
import numpy as np
import numpy as np
import h5py
import re
import operator

import sys


from nltk import word_tokenize
from nltk.corpus import reuters
from nltk.corpus import stopwords

nltk.download("punkt")


def load_word_vector(fname, vocab):
    model = {}
    with open(fname) as fin:
        for line_no, line in enumerate(fin):
            try:
                parts = line.strip().split(' ')
                word, weights = parts[0], parts[1:]
                if word in vocab:
                    model[word] = np.array(weights, dtype=np.float32)
            except:
                pass
    return model


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')
            else:
               f.read(binary_len)
    return word_vecs


def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []
    f = open(file_path, encoding="utf8")

    for index, line in enumerate(f):
        if index == 0:
            continue
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue

        embedding_list.append(coefs)
        embedding_word_dict[word] = len(embedding_word_dict)

    f.close()
    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)

    return words_train


def numeric_to_one_hot(test_y, number_classes):
    onehot_encoded = []
    for value in test_y:
        letter = [0 for _ in range(number_classes)]
        letter[int(value)] = 1
        onehot_encoded.append(letter)

    return np.array(onehot_encoded)

