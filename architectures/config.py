train_file_path = "../../resources/train_TREC.txt"
#train_file_path = "C:/Users/chara/Desktop/UvA/project/donorschooseorg-preprocessed-data/train_small.csv"

test_file_path = "../../resources/test_TREC.txt"
#test_file_path = "C:/Users/chara/Desktop/UvA/project/donorschooseorg-preprocessed-data/test_small.csv"

embedding_path = "../../resources/embeddings_small.vec"

batch_size = 128
recurrent_units = 16
dropout_rate = 0.3
dense_size = 8
sentences_length = 20

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

TREC_CLASSES = {'DESC': 0, 'ENTY': 1, 'ABBR': 2, 'HUM': 3, 'NUM': 4, 'LOC': 5}

