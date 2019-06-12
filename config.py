
train_file_path = "C:/Users/chara/Desktop/UvA/project/donorschooseorg-preprocessed-data/train_preprocessed.csv"
#train_file_path = "C:/Users/chara/Desktop/UvA/project/donorschooseorg-preprocessed-data/train_small.csv"

test_file_path = "C:/Users/chara/Desktop/UvA/project/donorschooseorg-preprocessed-data/test_preprocessed.csv"
#test_file_path = "C:/Users/chara/Desktop/UvA/project/donorschooseorg-preprocessed-data/test_small.csv"

embedding_path = "C:/Users/chara/Desktop/UvA/project/donorschooseorg-preprocessed-data/embeddings_small.vec"

batch_size = 128
recurrent_units = 16
dropout_rate = 0.3
dense_size = 8
sentences_length = 10
fold_count = 2

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
CLASSES = ["project_is_approved"]

