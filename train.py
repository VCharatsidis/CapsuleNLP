import numpy as np
import os

from keras.layers import SpatialDropout1D
from keras.models import Model
from sklearn.metrics import log_loss
from keras.layers import Bidirectional, Dense, Dropout, Embedding, Flatten
from keras.layers import GRU, Input

from Capsule import Capsule

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.3
rate_drop_dense = 0.3


def train_folds(X, y, X_test, fold_count, batch_size, get_model_func):
    print("="*75)
    fold_size = len(X) // fold_count
    models = []
    result_path = "predictions"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = np.array(X[fold_start:fold_end])
        val_y = np.array(y[fold_start:fold_end])

        model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y)
        train_predicts_path = os.path.join(result_path, "train_predicts{0}.npy".format(fold_id))
        test_predicts_path = os.path.join(result_path, "test_predicts{0}.npy".format(fold_id))
        train_predicts = model.predict(X, batch_size=512, verbose=1)
        test_predicts = model.predict(X_test, batch_size=512, verbose=1)
        np.save(train_predicts_path, train_predicts)
        np.save(test_predicts_path, test_predicts)

    return models


def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    num_labels = train_y.shape[1]
    patience = 5
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_loss = 0
        for j in range(num_labels):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= num_labels

        print("Epoch {0} loss {1} best_loss {2}".format(current_epoch, total_loss, best_loss))

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == patience:
                break

    model.set_weights(best_weights)
    return model


def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input1 = Input(shape=(sequence_length,))
    embed_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input1)

    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)

    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)

    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(1, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model