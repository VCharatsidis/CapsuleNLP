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


def train_folds(X, y, X_test, y_test, fold_count, batch_size, get_model_func):
    print("="*75)
    fold_size = len(X)

    model = _train_model(get_model_func(), batch_size, X, y, X_test, y_test)
    return model


def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    num_labels = 6
    #num_labels = train_y.shape[1]
    patience = 7
    best_loss = -1
    best_weights = None
    best_epoch = 0

    onehot_encoded = []
    for value in val_y:
        letter = [0 for _ in range(6)]
        letter[int(value)] = 1
        onehot_encoded.append(letter)

    val_y = np.array(onehot_encoded)

    print(val_y.shape)
    print(val_y[0])

    current_epoch = 0

    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)

        # Cross Entropy Loss
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
    output = Dense(6, activation='softmax')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model