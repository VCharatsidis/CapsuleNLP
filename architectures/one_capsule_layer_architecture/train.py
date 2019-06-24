import numpy as np
from utils.nlp_utils import numeric_to_one_hot
from matplotlib import pyplot as plt
number_classes = 6

def calc_accuracy(predictions, numeric_test_y):
    preds = np.argmax(predictions, 1)
    # print(preds)
    # print(numeric_val_y)
    result = preds == numeric_test_y
    sum = np.sum(result)
    accuracy = sum / float(len(preds))

    return accuracy


def train_model(model, batch_size, train_x, train_y, test_x, test_y):
    patience = 50
    best_loss = -1
    best_weights = None
    best_epoch = 0

    numeric_test_y = test_y
    print(numeric_test_y.shape)
    test_y = numeric_to_one_hot(test_y, number_classes)

    current_epoch = 0
    best_val_acc = 0
    accuracies = []
    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        predictions = model.predict(test_x, batch_size=batch_size)

        accuracy = calc_accuracy(predictions, numeric_test_y)

        # Cross Entropy Loss on the test set
        total_loss = np.mean(-test_y * np.log(predictions))

        # Check if the train should continue and save
        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == patience:
                break

        if accuracy > best_val_acc:
            best_val_acc = accuracy

        accuracies.append(accuracy)
        print("Epoch: {0} loss: {1} best_loss: {2} validation accuracy: {3}".format(current_epoch, total_loss, best_loss, accuracy))

    print(best_val_acc)
    plt.plot(accuracies)
    plt.title("Accuracies ")

    plt.show()
    model.set_weights(best_weights)
    return model





