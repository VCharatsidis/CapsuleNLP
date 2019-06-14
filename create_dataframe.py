import config
import pandas as pd

def to_dataFrame(filepath):
    f = open(filepath, "r")
    contents = f.readlines()

    data = []
    labels = {}
    for line in contents:

        X = line.split(' ', 1)
        print(X)
        X[1] = X[1].strip('\n')

        label = X[0].split(':')[0]

        X[0] = int(config.TREC_CLASSES[label])

        data.append(X)
        print(X)
    print(labels)
    df = pd.DataFrame(data, columns=['Class', 'application_text'])
    return df

#to_dataFrame(config.train_file_path)