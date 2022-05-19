import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras as keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, Softmax, Embedding, Flatten
from keras.activations import sigmoid
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

def Datavisualization(switch):
    if switch is True:
        print(dataset.groupby('HeartDisease').describe())
        print(dataset.groupby('Smoking').describe())
        print(dataset.groupby('AlcoholDrinking').describe())
        print(dataset.groupby('Stroke').describe())
        print(dataset.groupby('DiffWalking').describe())
        print(dataset.groupby('Sex').describe())
        print(dataset.groupby('AgeCategory').describe())
        print(dataset.groupby('Race').describe())
        print(dataset.groupby('Diabetic').describe())
        print(dataset.groupby('PhysicalActivity').describe())
        print(dataset.groupby('GenHealth').describe())
        print(dataset.groupby('Asthma').describe())
        print(dataset.groupby('KidneyDisease').describe())
        print(dataset.groupby('SkinCancer').describe())

def min_max_normalization(x):
    x_min = min(x)
    x_max = max(x)
    x = [(a - x_min)/(x_max - x_min) for a in x]
    return x

def categorical_embedding(embedding_size, input_dim, x, y):
    model_embedding = Sequential()
    model_embedding.add(Embedding(input_dim=input_dim, output_dim=embedding_size, input_length=1, name='embedding'))
    model_embedding.add(Flatten())
    model_embedding.add(Dense(50, activation='relu'))
    model_embedding.add(Dense(15, activation='relu'))
    model_embedding.add(Dense(1))
    optimizer1 = keras.optimizers.Adam(lr=0.0001)
    model_embedding.compile(loss='mse', optimizer=optimizer1, metrics=['accuracy'])
    model_embedding_result = model_embedding.fit(x, y, epochs=500, batch_size=1000)
    return model_embedding_result

def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return fig


def print_heatmap(labels, predictions, class_names):
    matrix = confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1))
    # row_sum = np.sum(matrix, axis=1)
    w, h = matrix.shape
    c_m = np.zeros((w, h))
    for i in range(h):
        c_m[i] = matrix[i]
    c = c_m.astype(dtype=np.uint8)
    heatmap = print_confusion_matrix(c, class_names, figsize=(18, 10), fontsize=20)


if __name__ == '__main__':
    dataset = pd.read_csv('heart_2020.csv', sep=',', header=0)
    sample = False
    model_training = True
    adaboost = False
    if sample:
        dataset_n = dataset.query('HeartDisease=="No"').sample(n=50000)
        dataset_y = dataset.query('HeartDisease=="Yes"').sample(n=20000)
        dataset_u = dataset_n.append(dataset_y)
    else:
        dataset_u = dataset
    X = dataset_u.drop(['HeartDisease'], axis=1)
    Y = dataset_u['HeartDisease']
    Datavisualization(switch=True)

    # Preprocessing

    X['BMI'] = min_max_normalization(X['BMI'])
    X['SleepTime'] = min_max_normalization(X['SleepTime'])
    X['PhysicalHealth'] = min_max_normalization(X['PhysicalHealth'])
    X['MentalHealth'] = min_max_normalization(X['MentalHealth'])
    onehot = OneHotEncoder()
    if adaboost:
        Y = Y.replace(['No', 'Yes'], [-1, 1])
    else:
        Y = onehot.fit_transform(np.expand_dims(Y, 1)).toarray()
    onehotlist = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
    for category in onehotlist:
        onehot = OneHotEncoder()
        X[category] = onehot.fit_transform(np.expand_dims(X[category], 1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.1)

    # print(y_test)
    # result_test = categorical_embedding(embedding_size=3, input_dim=13, x=X['AgeCategory'], y=Y)
    # print(result_test)

    # try a model

    if model_training:
        model = Sequential()
        model.add(Input(shape=(17,)))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        # model.add(Softmax())
        optimizer = keras.optimizers.Adam(lr=0.00001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        if adaboost:
            estimators = KerasRegressor(build_fn=model, epochs=300, batch_size=100, verbose=0)
            adaboost_model = AdaBoostRegressor(base_estimator=estimators)
            model_save = adaboost_model.fit(X_train, y_train)
            y_pred = adaboost_model.predict(X_test)
        else:
            model_save = model.fit(X_train, y_train, epochs=20, batch_size=100)
            y_pred = model.predict(X_test)
        # model_save = model.fit(X_train, y_train, epochs=200, batch_size=)
        class_names = ['No', 'Yes']


        print_heatmap(y_test, y_pred, class_names)