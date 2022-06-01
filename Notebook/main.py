import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras as keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, Embedding, Flatten
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from keras.callbacks import EarlyStopping

def min_max_normalization(x):
    x_min = min(x)
    x_max = max(x)
    x = [(a - x_min)/(x_max - x_min) for a in x]
    return x


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
    # matrix = confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1))
    matrix = confusion_matrix(labels, predictions)
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
    adaboost = True
    onehotencode = False
    dataset_u = dataset
    X = dataset_u.drop(['HeartDisease'], axis=1)
    Y = dataset_u['HeartDisease']

    X['BMI'] = min_max_normalization(X['BMI'])
    X['SleepTime'] = min_max_normalization(X['SleepTime'])
    X['PhysicalHealth'] = min_max_normalization(X['PhysicalHealth'])
    X['MentalHealth'] = min_max_normalization(X['MentalHealth'])
    if onehotencode:
        onehot = OneHotEncoder()
        if adaboost:
            Y = Y.replace(['No', 'Yes'], [-1, 1])
        else:
            Y = onehot.fit_transform(np.expand_dims(Y, 1)).toarray()
        onehotlist = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
        for category in onehotlist:
            onehot = OneHotEncoder()
            X[category] = onehot.fit_transform(np.expand_dims(X[category], 1)).toarray()
    else:
        X.loc[:, 'Diabetic'] = X.loc[:, 'Diabetic'].replace(
            ['Yes (during pregnancy)', 'No, borderline diabetes'], ['Yes', 'No'])
        X.loc[:, 'Race'] = X.loc[:, 'Race'].replace(
            ['Hispanic', 'Black', 'Asian', 'American Indian/Alaskan Native'], 'Other')
        for i in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex', 'Race', 'AgeCategory']:
            X[i] = LabelEncoder().fit_transform(X[i])
        if adaboost:
            # 1 dimension
            Y = LabelEncoder().fit_transform(Y)
        else:
            # 2 dimension
            onehot = OneHotEncoder()
            Y = onehot.fit_transform(np.expand_dims(Y, 1)).toarray()
        X.loc[:, 'GenHealth'] = X.loc[:, 'GenHealth'].replace(['Poor', 'Fair', 'Good', 'Very good', 'Excellent'], [0, 1, 2, 3, 4])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

    # try a model

    if model_training:
        model = Sequential()
        model.add(Input(shape=(17,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        optimizer = keras.optimizers.Adam(lr=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        if adaboost:
            early_stopping = EarlyStopping()
            model = AdaBoostClassifier(n_estimators=100, learning_rate=1.65, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            early_stopping = EarlyStopping()
            model_save = model.fit(X_train, y_train, epochs=100, batch_size=200, validation_split=0.25, callbacks=[early_stopping])
            y_pred = model.predict(X_test)

        class_names = ['No', 'Yes']
        print_heatmap(y_test, y_pred, class_names)
