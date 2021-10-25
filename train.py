import tensorflow as tf
from keras.layers import *
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd

np.random.seed(777)
flag_value = -10
max_batch = 0

X_train = []
Y_train = []
X_test = []
Y_test = []


def load():
    df = pd.read_csv('data/train.csv',
                     usecols=['ID', 'USERID', 'EVENTID', 'TIMESTAMP', 'LABEL'])

    num_user = df['USERID'].max() + 1
    global max_batch
    max_batch = 0

    x_train = []
    # Initialize all y value to 0
    y_train = np.zeros((num_user, 1))

    # Data normalize
    scaler = MinMaxScaler()
    df.iloc[:, 2:4] = scaler.fit_transform(df.iloc[:, 2:4].to_numpy())
    # Data standardize
    scaler = StandardScaler()
    df.iloc[:, 2:4] = scaler.fit_transform(df.iloc[:, 2:4].to_numpy())

    print('\nSplitting data')
    for i in range(num_user):
        rows = df.loc[df['USERID'] == i]
        # Group records by user, and use only event and timestamp from each sequence to determine label
        # Each element of x_train (or each unique user) contains a number of arrays (for each event), which contains 2 numbers: eventid and timestamp
        x_rows = rows.iloc[:, 2:4].to_numpy()
        x_train.append(x_rows)

        # Each group corresponds to 1 label, which indicates whether or not the user closed their account after their final action
        y_rows = rows['LABEL'].to_numpy()
        if [1] in y_rows:
            y_train[i] = np.ones(1)

        # Get the max number of records of any user
        a, b = zip(*x_rows)
        if max_batch < len(a):
            max_batch = len(a)

    # Pad the training data so each group of records have same number of arrays
    xpad = np.full((num_user, max_batch, 2), fill_value=flag_value)
    for i, x in enumerate(x_train):
        seq = x.shape[0]
        xpad[i, 0:seq, :] = x

    # Shuffle the data
    p = np.random.permutation(num_user)
    xpad = xpad[p]
    y_train = y_train[p]

    # Split training and testing data
    global X_train, Y_train, X_test, Y_test
    X_train = xpad[6000:]
    Y_train = y_train[6000:]
    X_test = xpad[:6000]
    Y_test = y_train[:6000]


def create():
    # Create sequential model
    model = Sequential()
    # Add masking layer to filter out the padded arrays
    model.add(Masking(mask_value=flag_value, input_shape=(max_batch, 2)))
    # Add lstm to use previous records and current record to determine label
    model.add(LSTM(50))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.5))
    # Add dense layer with sigmoid activation to get the binary classification result
    model.add(Dense(1, activation='sigmoid'))

    # Initialize optimizer
    opt = tf.keras.optimizers.RMSprop(
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False,
        name="RMSprop",
    )

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy',
                                                                      tf.keras.metrics.TruePositives(),
                                                                      tf.keras.metrics.TrueNegatives(),
                                                                      tf.keras.metrics.FalsePositives(),
                                                                      tf.keras.metrics.FalseNegatives(), ])
    print(model.summary())
    return model


def test(model):
    print('\nTraining model')
    # Train for 100 epochs on batch size 32
    model.fit(X_train, Y_train, batch_size=32, epochs=100, shuffle=True, verbose=1)

    print('Testing model')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (score[1] * 100))


def save(model):
    model.save('model.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
