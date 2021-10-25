import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd


def label():
    # Load saved model
    model = tf.keras.models.load_model('model.h5')

    df = pd.read_csv('data/test.csv',
                     usecols=['ID', 'USERID', 'EVENTID', 'TIMESTAMP', 'LABEL'])

    users = df['USERID'].tolist()
    users = list(dict.fromkeys(users))
    num_user = len(users)
    max_batch = 0

    x_train = []

    # Data normalize
    scaler = MinMaxScaler()
    df.iloc[:, 2:4] = scaler.fit_transform(df.iloc[:, 2:4].to_numpy())
    # Data standardize
    scaler = StandardScaler()
    df.iloc[:, 2:4] = scaler.fit_transform(df.iloc[:, 2:4].to_numpy())

    for i in users:
        rows = df.loc[df['USERID'] == i]
        # Group records by user, and use only event and timestamp from each sequence to determine label
        x_rows = rows.iloc[:, 2:4].to_numpy()
        x_train.append(x_rows)

        a, b = zip(*x_rows)
        if max_batch < len(a):
            max_batch = len(a)

    # Pad the training data so each group of records have same number of arrays
    flag_value = -10
    xpad = np.full((num_user, max_batch, 2), fill_value=flag_value)
    for i, x in enumerate(x_train):
        seq = x.shape[0]
        xpad[i, 0:seq, :] = x

    y_result = model.predict(xpad, batch_size=32)

    # Round prediction to 0 or 1
    y_result = [round(num[0]) for num in y_result]
    df = pd.read_csv('data/test.csv',
                     usecols=['ID', 'USERID', 'EVENTID', 'TIMESTAMP', 'LABEL'])

    # Initialize all labels t0 0
    df['LABEL'] = 0
    i = 0

    # If y_result[i] == 1, then set the label of the last record of that user to 1
    for u in users:
        if y_result[i] == 1:
            row_i = int(df.loc[df['USERID'] == u].tail(1)['ID']) - 391749
            df.at[row_i, 'LABEL'] = 1

        i += 1

    # Write dataframe to csv
    df.to_csv('labelled.csv', sep=',', index=False, encoding='utf-8')
    print('Test file has been updated')
