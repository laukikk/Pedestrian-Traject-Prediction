import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def model_lstm(df_lstm, json_data):
    data_models = json_data['models']
    window_size = data_models['lstm']['window_size']
    no_of_forecasts = data_models['lstm']['no_of_forecasts']
    
    X = df_lstm.iloc[:,:window_size*2].values
    y = df_lstm.iloc[:,window_size*2:].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32),

        tf.keras.layers.Dense(8)
    ])

    model.compile(loss='mae', optimizer='adam', metrics='acc')

    history = model.fit(X_train, y_train, epochs=data_models['epochs'], batch_size=data_models['batch_size'], validation_data=(X_test, y_test), shuffle=False)
