import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame
from pandas import concat

'''
def findMissingFrames(df):
    missing = []
    prev = -1
    for x in df.frame_number.unique():
        x = int(x)
        if x != prev + 1:
            for i in range(prev+1, x):
                missing.append(i)
                
        prev = x
                
    return missing
'''


# Load the data from .txt and return & store csv file
def getData(path: str, type = "obsmat"):
    """
    Load the data from .txt and return & store csv file

    Args:
        - path (string) -- path of the folder of the data to be processed
        - type (string) -- type of data (eg. obsmat)

    Returns:
        Preprocessed pandas DataFrame
    """

    if type == "obsmat":
        obsmat = path + "/" + type + ".txt"
        name = path.split("/")[-1]

        df = pd.read_csv(obsmat, delimiter=r"\s+", header = None)
        df.columns = ["frame_number", "pedestrian_ID", "pos_x", "pos_z", "pos_y", "v_x", "v_z", "v_y"]

        # Dropping irrelevent columns
        df = df.drop(["pos_z", "v_z"], axis = 1)

        # Converting the frames column into a proper range
        for i in range(len(df)):
            df.iloc[i,0] -= 1
            df.iloc[i,0] = df.iloc[i,0]/10

        # Columns Type Casting
        df = df.astype({'frame_number': int, 'pedestrian_ID' : int})

        df.to_csv(f"datasets/csvs/{name}.csv", index=False)
        return df

    else:
        print("Invalid Type")
        return None


# Convert normal series into a series suitable for LSTMs
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Convert normal series into a series suitable for LSTMs

    Args:
        - data (Pandas DataFrame) -- Sequence of observations as a list or NumPy array.
        - n_in (int)              -- Number of lag observations as input (X).
        - n_out (int)             -- Number of observations as output (y).
        - dropnan (int)           -- Boolean whether or not to drop rows with NaN values.

    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Get data that can be fed into a machine learning model
def getModelData(df, lstm_data):
    """
    Get data that can be fed into a machine learning model

    Args:
        - df (Pandas DataFrame) -- df of the position coords

    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    # Grouping the DataFrame and dropping peds with less data
    pos_df = df.drop(['v_x', 'v_y'], axis = 1)
    pos_grp = pos_df.groupby('pedestrian_ID')

    for name, group in pos_grp:
        if len(group) <= lstm_data['window_size'] + lstm_data['no_of_forecasts']:
            pos_df = pos_df.drop(pos_grp.get_group(name).index)
    pos_grp = pos_df.groupby('pedestrian_ID')

    df_lstm = pd.DataFrame()
    
    for name, group in pos_grp:
        pos_cols = group[['pos_x', 'pos_y']]
        pos_cols_converted = series_to_supervised(
            pos_cols, 
            lstm_data['window_size'],
            lstm_data['no_of_forecasts']
        )
        
        df_lstm = pd.concat([df_lstm, pos_cols_converted])
        
    df_lstm = df_lstm.dropna()
    df_lstm.to_csv(f"datasets/csvs/lstms/seq.csv", index=False)

    return df_lstm


# Creates a forecast of the trajectory of the pedestrian
def getForecast(model, df, window_size=3):
  """
  Creates a forecast of the trajectory of the pedestria

  Args:
    - model (tensorflow sequential) -- the trained model with 2 output tensors
    - df (pandas dataframe)         -- the part of the dataframe on which the predictions are to be made
    - size_of_prediction (int)      -- dimension of the prediction
    - window_size (int)             -- size of the window

  Returns:
    Numpy array with the predictions of the data

  Note:
    the prediction is expected to have only one pair of coordinates
  """

  series = []
  forecasts = []
  len_of_df = len(df)
  size_of_prediction = 2

  try:
    for i in range(window_size):
      for j in range(size_of_prediction):
        series.append(df.iloc[i][j])
  except IndexError:
    print(f"Length of the dataframe={len(df)} is smaller than the window_size={window_size}. Add more data or reduce the window_size")

  for i in range(len_of_df-window_size):
    predict = np.array(series[-window_size*size_of_prediction:])[np.newaxis]
    forecast = model.predict(predict[-window_size*size_of_prediction:][np.newaxis])
    forecasts.append(forecast[0])
    for j in range(size_of_prediction):
      series.append(forecast[0][j])

  return np.array(forecasts)


# Draws the predictions of the data
def drawPredictions(df, forecasts, window_size):
    """
    Draws the predictions of the data

    Args:
    - df (pandas dataframe) -- the part of the dataframe on which the predictions were made
    - forecasts (array of int) -- all the predictions from make_forecast()
    """
    x_val = [val[0] for val in forecasts]
    y_val = [val[1] for val in forecasts]

    b = plt.scatter(np.array(df.iloc[:window_size,0]), np.array(df.iloc[:window_size,1]), c='b')
    c = plt.scatter(np.array(df.iloc[window_size:,0]), np.array(df.iloc[window_size:,1]), c='c')
    o = plt.scatter(x_val, y_val, c='orange')
    plt.legend((b, c, o), ('before', 'after', 'prediction'))
    plt.show()



if __name__ == "__main__":
    file = open('data.json')
    json_data = json.load(file)

    dfRaw = getData(json_data['datasets']['seq_hotel'])
    # dfLSTM = getModelData(dfRaw, json_data)