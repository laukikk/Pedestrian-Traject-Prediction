import pandas as pd
import json

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

def getData(data, type):
    if type == "obsmat":
        hotel_obsmat = data['datasets']['seq_hotel'] + '/' + data['datasets']['type'] + ".txt"
        df = pd.read_csv(hotel_obsmat, delimiter=r"\s+", header = None)
        df.columns = ["frame_number", "pedestrian_ID", "pos_x", "pos_z", "pos_y", "v_x", "v_z", "v_y"]

        # Dropping irrelevent columns
        df = df.drop(["pos_z", "v_z"], axis = 1)

        # Converting the frames column into a proper range
        for i in range(len(df)):
            df.iloc[i,0] -= 1
            df.iloc[i,0] = df.iloc[i,0]/10

        # Columns Type Casting
        df = df.astype({'frame_number': int, 'pedestrian_ID' : int})

        df.to_csv(f"datasets/csvs/{data['datasets']['name']}.csv", index=False)

        ### Converting to LSTM suited data
        if (data['datasets']['lstm']):
            lstm = data['lstm']
            df_lstm = series_to_supervised(df, lstm['n_in'], lstm['n_out'])
            df_lstm.to_csv(f"datasets/csvs/lstms/{data['datasets']['name']}.csv", index=False)




    else:
        print("Invlaid Type")

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
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

if __name__ == "__main__":
    f = open('data.json')
    data = json.load(f)

    getData(data, "obsmat")