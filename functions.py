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

def getData(path: str, type = "obsmat"):
    """
    Get data that can be fed into a machine learning model

    Arguments:
        path (string) -- path of the folder of the data to be processed
        type (string) -- type of data (eg. obsmat)
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


def getModelData(df, lstm_data):
    """
    Get data that can be fed into a machine learning model

    Arguments:
        df (Pandas DataFrame) -- df of the position coords
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


if __name__ == "__main__":
    file = open('data.json')
    json_data = json.load(file)

    dfRaw = getData(json_data['datasets']['seq_hotel'])
    # dfLSTM = getModelData(dfRaw, json_data)