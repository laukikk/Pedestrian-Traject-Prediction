import tensorflow as tf
import json
import pandas as pd

import functions as f

file = open('data.json')
json_data = json.load(file)

model = tf.keras.models.load_model("models/lstm-5-3")
df = pd.read_csv('datasets/csvs/combined.csv')
print(df)
