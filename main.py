import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pygame
import time
import json

import functions as f
import models as f

# Get the data
file = open('data.json')
json_data = json.load(file)

df_lstm = pd.DataFrame()
for path in json_data['datasets']:
    df_raw = f.getData(path)
    df_converted = f.getModelData(df_raw, json_data['models']['lstms'])
    df_lstm = pd.concat([df_lstm, df_converted])

