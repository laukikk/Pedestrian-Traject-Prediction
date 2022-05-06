import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Getting the Data
seq_hotel = "datasets/ewap_dataset/seq_hotel"
seq_eth   = "datasets/ewap_dataset/seq_eth"
hotel_obsmat = seq_hotel + "/obsmat.txt"

df_obsmat = pd.read_csv(hotel_obsmat, delimiter=r"\s+", header = None)
df_obsmat.columns = ["frame_number", "pedestrian_ID", "pos_x", "pos_z", "pos_y", "v_x", "v_z", "v_y"]

