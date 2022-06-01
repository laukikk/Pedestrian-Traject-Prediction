import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pygame
import time
from constants import *
import functions as f

# Getting the Data (unused)
seq_hotel = "datasets/ewap_dataset/seq_hotel"
seq_eth   = "datasets/ewap_dataset/seq_eth"
hotel_obsmat = seq_hotel + "/obsmat.txt"

# Constants
df = pd.read_csv('datasets\ewap_dataset\obsmat_hotel.csv')

