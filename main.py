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

# Getting the Data
seq_hotel = "datasets/ewap_dataset/seq_hotel"
seq_eth   = "datasets/ewap_dataset/seq_eth"
hotel_obsmat = seq_hotel + "/obsmat.txt"

df_obsmat = pd.read_csv(hotel_obsmat, delimiter=r"\s+", header = None)
df_obsmat.columns = ["frame_number", "pedestrian_ID", "pos_x", "pos_z", "pos_y", "v_x", "v_z", "v_y"]

# Constants
df = f.preProcessDF(df_obsmat)
missing = f.findMissingFrames(df)

# Taking a pedestrian
pedestrian = df[df.pedestrian_ID == 25]
ped_x = list(pedestrian.pos_x)
ped_y = list(pedestrian.pos_y)

# Pygame
pygame.init()  
screen = pygame.display.set_mode((500, 500))
screen.fill(kBgColor)
done = False  
  
for x, y in zip(ped_x, ped_y):
    time.sleep(0.5)
    for event in pygame.event.get():  
        if event.type == pygame.QUIT:  
            done = True  
    screen.fill(kBgColor)
    pygame.draw.rect(screen, kObjColor, pygame.Rect(x*10+200, y+200, kObjSize[0], kObjSize[1]))    
  
    pygame.display.flip()