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
    X = x*30+200
    Y = y*30+200
    print(X, Y)
    time.sleep(0.5)
    for event in pygame.event.get():  
        if event.type == pygame.QUIT:  
            done = True  
    # screen.fill(kBgColor)
    pygame.draw.rect(screen, kObjColor, pygame.Rect(X, Y, kObjSize[0], kObjSize[1]))    
  
    pygame.display.flip()