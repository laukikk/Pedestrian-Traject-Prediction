import pygame
import time
from constants import *


# Pygame
pygame.init()  
screen = pygame.display.set_mode((400, 300))
screen.fill(kBgColor)
done = False  
  
for x in range(100):
    time.sleep(0.5)
    for event in pygame.event.get():  
        if event.type == pygame.QUIT:  
            done = True  
    screen.fill(kBgColor)
    pygame.draw.rect(screen, kObjColor, pygame.Rect(x, x, kObjSize[0], kObjSize[1]))    
  
    pygame.display.flip()