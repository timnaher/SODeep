#%%
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
import os

class ZScoreTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

def population_zscore_transform(mean=0, std=50):
    return {"transform": ZScoreTransform(mean, std)}



# %%
