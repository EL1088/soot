# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:29:55 2023

@author: elusi
"""

import scipy.io
import numpy as np
import pandas as pd

mat = scipy.io.loadmat('C:/Users/elusi/OneDrive/Desktop/soot_proj/datasets/dataDifferentHeights2_121024.mat')

data_key = list(mat.keys())[3] # Getting the data 'key' (header of the mat database)
df = mat[data_key] # Exctracting the database out of mat
df_transposed = df.transpose(2,0,1)
df_split = np.split(df_transposed, 17500) # Split to a list of numpy arrays. Needs to be the number of samples in the database, i.e. 10,000

# Writing df as .xlsx seperating numpy arrays (samples) by worksheet
writer = pd.ExcelWriter('C:/Users/elusi/OneDrive/Desktop/soot_proj/datasets/dataDifferentHeights2_121024.xlsx')
for array, i in zip(df_split, range(len(df_split))):
    num_pixels = df_transposed.shape[1] # Getting the number of pixels in the samples
    array = array.reshape(num_pixels, 5)
    df = pd.DataFrame(array)
    df.to_excel(writer,'df ' + str(i),index=False,header=True)

writer.save()
