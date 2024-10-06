# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:29:55 2023

@author: elusi
"""

import scipy.io
import numpy as np
import pandas as pd

mat = scipy.io.loadmat('C:/Users/elusi/OneDrive/Desktop/soot_proj/datasets/dataDifferentHeights2.mat')
df = mat['dataForEyal']
df_transposed = df.transpose(2,0,1)
df_split = np.split(df_transposed, 5000)

# Saving test 1
writer = pd.ExcelWriter('C:/Users/elusi/OneDrive/Desktop/soot_proj/datasets/dataDifferentHeights2.xlsx')
for array, i in zip(df_split, range(len(df_split))):
    array = array.reshape(112, 5)
    df = pd.DataFrame(array)
    df.to_excel(writer,'df ' + str(i),index=False,header=True)

writer.save()