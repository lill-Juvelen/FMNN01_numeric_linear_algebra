# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 08:58:02 2019

@author: lill-Juvelen
"""
import numpy as np

data = np.genfromtxt('../signal.dat',
                     skip_header=1,
                     skip_footer=1,
                     names=True,
                     dtype=None)

#%%
import numpy as np
import pandas as pd

df = pd.read_csv('signal.csv')
data = df.values
#for ind, dat in enumerate(data):
 #   x[ind] = data[1][ind]