# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:58:20 2021

@author: siyer
"""
import pandas as pd

data = pd.read_csv('data/train_contexts.csv')
s = data.iloc[0,2]

print(s[269:])