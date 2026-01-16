import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('bengaluru_house_prices.csv')
print(df.head())
# dropping useless columns
df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
print(df2.head())
