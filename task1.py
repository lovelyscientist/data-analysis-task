import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv('./dataset_time_series.csv', decimal=",", skiprows=1, names=['N', 'Dat1', 'Dat2', 'N1', 'N2', 'N3', 'N4', 'N6', 'N7'] ,delimiter=';')
new_df = df.drop(['N1', 'N2', 'N3', 'N4', 'N6', 'N7'], axis=1)

print(new_df)

new_df['Dat1'] = pd.to_numeric(new_df['Dat1'], errors='coerce')
new_df['Dat2'] = pd.to_numeric(new_df['Dat2'], errors='coerce')

plt.plot(new_df['N'], new_df['Dat2'], linestyle='-')
plt.ylabel('some numbers')
plt.show()

print(new_df['Dat1'])
