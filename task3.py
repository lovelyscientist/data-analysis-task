import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv('./dataset_time_series.csv', decimal=",", skiprows=1, names=['N', 'Dat1', 'Dat2', 'N1', 'N2', 'N3', 'N4', 'N6', 'N7'] ,delimiter=';')
new_df = df.drop(['N1', 'N2', 'N3', 'N4', 'N6', 'N7'], axis=1)

print(new_df)

new_df['Dat1'] = pd.to_numeric(new_df['Dat1'], errors='coerce')
new_df['Dat2'] = pd.to_numeric(new_df['Dat2'], errors='coerce')

rolmean = pd.rolling_mean(new_df['Dat2'], window=10)
rolstd = pd.rolling_std(new_df['Dat2'], window=10)

plt.plot(new_df['N'], new_df['Dat2'], color='blue', label='Original')
plt.plot(new_df['N'], rolmean, color='red', label='Rolling Mean')
plt.plot(new_df['N'], rolstd, color='green', label='Std Mean')

turnsCount = 0
index = 0
n = 2000

for row in new_df['Dat2']:
  if index > 0 and index < n-2:
    if (new_df['Dat2'][index] < new_df['Dat2'][index+1] and new_df['Dat2'][index+1] > new_df['Dat2'][index+2]):
         turnsCount = turnsCount + 1
    if (new_df['Dat2'][index] > new_df['Dat2'][index+1] and new_df['Dat2'][index+1] < new_df['Dat2'][index+2]):
        turnsCount = turnsCount + 1
  index = index + 1

print(turnsCount)

turnsCount = 0
index = 0
n = 2000

for row in new_df['Dat1']:
  if index > 0 and index < 20:
    if (new_df['Dat1'][index] < new_df['Dat1'][index+1] and new_df['Dat1'][index+1] > new_df['Dat1'][index+2]):
         turnsCount = turnsCount + 1
    if (new_df['Dat1'][index] > new_df['Dat1'][index+1] and new_df['Dat1'][index+1] < new_df['Dat1'][index+2]):
        turnsCount = turnsCount + 1
    print(new_df['Dat1'][index])
    print(new_df['Dat1'][index+1])
    print(new_df['Dat1'][index+2])
  index = index + 1

print(turnsCount)

print("Вариант 3. Используются фунция sin")
    n = 1000
    a = []
    for i in xrange(0, n):
        a.append(math.sin(i/math.pi))
    print("Сегерировано %d значений" % n)
    k, k0, d, dk95 = turningPointsCounter(n, a)
    printResult(k, k0, dk95)