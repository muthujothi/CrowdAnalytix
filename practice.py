import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing

df_1 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Train.csv')

robbery_rate = df_1.ix[:,-3]
log_robbery_rate = np.log(1+robbery_rate)

plt.hist(robbery_rate)
plt.show()

scaler_y1 = preprocessing.StandardScaler().fit(robbery_rate)
scaled_robbery_rate = scaler_y1.transform(robbery_rate)
'''
plt.hist(robbery_rate)
plt.show()
'''
plt.hist(log_robbery_rate)
plt.show()

'''
plt.hist(np.log(1+scaled_robbery_rate))
plt.show()
'''
