import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv

#Load the train data
df_1 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_Test_old-bk .csv')
#x = df_1.ix[:,8]

f = lambda x: x.fillna(x.median(), inplace=True)
df_1.apply(f)

df_1.to_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Test.csv')

print "done"
'''
f = lambda x: x.fillna(x.mean())
df_1.apply(f)

df_1.to_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData .csv')

print "done"
'''

'''
feature_means = []
feature_medians = []

for i in range (2, 120):
    df_feature = df_1.ix[:,i]
    ft_mean = round(np.mean(df_feature), 5)
    ft_median = round(np.median(df_feature), 5)
    feature_means.append(ft_mean)
    feature_medians.append(ft_median)


data_reader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_Train_old-bk .csv','rb'))
open_file_object = csv.writer(open("C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData.csv", "wb"))
header = data_reader.next() #skip the first line of the test file.

for data in data_reader:
    for index in data:
        idx = 0
        val = str(data[idx])
        if val != "":
            data[idx] = feature_medians[idx-2]
        idx += 1
    open_file_object.writerow(data)

print "done"
'''
