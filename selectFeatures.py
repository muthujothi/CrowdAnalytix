import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv
from scipy.stats.stats import pearsonr
from collections import OrderedDict
from collections import defaultdict

#Load the train data
df_1 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Train.csv')

df_robbery_rate = df_1.ix[:,2]
#df_blacks = np.sqrt(df_1.ix[:,-1])
df_blacks = np.sqrt(df_1.ix[:,-1])


print pearsonr(df_blacks.values.reshape(-1,1), df_robbery_rate.values.reshape(-1,1))[0][0]
#print df_robbery_rate.name #get the name of the column like this

'''
feature_correlation = {}
for i in range (7, 120):
    df_feature = df_1.ix[:,i]
    ft_name = df_feature.name
    ft_corr = round(((pearsonr(df_feature.values.reshape(-1,1), df_robbery_rate.values.reshape(-1,1)))[0])[0], 4)
    feature_correlation[ft_name] = ft_corr

robbery_corr = OrderedDict(sorted(feature_correlation.items(), key=lambda t: (t[1])))
for k, v in robbery_corr.items():
    print k + "  *****  " + str(v)

'''
