import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
#GBM model performed bad with single independentvariable
#from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from scipy import sparse
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df_1 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Train_FullModel.csv')

df_XTrain_Year = df_1.ix[:,6] 
df_XTrain = df_1.ix[:,7:]
scaler_x = preprocessing.StandardScaler().fit(df_XTrain)#0 starts from population
np_XTrain = scaler_x.transform(df_XTrain)


df_YTrain_Robbery = df_1.ix[:,1]
np_YTrain = np.log(1+df_YTrain_Robbery)
Y1_Train = np_YTrain #This will leave the array in 1-dimensional and reshape will make something else. Under what is the diff between (503L,) and (503,1)

X_new = SelectKBest(chi2, k=2).fit_transform(np_XTrain, Y1_Train)
print X_new
