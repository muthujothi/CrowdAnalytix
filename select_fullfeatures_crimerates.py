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
from sklearn import tree

#Load the train data
df_1 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Train_FullModel.csv')

df_XTrain_Year = df_1.ix[:,6] #504 X 1
df_XTrain = df_1.ix[:,7:] #504 X 116
scaler_x = preprocessing.StandardScaler().fit(df_XTrain)#0 starts from population
np_XTrain = scaler_x.transform(df_XTrain)

#Now Start
#pca = PCA(n_components = 10)
#np_XTrain = pca.fit_transform(df_XTrain)
#scaler_x = preprocessing.StandardScaler().fit(df_XTrain)#0 starts from population
#np_XTrain = scaler_x.transform(df_XTrain)
#Now end

np_XTrain_Year = df_XTrain_Year.values.reshape(-1, 1)
XTrain = np.hstack((np_XTrain_Year, np_XTrain)) #504 X 117
#print XTrain.shape

#df_YTrain = df_1.ix[:,1:6]#start inclusive and end exclusive #504 X 5
#print df_XTrain_Year.shape, df_XTrain.shape, df_YTrain.shape
#print scaler.mean_.shape
#print scaler.mean_[0]#print XTrain.shape #504 X117 #print Y1.shape #504 X 1
df_YTrain_Robbery = df_1.ix[:,1]#start inclusive and end exclusive #504 X 1
scaler_y1 = preprocessing.StandardScaler().fit(df_YTrain_Robbery)#0 start from robbery
np_YTrain = scaler_y1.transform(df_YTrain_Robbery)
Y1_Train = np_YTrain.reshape(-1, 1)

df_YTrain_Property = df_1.ix[:,2]#start inclusive and end exclusive #504 X 1
scaler_y2 = preprocessing.StandardScaler().fit(df_YTrain_Property)#0 start from robbery
np_Y2Train = scaler_y2.transform(df_YTrain_Property)
Y2_Train = np_Y2Train.reshape(-1, 1)

df_YTrain_Burgalry = df_1.ix[:,3]#start inclusive and end exclusive #504 X 1
scaler_y3 = preprocessing.StandardScaler().fit(df_YTrain_Burgalry)#0 start from robbery
np_Y3Train = scaler_y3.transform(df_YTrain_Burgalry)
Y3_Train = np_Y3Train.reshape(-1, 1)

df_YTrain_Larceny = df_1.ix[:,4]#start inclusive and end exclusive #504 X 1
scaler_y4 = preprocessing.StandardScaler().fit(df_YTrain_Larceny)#0 start from robbery
np_Y4Train = scaler_y4.transform(df_YTrain_Larceny)
Y4_Train = np_Y4Train.reshape(-1, 1)

df_YTrain_Motor = df_1.ix[:,5]#start inclusive and end exclusive #504 X 1
scaler_y5 = preprocessing.StandardScaler().fit(df_YTrain_Motor)#0 start from robbery
np_Y5Train = scaler_y5.transform(df_YTrain_Motor)
Y5_Train = np_Y5Train.reshape(-1, 1)
#print scaler_y5.inverse_transform(Y5_Train)
#print scaler_y.inverse_transform(Y1_Train)


#(XTrain, Y1_Train)
print XTrain.shape, Y1_Train.shape
print "Fitting model for Crime: Robbery"
reg_robbery_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_robbery_rate = tree.DecisionTreeRegressor()
reg_robbery_rate.fit(XTrain, Y1_Train)

print "Fitting model for Crime: Property"
reg_propertycrime_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_propertycrime_rate = tree.DecisionTreeRegressor()
reg_propertycrime_rate.fit(XTrain, Y2_Train)

print "Fitting model for Crime: Burgalry"
reg_burgalry_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_burgalry_rate = tree.DecisionTreeRegressor()
reg_burgalry_rate.fit(XTrain, Y3_Train)

print "Fitting model for Crime: Larceny"
reg_larceny_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_larceny_rate = tree.DecisionTreeRegressor()
reg_larceny_rate.fit(XTrain, Y4_Train)

print "Fitting model for Crime: Motor"
reg_motor_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_motor_rate = tree.DecisionTreeRegressor()
reg_motor_rate.fit(XTrain, Y5_Train)


'''
print "Predicting on the Train Set using the trained model"
Y1_Predict = reg_robbery_rate.predict(XTrain)
Y1_Predict = scaler_y1.inverse_transform(Y1_Predict)
#for pred in Y1_Predict:
#    print '%.4f' %pred
'''


#Load the Test and generate all the crime predictions for all cities and save the predictions in  the test csv file.
df_2 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Test_FullModel.csv')
df_XTest_Year = df_2.ix[:,6] #504 X 1
df_XTest = df_2.ix[:,7:] #504 X 116

np_XTest = scaler_x.transform(df_XTest)
np_XTest_Year = df_XTest_Year.values.reshape(-1, 1)
XTest = np.hstack((np_XTest_Year, np_XTest)) #504 X 117

Y1_Predict = reg_robbery_rate.predict(XTest)
Y1_Predict = scaler_y1.inverse_transform(Y1_Predict)
for pred in Y1_Predict:
    print '%.4f' %pred
print "*********************"

Y2_Predict = reg_propertycrime_rate.predict(XTest)
Y2_Predict = scaler_y2.inverse_transform(Y2_Predict)
for pred in Y2_Predict:
    print '%.4f' %pred
print "*********************"

Y3_Predict = reg_burgalry_rate.predict(XTest)
Y3_Predict = scaler_y3.inverse_transform(Y3_Predict)
for pred in Y3_Predict:
    print '%.4f' %pred
print "*********************"

Y4_Predict = reg_larceny_rate.predict(XTest)
Y4_Predict = scaler_y4.inverse_transform(Y4_Predict)
for pred in Y4_Predict:
    print '%.4f' %pred
print "*********************"

Y5_Predict = reg_motor_rate.predict(XTest)
Y5_Predict = scaler_y5.inverse_transform(Y5_Predict)
for pred in Y5_Predict:
    print '%.4f' %pred
print "*********************"

#for pred in Y5_Predict:
#    print '%.2f' %pred

