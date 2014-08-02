import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
#GBM model performed bad with single independentvariable
from sklearn.metrics import mean_squared_error

#Load the train data
df_1 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Train.csv')

df_robbery_rate = df_1.ix[:,2]
mean_robbery_rate = np.mean(df_robbery_rate)
#print mean_robbery_rate
#print np.median(df_robbery_rate)
std_robbery_rate = np.std(df_robbery_rate)
df_robbery_rate = (df_robbery_rate - mean_robbery_rate )/(std_robbery_rate)
np_robbery_rate = df_robbery_rate.values.reshape(-1, 1)



df_propertycrime_rate = df_1.ix[:,3]
mean_propertycrime_rate = np.mean(df_propertycrime_rate)
std_propertycrime_rate = np.std(df_propertycrime_rate)
df_propertycrime_rate = (df_propertycrime_rate - mean_propertycrime_rate)/(std_propertycrime_rate)
np_propertycrime_rate = df_propertycrime_rate.values.reshape(-1, 1)

df_burgalry_rate = df_1.ix[:,4]
mean_burgalry_rate = np.mean(df_burgalry_rate)
std_burgalry_rate = np.std(df_burgalry_rate)
df_burgalry_rate = (df_burgalry_rate - mean_burgalry_rate)/(std_burgalry_rate)
np_burgalry_rate = df_burgalry_rate.values.reshape(-1, 1)

df_larceny_rate = df_1.ix[:,5]
mean_larceny_rate = np.mean(df_larceny_rate)
std_larceny_rate = np.std(df_larceny_rate)
df_larceny_rate = (df_larceny_rate - mean_larceny_rate)/(std_larceny_rate)
np_larceny_rate = df_larceny_rate.values.reshape(-1, 1)

df_motortheft_rate = df_1.ix[:,6]
mean_motortheft_rate = np.mean(df_motortheft_rate)
std_motortheft_rate = np.std(df_motortheft_rate)
df_motortheft_rate = (df_motortheft_rate - mean_motortheft_rate)/(std_motortheft_rate)
np_motortheft_rate = df_motortheft_rate.values.reshape(-1, 1)

##### High Positive Correlated features for Robbery ####
df_blacks = df_1.ix[:,102]
mean_blacks = np.mean(df_blacks)
std_blacks = np.std(df_blacks)
df_blacks = (df_blacks - mean_blacks)/(std_blacks)
np_blacks = df_blacks.values.reshape(-1, 1)

df_novehicles = df_1.ix[:,84]
mean_novehicles = np.mean(df_novehicles)
std_novehicles = np.std(df_novehicles)
df_novehicles = (df_novehicles - mean_novehicles)/(std_novehicles)
np_novehicles = df_novehicles.values.reshape(-1, 1)

df_incomelessthan10k = df_1.ix[:,45]
mean_incomelessthan10k = np.mean(df_incomelessthan10k)
std_incomelessthan10k = np.std(df_incomelessthan10k)
df_incomelessthan10k = (df_incomelessthan10k - mean_incomelessthan10k)/(std_incomelessthan10k)
np_incomelessthan10k = df_incomelessthan10k.values.reshape(-1, 1)

df_nevermarried = df_1.ix[:,86]
mean_nevermarried = np.mean(df_nevermarried)
std_nevermarried = np.std(df_nevermarried)
df_nevermarried = (df_nevermarried - mean_nevermarried)/(std_nevermarried)
np_nevermarried = df_nevermarried.values.reshape(-1, 1)
##### High Positive Correlated features for Robbery ####

##### High Positive+Negative Correlated features for Property Crime ####
#First incomelessthan10K

df_occupiedhousing = df_1.ix[:,72]
mean_occupiedhousing = np.mean(df_occupiedhousing)
std_occupiedhousing = np.std(df_occupiedhousing)
df_occupiedhousing = (df_occupiedhousing - mean_occupiedhousing)/(std_occupiedhousing)
np_occupiedhousing = df_occupiedhousing.values.reshape(-1, 1)

df_medianfamilyincome = df_1.ix[:,57]
mean_medianfamilyincome = np.mean(df_medianfamilyincome)
std_medianfamilyincome = np.std(df_medianfamilyincome)
df_medianfamilyincome = (df_medianfamilyincome - mean_medianfamilyincome)/(std_medianfamilyincome)
np_medianfamilyincome = df_medianfamilyincome.values.reshape(-1, 1)
##### High Positive Correlated features for Property Crime ####

##### High Positive Correlated features for Burgalry ####
#First incomelessthan10K
#second occupied housing units
#third percent of black race
#fourth median family income
df_nowmarried = df_1.ix[:,95]
mean_nowmarried = np.mean(df_nowmarried)
std_nowmarried = np.std(df_nowmarried)
df_nowmarried = (df_nowmarried - mean_nowmarried)/(std_nowmarried)
np_nowmarried = df_nowmarried.values.reshape(-1, 1)
##### High Positive Correlated features for Burgalry ####

##### High Positive Correlated features for Larceny Theft ####
df_1personhouse = df_1.ix[:,116]
mean_1personhouse = np.mean(df_1personhouse)
std_1personhouse = np.std(df_1personhouse)
df_1personhouse = (df_1personhouse - mean_1personhouse)/(std_1personhouse)
np_1personhouse = df_1personhouse.values.reshape(-1, 1)

df_1vehicle = df_1.ix[:,85]
mean_1vehicle = np.mean(df_1vehicle)
std_1vehicle = np.std(df_1vehicle)
df_1vehicle = (df_1vehicle - mean_1vehicle)/(std_1vehicle)
np_1vehicle = df_1vehicle.values.reshape(-1, 1)

#Third income less than 10K dollars
##### High Positive Correlated features for Larceny Theft ####

##### High Positive Correlated features for Motor Theft ####
df_nodiploma = df_1.ix[:,66]
mean_nodiploma = np.mean(df_nodiploma)
std_nodiploma = np.std(df_nodiploma)
df_nodiploma = (df_nodiploma - mean_nodiploma)/(std_nodiploma)
np_nodiploma = df_nodiploma.values.reshape(-1, 1)

df_whiterace = df_1.ix[:,101]#101 is percent of white race, #7 is the population
mean_whiterace = np.mean(df_whiterace)
std_whiterace = np.std(df_whiterace)
df_whiterace = (df_whiterace - mean_whiterace)/(std_whiterace)
np_whiterace = df_whiterace.values.reshape(-1, 1)

df_2vehicle = df_1.ix[:,86]
mean_2vehicle = np.mean(df_2vehicle)
std_2vehicle = np.std(df_2vehicle)
df_2vehicle = (df_2vehicle - mean_2vehicle)/(std_2vehicle)
np_2vehicle = df_2vehicle.values.reshape(-1, 1)
##### High Positive Correlated features for Motor Theft ####

df_year = df_1.ix[:,1]
np_year = df_year.values.reshape(-1, 1)

#X_Robbery = np.hstack((np_year, np_blacks, np_novehicles, np_incomelessthan10k, np_nevermarried))
X_Robbery = np.hstack((np_year, np_blacks))
#X_Property = np.hstack((np_year, np_incomelessthan10k, np_occupiedhousing, np_medianfamilyincome))
X_Property = np.hstack((np_year, np_occupiedhousing))
X_Burgalry = np.hstack((np_year, np_incomelessthan10k, np_occupiedhousing, np_blacks, np_medianfamilyincome, np_nowmarried))
X_Larceny = np.hstack((np_year, np_1personhouse, np_1vehicle, np_incomelessthan10k))
X_Motor = np.hstack((np_year, np_nodiploma, np_whiterace, np_2vehicle))

y1 = np_robbery_rate
y2 = np_propertycrime_rate
y3 = np_burgalry_rate
y4 = np_larceny_rate
y5 = np_motortheft_rate

#tuned_parameters = [{'n_estimators': [100, 200, 500, 750], 'learning_rate': [0.01, 0.1, 1.0], 'min_samples_split': [1, 2]}]
reg_robbery_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
reg_robbery_rate.fit(X_Robbery, y1)
#reg_robbery_rate = GridSearchCV( GradientBoostingRegressor(loss='ls'), tuned_parameters, cv = 3, verbose = 2 ).fit(X_Robbery, y1)
#reg_robbery_rate = GradientBoostingRegressor(loss='ls', alpha=0.9, n_estimators=250, max_depth=3, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)

reg_propertycrime_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
reg_propertycrime_rate.fit(X_Property, y2)
#reg_propertycrime_rate = GridSearchCV( GradientBoostingRegressor(loss='ls'), tuned_parameters, cv = 3, verbose = 2 ).fit(X_Property, y2)


reg_burgalry_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
reg_burgalry_rate.fit(X_Burgalry, y3)
#reg_burgalry_rate = GridSearchCV( GradientBoostingRegressor(loss='ls'), tuned_parameters, cv = 3, verbose = 2 ).fit(X_Burgalry, y3)


reg_larceny_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
reg_larceny_rate.fit(X_Larceny, y4)
#reg_larceny_rate = GridSearchCV( GradientBoostingRegressor(loss='ls'), tuned_parameters, cv = 3, verbose = 2 ).fit(X_Larceny, y4)


reg_motortheft_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
reg_motortheft_rate.fit(X_Motor, y5)
#reg_motortheft_rate = GridSearchCV( GradientBoostingRegressor(loss='ls'), tuned_parameters, cv = 3, verbose = 2 ).fit(X_Motor, y5)


data_reader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Train.csv','rb'))
header = data_reader.next() #skip the first line of the test file.
predicted_rob = []

for data in data_reader:
    x = []

    year = int(data[1])
    x.append(year) 

    x_test = np.asarray(x)
    '''
    blacks = float (data[102])
    normalized_blacks = (blacks - mean_blacks)/(std_blacks)
    x_test = np.append(x_test, normalized_blacks) 
    
    novehicles = float(data[84])
    normalized_novehicles = (novehicles - mean_novehicles)/(std_novehicles)
    x_test = np.append(x_test, normalized_novehicles)
    
    incomelessthan10k = float(data[45])
    normalized_incomelessthan10k = (incomelessthan10k - mean_incomelessthan10k)/(std_incomelessthan10k)
    x_test = np.append(x_test, normalized_incomelessthan10k)
    
    nevermarried = float(data[86])
    normalized_nevermarried = (nevermarried - mean_nevermarried)/(std_nevermarried)
    x_test = np.append(x_test, normalized_nevermarried)
    '''

    '''
    incomelessthan10k = float(data[45])
    normalized_incomelessthan10k = (incomelessthan10k - mean_incomelessthan10k)/(std_incomelessthan10k)
    x_test = np.append(x_test, normalized_incomelessthan10k)
    '''    
    occupiedhousing = float(data[72])
    normalized_occupiedhousing = (occupiedhousing - mean_occupiedhousing)/(std_occupiedhousing)
    x_test = np.append(x_test, normalized_occupiedhousing)
    '''
    medianfamilyincome = float(data[57])
    normalized_medianfamilyincome = (medianfamilyincome - mean_medianfamilyincome)/(std_medianfamilyincome)
    x_test = np.append(x_test, normalized_medianfamilyincome)
    '''
    pred_crime_rate = reg_propertycrime_rate.predict(x_test)[0]
    pred_crime_rate = round(((pred_crime_rate*std_propertycrime_rate)+ (mean_propertycrime_rate)), 1)

    predicted_rob.append((round(pred_crime_rate, 3)))

#for rob in predicted_rob:
#    print rob

predicted_robbery_rate = np.asarray(predicted_rob)
print mean_squared_error(y2, predicted_robbery_rate)






