'''
This code gets an accuracy of 662 on the leaderboard.
'''
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

#Load the train data
df_1 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Train.csv')

df_robbery_rate = df_1.ix[:,2]
mean_robbery_rate = np.mean(df_robbery_rate)
std_robbery_rate = np.std(df_robbery_rate)
df_robbery_rate = (df_robbery_rate - mean_robbery_rate )/(std_robbery_rate)
#df_robbery_rate = np.log(1+df_robbery_rate)
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

df_sqrtblacks = np.sqrt(df_1.ix[:,-3])
mean_sqrtblacks = np.mean(df_sqrtblacks)
std_sqrtblacks = np.std(df_sqrtblacks)
df_sqrtblacks = (df_sqrtblacks - mean_sqrtblacks)/(std_sqrtblacks)
np_sqrtblacks = df_sqrtblacks.values.reshape(-1, 1)

df_sqrtnovehicles = np.sqrt(df_1.ix[:,-2])
mean_sqrtnovehicles = np.mean(df_sqrtnovehicles)
std_sqrtnovehicles = np.std(df_sqrtnovehicles)
df_sqrtnovehicles = (df_sqrtnovehicles - mean_sqrtnovehicles)/(std_sqrtnovehicles)
np_sqrtnovehicles = df_sqrtnovehicles.values.reshape(-1, 1)

df_sqrtincomelessthan10k = np.sqrt(df_1.ix[:,-1])
mean_sqrtincomelessthan10k = np.mean(df_sqrtincomelessthan10k)
std_sqrtincomelessthan10k = np.std(df_sqrtincomelessthan10k)
df_sqrtincomelessthan10k = (df_sqrtincomelessthan10k - mean_sqrtincomelessthan10k)/(std_sqrtincomelessthan10k)
np_sqrtincomelessthan10k = df_sqrtincomelessthan10k.values.reshape(-1, 1)

df_blacksqr = np.square(df_1.ix[:,102])
mean_blacksqr = np.mean(df_blacksqr)
std_blacksqr = np.std(df_blacksqr)
df_blacksqr = (df_blacksqr - mean_blacksqr)/(std_blacksqr)
np_blacksqr = df_blacksqr.values.reshape(-1, 1)
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

df_sqroccupiedhousing = np.square(df_1.ix[:,72])
mean_sqroccupiedhousing = np.mean(df_sqroccupiedhousing)
std_sqroccupiedhousing = np.std(df_sqroccupiedhousing)
df_sqroccupiedhousing = (df_sqroccupiedhousing - mean_sqroccupiedhousing)/(std_sqroccupiedhousing)
np_sqroccupiedhousing = df_sqroccupiedhousing.values.reshape(-1, 1)

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

df_income10k = df_1.ix[:,46]
mean_income10k = np.mean(df_income10k)
std_income10k = np.std(df_income10k)
df_income10k = (df_income10k - mean_income10k)/(std_income10k)
np_income10k = df_income10k.values.reshape(-1, 1)

df_income15k = df_1.ix[:,47]
mean_income15k = np.mean(df_income15k)
std_income15k = np.std(df_income15k)
df_income15k = (df_income15k - mean_income15k)/(std_income15k)
np_income15k = df_income15k.values.reshape(-1, 1)

df_percapita = df_1.ix[:,59]
mean_percapita = np.mean(df_percapita)
std_percapita = np.std(df_percapita)
df_percapita = (df_percapita - mean_percapita)/(std_percapita)
np_percapita = df_percapita.values.reshape(-1, 1)
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

df_divorced = df_1.ix[:,97]
mean_divorced = np.mean(df_divorced)
std_divorced = np.std(df_divorced)
df_divorced = (df_divorced - mean_divorced)/(std_divorced)
np_divorced = df_divorced.values.reshape(-1, 1)

df_rent1k = df_1.ix[:,93]
mean_rent1k = np.mean(df_rent1k)
std_rent1k = np.std(df_rent1k)
df_rent1k = (df_rent1k - mean_rent1k)/(std_rent1k)
np_rent1k = df_rent1k.values.reshape(-1, 1)

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

df_unemploymentrate = df_1.ix[:,100]
mean_unemploymentrate = np.mean(df_unemploymentrate)
std_unemploymentrate = np.std(df_unemploymentrate)
df_unemploymentrate = (df_unemploymentrate - mean_unemploymentrate)/(std_unemploymentrate)
np_unemploymentrate = df_unemploymentrate.values.reshape(-1, 1)

df_industrytrans = df_1.ix[:,60]
mean_industrytrans = np.mean(df_industrytrans)
std_industrytrans = np.std(df_industrytrans)
df_industrytrans = (df_industrytrans - mean_industrytrans)/(std_industrytrans)
np_industrytrans = df_industrytrans.values.reshape(-1, 1)

df_bachelors = df_1.ix[:,70]
mean_bachelors = np.mean(df_bachelors)
std_bachelors = np.std(df_bachelors)
df_bachelors = (df_bachelors - mean_bachelors)/(std_bachelors)
np_bachelors = df_bachelors.values.reshape(-1, 1)
##### High Positive Correlated features for Motor Theft ####

df_year = df_1.ix[:,1]
np_year = df_year.values.reshape(-1, 1)

#This corresponds to RMSE score of 662.176. The one before is 662.0522
#X_Robbery = np.hstack((np_year, np_blacks, np_novehicles, np_incomelessthan10k, np_sqrtblacks, np_sqrtnovehicles, np_sqrtincomelessthan10k))
#X_Robbery = np.hstack((np_year, np_blacks, np_novehicles, np_incomelessthan10k, np_sqrtblacks, np_sqrtnovehicles, np_sqrtincomelessthan10k))
X_Robbery = np.hstack((np_year, np_blacks, np_novehicles, np_incomelessthan10k, np_2vehicle, np_sqrtblacks))
X_Property = np.hstack((np_year, np_incomelessthan10k, np_occupiedhousing, np_medianfamilyincome))
#X_Property = np.hstack((np_year, np_occupiedhousing))
X_Burgalry = np.hstack((np_year, np_incomelessthan10k, np_occupiedhousing, np_blacks, np_medianfamilyincome, np_nowmarried, np_income10k, np_percapita))
#X_Burgalry = np.hstack((np_year, np_blacks))
X_Larceny = np.hstack((np_year, np_1personhouse, np_1vehicle, np_incomelessthan10k, np_divorced))
#X_Larceny = np.hstack((np_year, np_1personhouse, np_1vehicle))
X_Motor = np.hstack((np_year, np_nodiploma, np_whiterace, np_2vehicle, np_unemploymentrate))

y1 = np_robbery_rate
y2 = np_propertycrime_rate
y3 = np_burgalry_rate
y4 = np_larceny_rate
y5 = np_motortheft_rate

#tuned_parameters = [{'n_estimators': [100, 200, 500, 750], 'learning_rate': [0.01, 0.1, 1.0], 'min_samples_split': [1, 2]}]

reg_robbery_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
reg_robbery_rate.fit(X_Robbery, y1)

#reg_robbery_rate = linear_model.SGDRegressor()
#tuned_parameters = [{'C': [0.001, 0.1, 1.0, 10.0]}]
#reg_robbery_rate = linear_model.Lasso(alpha=0.1)
#reg_robbery_rate.fit(X_Robbery, y1)
#reg_robbery_rate = GridSearchCV( GradientBoostingRegressor(loss='ls'), tuned_parameters, cv = 3, verbose = 2 ).fit(X_Robbery, y1)
#reg_robbery_rate = GradientBoostingRegressor(loss='ls', alpha=0.9, n_estimators=250, max_depth=3, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
#reg_robbery_rate.fit(X_Robbery, y1)

#reg_propertycrime_rate_1 = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_propertycrime_rate_1.fit(X_Property, y2)
reg_propertycrime_rate = linear_model.SGDRegressor()
reg_propertycrime_rate.fit(X_Property, y2)
#reg_propertycrime_rate = GridSearchCV( GradientBoostingRegressor(loss='ls'), tuned_parameters, cv = 3, verbose = 2 ).fit(X_Property, y2)


#reg_burgalry_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
reg_burgalry_rate = linear_model.SGDRegressor()
reg_burgalry_rate.fit(X_Burgalry, y3)
#reg_burgalry_rate = AdaBoostRegressor.fit(X_Burgalry, y3)


reg_larceny_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_larceny_rate = linear_model.SGDRegressor()
reg_larceny_rate.fit(X_Larceny, y4)
#reg_larceny_rate = GridSearchCV( GradientBoostingRegressor(loss='ls'), tuned_parameters, cv = 3, verbose = 2 ).fit(X_Larceny, y4)


reg_motortheft_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_motortheft_rate = linear_model.SGDRegressor()
reg_motortheft_rate.fit(X_Motor, y5)
#reg_motortheft_rate = GridSearchCV( GradientBoostingRegressor(loss='ls'), tuned_parameters, cv = 3, verbose = 2 ).fit(X_Motor, y5)



'''
store the following hashes:
pred1- [1 city9 Robbery_rate]
pred2 - [2 city9 Robbery_rate]

the next hash
1city9 - [100702 158 114 44]
2city9- [105009 112 86 26]
'''
data_reader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Prediction_ID.csv','rb'))
header = data_reader.next() #skip the first line of the test file.

what_to_predict = {}

for data in data_reader:
    pred_id = str(data[3])
    what_to_predict[pred_id] = data

data_reader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Test.csv','rb'))
header = data_reader.next() #skip the first line of the test file.

city_details = {}

for data in data_reader:
    year = str(data[1])
    city = str(data[0])
    city_details[year+city] = data

data_reader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_Submission_Format.csv','rb'))
open_file_object = csv.writer(open("C:/Pst Files/CrowdAnalytix/CrimeRates/submission_linearmodel_selefts_svr.csv", "wb"))
header = data_reader.next() #skip the first line of the test file.

for data in data_reader:
    pred_id = str(data[0])#pred_1
    ques_asked = what_to_predict[pred_id]
    key = ques_asked[1]+ques_asked[0]
    #print key
    details = city_details[key]

    x = []
    x.append(int(details[1])) #year
  
    x_test = np.asarray(x)

    incomelessthan10k = float(details[45])
    normalized_incomelessthan10k = (incomelessthan10k - mean_incomelessthan10k)/(std_incomelessthan10k)
    
    crime_to_predict = ques_asked[2]
    pred_crime_rate = 1

    if crime_to_predict == 'Robbery_rate':
        
        blacks = float(details[102])
        normalized_blacks = (blacks - mean_blacks)/(std_blacks)
        x_test = np.append(x_test, normalized_blacks)

        
        novehicles = float(details[84])
        normalized_novehicles = (novehicles - mean_novehicles)/(std_novehicles)
        x_test = np.append(x_test, normalized_novehicles)
        
        x_test = np.append(x_test, normalized_incomelessthan10k)
        '''
        sqrtnovehicles = np.sqrt(float(details[-2]))
        normalized_sqrtnovehicles = (sqrtnovehicles - mean_sqrtnovehicles)/(std_sqrtnovehicles)
        x_test = np.append(x_test, normalized_sqrtnovehicles)
        
        sqrtincomelessthan10k = np.sqrt(float(details[-1]))
        normalized_sqrtincomelessthan10k = (sqrtincomelessthan10k - mean_sqrtincomelessthan10k)/(std_sqrtincomelessthan10k)
        x_test = np.append(x_test, normalized_sqrtincomelessthan10k)
        '''
        twovehicle = float(details[86])
        normalized_2vehicle = (twovehicle - mean_2vehicle)/(std_2vehicle)
        x_test = np.append(x_test, normalized_2vehicle)
        '''
        industrytrans = float(details[60])
        normalized_industrytrans = (industrytrans - mean_industrytrans)/(std_industrytrans)
        x_test = np.append(x_test, normalized_industrytrans)
        '''
        sqrtblacks = np.sqrt(float(details[-3]))
        normalized_sqrtblacks = (sqrtblacks - mean_sqrtblacks)/(std_sqrtblacks)
        x_test = np.append(x_test, normalized_sqrtblacks)
        '''
        #np.square(df_1.ix[:,102])#np_blacksqr
        '''
        '''
        blacksqr = blacks*blacks
        normalized_blacksqr = (blacksqr - mean_blacksqr)/(std_blacksqr)
        x_test = np.append(x_test, normalized_blacksqr)
        '''
        
        pred_crime_rate = reg_robbery_rate.predict(x_test)[0]
        pred_crime_rate = round(((pred_crime_rate*std_robbery_rate)+ (mean_robbery_rate)), 1)
        #pred_crime_rate = round(np.exp(pred_crime_rate), 1) - 1
        #pred_crime_rate = round((1.25 * pred_crime_rate), 1)
    elif crime_to_predict == 'Property_crime_rate':
        x_test = np.append(x_test, normalized_incomelessthan10k)
        
        occupiedhousing = float(details[72])
        normalized_occupiedhousing = (occupiedhousing - mean_occupiedhousing)/(std_occupiedhousing)
        x_test = np.append(x_test, normalized_occupiedhousing)
        
        medianfamilyincome = float(details[57])
        normalized_medianfamilyincome = (medianfamilyincome - mean_medianfamilyincome)/(std_medianfamilyincome)
        x_test = np.append(x_test, normalized_medianfamilyincome)

        '''
        income100k = float(details[52])
        normalized_income100k = (income100k - mean_income100k)/(std_income100k)
        x_test = np.append(x_test, income100k)
        
        sqroccupiedhousing = occupiedhousing*occupiedhousing
        normalized_sqroccupiedhousing = (sqroccupiedhousing - mean_sqroccupiedhousing)/(std_sqroccupiedhousing)
        x_test = np.append(x_test, normalized_sqroccupiedhousing)
        '''
        pred_crime_rate = reg_propertycrime_rate.predict(x_test)[0]
        pred_crime_rate = round(((pred_crime_rate*std_propertycrime_rate)+ (mean_propertycrime_rate)), 1)
        '''
        pred_crime_rate_1 = reg_propertycrime_rate.predict(x_test)[0]
        pred_crime_rate_1 = round(((pred_crime_rate_1*std_propertycrime_rate)+ (mean_propertycrime_rate)), 1)

        pred_crime_rate_2 = reg_propertycrime_rate_1.predict(x_test)[0]
        pred_crime_rate_2 = round(((pred_crime_rate_2*std_propertycrime_rate)+ (mean_propertycrime_rate)), 1)
        pred_crime_rate = round((pred_crime_rate_1 + pred_crime_rate_2)/2, 1)
        '''
        #pred_crime_rate = round((0.95 * pred_crime_rate), 1)

    elif crime_to_predict == 'Burglary_rate':
        
        x_test = np.append(x_test, normalized_incomelessthan10k)
        
        occupiedhousing = float(details[72])
        normalized_occupiedhousing = (occupiedhousing - mean_occupiedhousing)/(std_occupiedhousing)
        x_test = np.append(x_test, normalized_occupiedhousing)
        
        blacks = float(details[102])
        normalized_blacks = (blacks - mean_blacks)/(std_blacks)
        x_test = np.append(x_test, normalized_blacks)
        
        medianfamilyincome = float(details[57])
        normalized_medianfamilyincome = (medianfamilyincome - mean_medianfamilyincome)/(std_medianfamilyincome)
        x_test = np.append(x_test, normalized_medianfamilyincome)

        nowmarried = float(details[95])
        normalized_nowmarried = (nowmarried - mean_nowmarried)/(std_nowmarried)
        x_test = np.append(x_test, normalized_nowmarried)

        income10k = float(details[46])
        normalized_income10k = (income10k - mean_income10k)/(std_income10k)
        x_test = np.append(x_test, normalized_income10k)

        percapita = float(details[59])
        normalized_percapita = (percapita - mean_percapita)/(std_percapita)
        x_test = np.append(x_test, normalized_percapita)
        
        pred_crime_rate = reg_burgalry_rate.predict(x_test)[0]
        #print pred_crime_rate*std_burgalry_rate, mean_burgalry_rate
        pred_crime_rate = round(((pred_crime_rate*std_burgalry_rate)+ (mean_burgalry_rate)), 1)
        #pred_crime_rate = round(pred_crime_rate, 1)

    elif crime_to_predict == 'Larceny_theft_rate':
        onepersonhouse = float(details[116])
        normalized_1personhouse = (onepersonhouse - mean_1personhouse)/(std_1personhouse)
        x_test = np.append(x_test, normalized_1personhouse)

        onevehicle = float(details[85])
        normalized_1vehicle = (onevehicle - mean_1vehicle)/(std_1vehicle)
        x_test = np.append(x_test, normalized_1vehicle)

        x_test = np.append(x_test, normalized_incomelessthan10k)

        divorced = float(details[97])
        normalized_divorced = (divorced - mean_divorced)/(std_divorced)
        x_test = np.append(x_test, normalized_divorced)
        '''
        rent1k = float(details[93])
        normalized_rent1k = (rent1k - mean_rent1k)/(std_rent1k)
        x_test = np.append(x_test, normalized_rent1k)
        '''
        pred_crime_rate = reg_larceny_rate.predict(x_test)[0]
        pred_crime_rate = round(((pred_crime_rate*std_larceny_rate)+ (mean_larceny_rate)), 1)

    elif crime_to_predict == 'Motor_vehicle_theft_rate':
        nodiploma = float(details[66])
        normalized_nodiploma = (nodiploma - mean_nodiploma)/(std_nodiploma)
        x_test = np.append(x_test, normalized_nodiploma)

        whiterace = float(details[101])
        normalized_whiterace = (whiterace - mean_whiterace)/(std_whiterace)
        x_test = np.append(x_test, normalized_whiterace)

        twovehicle = float(details[86])
        normalized_2vehicle = (twovehicle - mean_2vehicle)/(std_2vehicle)
        x_test = np.append(x_test, normalized_2vehicle)

        #np_unemploymentrate
        unemploymentrate = float(details[100])
        normalized_unemploymentrate = (unemploymentrate - mean_unemploymentrate)/(std_unemploymentrate)
        x_test = np.append(x_test, normalized_unemploymentrate)
        '''
        industrytrans = float(details[60])
        normalized_industrytrans = (industrytrans - mean_industrytrans)/(std_industrytrans)
        x_test = np.append(x_test, normalized_industrytrans)

        bachelors = float(details[70])
        normalized_bachelors = (bachelors - mean_bachelors)/(std_bachelors)
        x_test = np.append(x_test, normalized_bachelors)
        '''
        pred_crime_rate = reg_motortheft_rate.predict(x_test)[0]
        pred_crime_rate = round(((pred_crime_rate*std_motortheft_rate)+ (mean_motortheft_rate)), 1)

    else:
        print "Something wrong. Please check."

    open_file_object.writerow([pred_id, pred_crime_rate])


print "cool dude. done."
'''
y5_predict = reg_motortheft_rate.predict(X)
y5_predict = ((y5_predict*std_motortheft_rate)+ (mean_motortheft_rate))
for val in y5_predict:
    print '%.4f' %val
'''



'''
plt.scatter(X, y1)
plt.show()
plt.scatter(X, y2)
plt.show()
plt.scatter(X, y3)
plt.show()
plt.scatter(X, y4)
plt.show()
plt.scatter(X, y5)
plt.show()

print mean_robbery_rate, mean_propertycrime_rate, mean_burgalry_rate, mean_larceny_rate, mean_motortheft_rate
'''


#X = np.hstack((np_population, np_robbery_rate))
#print X.shape
#print X


'''
np_robbery_rate = df_robbery_rate.values.reshape(-1, 1)
np_robbery_rate = (np_robbery_rate - mean_robbery_rate)/(std_robbery_rate)
'''



#plt.hist(np_robbery_rate)
#plt.show()
#for val in np_robbery_rate:
#    print '%.2f' %val




'''
np_robbery_rate = df_robbery_rate.values
np_robbery_rate = np_robbery_rate.reshape(-1, 1)
np_robbery_rate = np_robbery_rate * 10
print np_robbery_rate

mean_robbery_rate = np.mean(df_robbery_rate)

print .reshape(-1,1).shape
print mean_robbery_rate
'''
