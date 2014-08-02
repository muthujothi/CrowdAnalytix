import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import GradientBoostingRegressor
#GBM model performed bad with single independentvariable

#Load the train data
df_1 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_Train.csv')

df_robbery_rate = df_1.ix[:,2]
mean_robbery_rate = np.mean(df_robbery_rate)
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


df_population = df_1.ix[:,7]#101 is percent of white race, #7 is the population
mean_population = np.mean(df_population)
std_population = np.std(df_population)
df_population = (df_population - mean_population)/(std_population)
np_population = df_population.values.reshape(-1, 1)

df_whiterace = df_1.ix[:,101]#101 is percent of white race, #7 is the population
mean_whiterace = np.mean(df_whiterace)
std_whiterace = np.std(df_whiterace)
df_whiterace = (df_whiterace - mean_whiterace)/(std_whiterace)
np_whiterace = df_whiterace.values.reshape(-1, 1)

df_occupiedhousing = df_1.ix[:,72]#101 is percent of white race, #7 is the population
mean_occupiedhousing = np.mean(df_occupiedhousing)
std_occupiedhousing = np.std(df_occupiedhousing)
df_occupiedhousing = (df_occupiedhousing - mean_occupiedhousing)/(std_occupiedhousing)
np_occupiedhousing = df_occupiedhousing.values.reshape(-1, 1)

df_belowpoverty = df_1.ix[:,60]#101 is percent of white race, #7 is the population
mean_belowpoverty = np.mean(df_belowpoverty)
std_belowpoverty = np.std(df_belowpoverty)
df_belowpoverty = (df_belowpoverty - mean_belowpoverty)/(std_belowpoverty)
np_belowpoverty = df_belowpoverty.values.reshape(-1, 1)

#This decreased the score - see if selective adding is required
df_labforce = df_1.ix[:,27]#101 is percent of white race, #7 is the population
mean_labforce = np.mean(df_labforce)
std_labforce = np.std(df_labforce)
df_labforce = (df_labforce - mean_labforce)/(std_labforce)
np_labforce = df_labforce.values.reshape(-1, 1)

df_blacks = df_1.ix[:,102]#101 is percent of white race, #7 is the population
mean_blacks = np.mean(df_blacks)
std_blacks = np.std(df_blacks)
df_blacks = (df_blacks - mean_blacks)/(std_blacks)
np_blacks = df_blacks.values.reshape(-1, 1)


df_year = df_1.ix[:,1]
np_year = df_year.values.reshape(-1, 1)


#X = np.hstack((np_year, np_population, np_whiterace, np_belowpoverty, np_labforce, np_blacks))
#X_PropertyCrime = np.hstack((np_year, np_population, np_whiterace, np_belowpoverty, np_labforce, np_blacks, np_occupiedhousing))
X = np.hstack((np_year, np_population, np_whiterace, np_belowpoverty))
X_PropertyCrime = np.hstack((np_year, np_population, np_whiterace, np_belowpoverty, np_occupiedhousing))
X_RobberyBurgalryCrime = np.hstack((np_year, np_population, np_whiterace, np_belowpoverty, np_blacks))
#X = np_population
y1 = np_robbery_rate
y2 = np_propertycrime_rate
y3 = np_burgalry_rate
y4 = np_larceny_rate
y5 = np_motortheft_rate

reg_robbery_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_robbery_rate = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=250, max_depth=3, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
reg_robbery_rate.fit(X_RobberyBurgalryCrime, y1)

reg_propertycrime_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_propertycrime_rate = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=250, max_depth=3, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
reg_propertycrime_rate.fit(X_PropertyCrime, y2)

reg_burgalry_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_burgalry_rate = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=250, max_depth=3, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
reg_burgalry_rate.fit(X_RobberyBurgalryCrime, y3)

reg_larceny_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_larceny_rate = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=250, max_depth=3, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
reg_larceny_rate.fit(X, y4)

reg_motortheft_rate = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True, cv = 3)
#reg_motortheft_rate = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=250, max_depth=3, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
reg_motortheft_rate.fit(X, y5)


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

data_reader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_Test.csv','rb'))
header = data_reader.next() #skip the first line of the test file.

city_details = {}

for data in data_reader:
    year = str(data[1])
    city = str(data[0])
    city_details[year+city] = data

data_reader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_Submission_Format.csv','rb'))
open_file_object = csv.writer(open("C:/Pst Files/CrowdAnalytix/CrimeRates/submission_linearmodel_y+p+wh+bp+sel+bl+oc.csv", "wb"))
header = data_reader.next() #skip the first line of the test file.

for data in data_reader:
    pred_id = str(data[0])#pred_1
    ques_asked = what_to_predict[pred_id]
    key = ques_asked[1]+ques_asked[0]
    print key
    details = city_details[key]
    x = []

    x.append(int(details[1])) #year

    population = int(details[7])#since it is population
    normalized_population = (population - mean_population)/(std_population)    
    x.append(normalized_population)

    whiterace = float(details[101])#since it is percent of white race
    normalized_whiterace = (whiterace - mean_whiterace)/(std_whiterace)
    x.append(normalized_whiterace) 

    belowpoverty = float(details[60])#since it is percent of belowpoverty
    normalized_belowpoverty = (belowpoverty - mean_belowpoverty)/(std_belowpoverty)
    x.append(normalized_belowpoverty) 

    labforce = float(details[27])#since it is percent of labforce
    normalized_labforce = (labforce - mean_labforce)/(std_labforce)
    #x.append(normalized_labforce) 

    x_test = np.asarray(x)
    crime_to_predict = ques_asked[2]
    pred_crime_rate = 1
    if crime_to_predict == 'Robbery_rate':
        blacks = float(details[102])#since it is percent of blacks
        normalized_blacks = (blacks - mean_blacks)/(std_blacks)
        x_test = np.append(x_test, normalized_blacks)
        pred_crime_rate = reg_robbery_rate.predict(x_test)[0]
        pred_crime_rate = round(((pred_crime_rate*std_robbery_rate)+ (mean_robbery_rate)), 1)
    elif crime_to_predict == 'Property_crime_rate':
        occupiedhousing = float(details[72])#since it is percent of occupied housing
        normalized_occupiedhousing = (occupiedhousing - mean_occupiedhousing)/(std_occupiedhousing)
        x_test = np.append(x_test, normalized_occupiedhousing)
        pred_crime_rate = reg_propertycrime_rate.predict(x_test)[0]
        pred_crime_rate = round(((pred_crime_rate*std_propertycrime_rate)+ (mean_propertycrime_rate)), 1)
    elif crime_to_predict == 'Burglary_rate':
        blacks = float(details[102])#since it is percent of blacks
        normalized_blacks = (blacks - mean_blacks)/(std_blacks)
        x_test = np.append(x_test, normalized_blacks)
        pred_crime_rate = reg_burgalry_rate.predict(x_test)[0]
        pred_crime_rate = round(((pred_crime_rate*std_burgalry_rate)+ (mean_burgalry_rate)), 1)
    elif crime_to_predict == 'Larceny_theft_rate':
        pred_crime_rate = reg_larceny_rate.predict(x_test)[0]
        pred_crime_rate = round(((pred_crime_rate*std_larceny_rate)+ (mean_larceny_rate)), 1)
    elif crime_to_predict == 'Motor_vehicle_theft_rate':
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
