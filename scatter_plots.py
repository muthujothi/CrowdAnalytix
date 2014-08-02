import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv
from scipy.stats.stats import pearsonr

#Load the train data
df_1 = pd.read_csv('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_Train.csv')

df_robbery_rate = df_1.ix[:,6]
y1 = df_robbery_rate.values.reshape(-1, 1)

#(60)Percent; PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL - All families
#(100)Unemployment rate in Population 16 years and over,
#Higher the number of whites, lesser the crime  - (101) Percent RACE  - White #robbery = 0.5, prop_crime= 0.13, burgalry=0.25, larceny=0.04, motor=0.45
#property crime and larceny seems to be spread everywhere irrespective of white race

#property_crime = {percent of occupied houses:  -0.434 (72)}, {percent of vanact houses: 0.434}, {percent of rental vacancy rate - +ve}
#{mobile homes - not good feature}, {mobile boat - poor feature}, {percent of 4 bedroom - -0.38}
df_feature = df_1.ix[:,102]
X = df_feature.values.reshape(-1, 1)

#larceny_theft - {}
#motor_thefy - {(86)Percent; VEHICLES AVAILABLE - 2 vehicles available - -0.415}
#below poverty level - robbery : 0.46, property:0.36 , burgalry:0.47, larceny:0.22, motor: 0.34

plt.scatter(X, y1)
plt.show()

print np.mean(X)
print np.mean(y1)
print pearsonr(X, y1)[0][0]

