import csv

'''
store the following hashes:
pred1- [1 city9 Robbery_rate]
pred2 - [2 city9 Robbery_rate]

the next hash
1city9 - [100702 158 114 44]
2city9- [105009 112 86 26]
'''
data_reader = csv.reader(open('C:/LearningMaterials/CrowdAnalytix/CrimeRates/CA_Prediction_ID.csv','rb'))

what_to_predict = {}

for data in data_reader:
    pred_id = str(data[3])
    what_to_predict[pred_id] = data

data_reader = csv.reader(open('C:/LearningMaterials/CrowdAnalytix/CrimeRates/CA_Crime_Rate_Test.csv','rb'))

city_details = {}

for data in data_reader:
    year = str(data[1])
    city = str(data[0])
    city_details[year+city] = data


key = what_to_predict['Pred_22'][1]+what_to_predict['Pred_22'][0]
print key
x_test = city_details[key]
print x_test
print int(x_test[1])
print int(x_test[7])
