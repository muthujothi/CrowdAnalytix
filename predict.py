import csv

data_reader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_MungedData_Test_FullModel - Predicted.csv','rb'))
header = data_reader.next() #skip the first line of the test file.

yearcity_crimedetails = {}
for data in data_reader:
    year = str(data[6])
    city = str(data[0])
    key = year+city
    yearcity_crimedetails[key] = data

data_reader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Prediction_ID.csv','rb'))
header = data_reader.next() #skip the first line of the test file.
open_file_object = csv.writer(open("C:/Pst Files/CrowdAnalytix/CrimeRates/submission_allfeatures_withsqrootofblacks.csv", "wb"))

for data in data_reader:
    pred_id = str(data[3])
    crime_type = str(data[2])
    year = str(data[1])
    city = str(data[0])
    key = year+city

    pred_row = yearcity_crimedetails[key]
    if crime_type == 'Robbery_rate':
        pred_val = pred_row[1]
    elif crime_type == 'Property_crime_rate':
        pred_val = pred_row[2]
    elif crime_type == 'Burglary_rate':
        pred_val = pred_row[3]
    elif crime_type == 'Larceny_theft_rate':
        pred_val = pred_row[4]
    elif crime_type == 'Motor_vehicle_theft_rate':
        pred_val = pred_row[5]
    else:
        print "Something wrong"

    open_file_object.writerow([pred_id, pred_val])

print "done"
