import csv
from collections import Counter
from collections import OrderedDict
from collections import defaultdict


year_crimerates = {}

year_crimerates['1Robbery_rate'] = 157.7
year_crimerates['2Robbery_rate'] = 114.55

year_crimerates['1Property_crime_rate'] = 3839.65
year_crimerates['2Property_crime_rate'] = 3220.6

year_crimerates['1Burglary_rate'] = 816.35
year_crimerates['2Burglary_rate'] = 703.8

year_crimerates['1Larceny_theft_rate'] = 2480.5
year_crimerates['2Larceny_theft_rate'] = 2220

year_crimerates['1Motor_vehicle_theft_rate'] = 371.25
year_crimerates['2Motor_vehicle_theft_rate'] = 252.45


datareader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Prediction_ID.csv','rb'))
header = datareader.next() #skip the first line

pred_year_crime = {}

for data in datareader:
    year = str(data[1])
    type_of_crime = str(data[2])
    pred = str(data[3])

    pred_year_crime[pred] = year+type_of_crime


datareader = csv.reader(open('C:/Pst Files/CrowdAnalytix/CrimeRates/CA_Crime_Rate_Submission_Format.csv','rb'))
open_file_object = csv.writer(open("C:/Pst Files/CrowdAnalytix/CrimeRates/submission_crime_yearwise_medians.csv", "wb"))
header = datareader.next() #skip the first line of the test file.

for data in datareader:
    #Get predictionID
    pred = str(data[0])
    #Get what needs to be predicted
    what_needs_tobe_predicted = pred_year_crime[pred]
    #Get the median value of what needs to be predicted
    bingo = year_crimerates[what_needs_tobe_predicted]

    open_file_object.writerow([pred,bingo])

    
print "done."
