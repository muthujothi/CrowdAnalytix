CrowdAnalytix - Modeling: Predict Crime Rate of US Cities
==========================================================

Code that produces an RMSE error of 613.4665 in the US cities crime rates predictive modelling contest
conducted by CrowdAnalytix between 14-Jul-2014 and 28-Jul-2014.

Ranked #6 among the #318 competitors using the ensembled models as described below.

My model was an ensemble of two,

1. Model 1 consisted of specific features handpicked for each type of crime based on correlation analysis between the crime and all the input variables. For example, for robbery I ended up having year, black race, percent of no_vehicles, incomelessthan10k, percent of 2 vehicle and square root of percent of black race. Similarly other subsets for other crimes (there were some overlap though)
All these features were centered and scaled by subtracting from mean and dividing by std.
Trained 5 linear models with regularization for 5 types of crimes based specific subset of features as selected above. (RMSE about 662)
(selectivefeatures_crimerates.py)

2. Model 2 consisted of all features. But all the features were log transformed to resolve the skewness and centered and scaled. Again they were trained on linear model with regularization.  (RMSE about 652). The output variables in both M1 and M2 were centered and scaled.
(select_fullfeatures_crimerates.py)

3. Finally i averaged out both the predictions from M1 and M2 to get the final prediction. Also the negative values of the final prediction (around 2 - 3) were scaled up to the minimum crime rate of that type. 
This gave a final RMSE of around 613.

The other files go by their name which were employed for the purpose of performing statistical exploratory analysis before building the
actual models.


